from bisect import bisect_left as bisect

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from utils import shortest_path_exact


class UtilityInfrastructureBalancer(nn.Module):
    def __init__(self,
                 delta: float=1e-4,
                 gamma: float=1e-1,
                 utility_scale: float = -1e-2,
                 priority_rail: float = 0.5,
                 utility_gain_multiplier: float = 1.0,
                 alpha_elu: float = 1.0,
                 entropy_thresholds: list | None = None,
                 entropy_levels: list | None = None,
                 mask_thresholds: list | None = None,
                 mask_levels: list | None = None,
                 lower_cost_training_power: float | None = None,
                 lower_cost_training_period: int | None = None
                 ):
        """
        Loss function
        ---
        :param delta: Avoid division by zero.
        :param gamma: Smoothing parameter for the softmin.
               Smaller values = better accuracy (though, negligible difference).
               Larger values = smoother gradients.
        :param utility_scale: to scale utilities (e^{ax}/(e^{ax}+e^{ay}) is sensitive to a
        :param priority_rail: scaling of the distance with the new service (e.g. greater speed)
        :param utility_gain_multiplier: balance between cost and utility gain
        :param alpha_elu: lower values for the exponent function
        :param entropy_thresholds: thresholds for increasing penalties with entropy
        :param entropy_levels: penalty levels with entropy (following thresholds)
        :param mask_thresholds: thresholds for increasing penalties with mask
        :param mask_levels: penalty levels for assigning links that do not exist in the original graph (following thresholds)
        :param lower_cost_training_power: during training, scale building cost B as B/[((N-n)^+)^p], where p is the power (parameter) and N is lower_cost_period
        :param lower_cost_training_period: scaling period for lower_cost_training_power
        """
        super(UtilityInfrastructureBalancer, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.utility_scale = utility_scale
        self.priority_rail = priority_rail
        self.utility_gain_multiplier = utility_gain_multiplier
        self.alpha_elu = alpha_elu
        self.elu = nn.ELU(self.alpha_elu)

        if entropy_thresholds is None or entropy_levels is None:
            self.entropy_thresholds = [0]
            self.entropy_levels = [0]
        else:
            self.entropy_thresholds = entropy_thresholds
            self.entropy_levels = entropy_levels

        if mask_levels is None or mask_thresholds is None:
            self.mask_levels = [0]
            self.mask_thresholds = [0]
        else:
            self.mask_levels = mask_levels
            self.mask_thresholds = mask_thresholds

        if lower_cost_training_power is None:
            self.lower_cost_training = False
        else:
            self.lower_cost_training = True
            self.lower_cost_training_power = lower_cost_training_power
            self.lower_cost_training_period = lower_cost_training_period

    @staticmethod
    @torch.jit.script
    def _shortest_path(
            _soft_adj: torch.Tensor,
            _distances: torch.Tensor,
            _delta: float,
            _gamma: float
    ) -> torch.Tensor:
        """
        Floyd-Warshall for shortest paths.
        ---
        :param _soft_adj: Soft adjacency matrix.
        :param _distances: Distance matrix.
        :param _delta: Avoid division by zero.
        :param _gamma: Smoothing parameter.
        """
        # Scale the distances w_ij = d_ij / (s_ij + delta)
        _W = (_distances / (_soft_adj + _delta))
        _N = _W.size(0)
        _w_dist = _W.clone()

        # Nullify the diagonal
        _w_dist = _w_dist * (1 - torch.eye(_N, device=_W.device))

        # The Floyd-Warshall loop
        for k in range(_N):
            # Create a matrix where first we take distances to k (k-th column) and second row with distances from k
            # Then, create a matrix where each entry is sum of distance to k and from k
            _path_through_k = _w_dist[:, k].unsqueeze(1) + _w_dist[k, :].unsqueeze(0)

            # logsumexp for differentiability (this is softmin)
            _stacked = torch.stack([_w_dist, _path_through_k], dim=0)
            _w_dist = -_gamma * torch.logsumexp(-_stacked / _gamma, dim=0)

        return _w_dist


    def forward(self,
                soft_adj: torch.Tensor,
                original_adj: torch.Tensor,
                distances: torch.Tensor,
                flow: torch.Tensor,
                epoch: int) -> Tensor:
        """
        Function to calculate all components of the loss
        :param soft_adj: soft adjacency matrix (optimisation argument)
        :param original_adj: original adjacency matrix
        :param distances: distance matrix
        :param flow: demand potential matrix (travellers between cities)
        :param epoch: current epoch to tune BCE loss
        :return: total loss (scalar)
        """
        # First loss, cost of the infrastructure
        loss_cost = torch.sum(torch.mul(soft_adj,distances))

        # Add dynamic multiplier for the loss so it is not dominating in the learning period
        if self.lower_cost_training:
            loss_cost_multiplier = 1
        else:
            if epoch >= self.lower_cost_training_period:
                loss_cost_multiplier = 1
            else:
                loss_cost_multiplier = self.lower_cost_training_period - epoch
                loss_cost_multiplier = torch.pow(torch.Tensor([loss_cost_multiplier]),
                                                 self.lower_cost_training_power).item()
                loss_cost_multiplier = 1/loss_cost_multiplier


        # Calculate the shortest paths
        shortest_paths = self._shortest_path(soft_adj, distances, _delta=self.delta, _gamma=self.gamma)

        # Choice probability matrix
        exp_ut_rail = torch.exp(self.utility_scale * shortest_paths * self.priority_rail)
        exp_ut_base = torch.exp(self.utility_scale * distances)
        choice_matrix = exp_ut_rail / (exp_ut_rail + exp_ut_base)

        # Second loss
        distance_saved = distances - self.priority_rail * shortest_paths
        distance_saved = self.elu(distance_saved) + self.alpha_elu
        distance_saved = distance_saved * (1 - torch.eye(distance_saved.shape[0], device=distance_saved.device))
        utility_gain = flow * choice_matrix * distance_saved

        # Third loss
        inverse_soft = torch.eye(soft_adj.size(0), device=soft_adj.device) - soft_adj
        entropy_loss = (torch.square(torch.mul(soft_adj, inverse_soft))).sum()
        entropy_scale = self.entropy_levels[bisect(self.entropy_thresholds, epoch)]

        # Fourth loss
        mask_loss = (soft_adj * (1 - original_adj)).sum()
        mask_scale = self.mask_levels[bisect(self.mask_thresholds, epoch)]

        return (
                loss_cost_multiplier * loss_cost +
                self.utility_gain_multiplier * utility_gain.sum() +
                entropy_loss * entropy_scale +
                mask_scale * mask_loss
        )

    def exact_loss(self,
                   soft_adj: torch.Tensor | np.ndarray,
                   original_adj: torch.Tensor,
                   distances: torch.Tensor,
                   flow: torch.Tensor) -> dict:
        if type(soft_adj) is np.ndarray:
            soft_adj = torch.from_numpy(soft_adj)
            soft_adj = soft_adj.to(device=original_adj.device)
        # Convert to hard adjacency matrix
        adj_matrix_hard = soft_adj.round()

        # L1: Building cost
        loss_cost = torch.sum(torch.mul(adj_matrix_hard, distances))

        # Shortest paths
        shortest_paths = torch.mul(adj_matrix_hard, distances)
        shortest_paths = shortest_path_exact(shortest_paths)

        # Choice probability matrix
        exp_ut_rail = torch.exp(self.utility_scale * shortest_paths * self.priority_rail)
        exp_ut_base = torch.exp(self.utility_scale * distances)
        choice_matrix = exp_ut_rail / (exp_ut_rail + exp_ut_base)

        # L2: final calculations of the utility
        distance_saved = distances - self.priority_rail * shortest_paths
        distance_saved = self.elu(distance_saved) + self.alpha_elu
        utility_gain = flow * choice_matrix * distance_saved
        loss_utility = self.utility_gain_multiplier * utility_gain.sum()

        # Illegal edges
        illegal_edges = adj_matrix_hard - original_adj
        illegal_edges = illegal_edges.cpu().apply_(lambda x: x if x == 1 else 0)

        # Cover fraction
        cover_fraction = adj_matrix_hard.sum()/original_adj.sum()

        return {
            "Building_cost": loss_cost.item(),
            "Utility_increase": loss_utility.item(),
            "Total_loss": (loss_cost + loss_utility).item(),
            "Illegal_edges": (illegal_edges.sum()/2).item(),
            "Cover_fraction": round(cover_fraction.item(), 4)
        }



class UtilityBalancerSplitPaths(nn.Module):
    def __init__(self,
                 adjacency_matrix: torch.Tensor,
                 distances: torch.Tensor,
                 flow_matrix: torch.Tensor,
                 utility_scale: float = -1e-2,
                 priority_rail: float = 0.5,
                 utility_gain_multiplier: float = 1.0,
                 alpha_elu: float = 1.0,
                 entropy_thresholds: list | None = None,
                 entropy_levels: list | None = None,
                 mask_thresholds: list | None = None,
                 mask_levels: list | None = None,
                 lower_cost_training_power: float | None = None,
                 lower_cost_training_period: int | None = None
                 ):
        """
        Loss function
        ---
        :param adjacency_matrix: adjacency matrix to calculate feasible paths
        :param distances: distances between all pairs of nodes
        :param flow_matrix: flow matrix (demand potential) between all pairs of nodes
        :param utility_scale: to scale utilities (e^{ax}/(e^{ax}+e^{ay}) is sensitive to a
        :param priority_rail: scaling of the distance with the new service (e.g. greater speed)
        :param utility_gain_multiplier: balance between cost and utility gain
        :param alpha_elu: lower values for the exponent function
        :param entropy_thresholds: thresholds for increasing penalties with entropy
        :param entropy_levels: penalty levels with entropy (following thresholds)
        :param mask_thresholds: thresholds for increasing penalties with mask
        :param mask_levels: penalty levels for assigning links that do not exist in the original graph (following thresholds)
        :param lower_cost_training_power: during training, scale building cost B as B/[((N-n)^+)^p], where p is the power (parameter) and N is lower_cost_period
        :param lower_cost_training_period: scaling period for lower_cost_training_power
        """
        super(UtilityBalancerSplitPaths, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.feasible_paths = self._predefined_paths(self.adjacency_matrix)
        self.distances = distances
        self.flow_matrix = flow_matrix

        # Loss components
        self.utility_scale = utility_scale
        self.priority_rail = priority_rail
        self.utility_gain_multiplier = utility_gain_multiplier
        self.alpha_elu = alpha_elu
        self.elu = nn.ELU(self.alpha_elu)

        if entropy_thresholds is None or entropy_levels is None:
            self.entropy_thresholds = [0]
            self.entropy_levels = [0]
        else:
            self.entropy_thresholds = entropy_thresholds
            self.entropy_levels = entropy_levels

        if mask_levels is None or mask_thresholds is None:
            self.mask_levels = [0]
            self.mask_thresholds = [0]
        else:
            self.mask_levels = mask_levels
            self.mask_thresholds = mask_thresholds

        if lower_cost_training_power is None:
            self.lower_cost_training = False
        else:
            self.lower_cost_training = True
            self.lower_cost_training_power = lower_cost_training_power
            self.lower_cost_training_period = lower_cost_training_period

    @staticmethod
    def _predefined_paths(adjacency_matrix):
        n = adjacency_matrix
        all_paths = {(i, j): [] for i in range(n) for j in range(i + 1, n)}

        def _progressive_check(_start_node, _current, path):
            if _start_node > _current:
                first = _start_node
                second = _current
            else:
                first = _current
                second = _start_node

            # Ensure no loop
            if _current in path:
                return

            path.append(_current)

            if (first, second) not in all_paths or path not in all_paths[(first, second)]:
                all_paths[(first, second)].append(path.copy())

            for _neighbour in range(n):
                if adjacency_matrix[_current, _neighbour] == 1 and _neighbour not in path:
                    _progressive_check(start_node, _neighbour, path)

            path.pop()

        for start_node in range(n):
            for neighbour in range(start_node + 1, n):
                if adjacency_matrix[start_node, neighbour] == 1:
                    _progressive_check(start_node, neighbour, [])

        return all_paths

    def forward(self,
                soft_adj: torch.Tensor,
                epoch: int) -> Tensor:
        """
        Function to calculate all components of the loss
        :param soft_adj: soft adjacency matrix (optimisation argument)
        :param epoch: current epoch to tune BCE loss
        :return: total loss (scalar)
        """
        # First loss, cost of the infrastructure
        loss_cost = torch.sum(torch.mul(soft_adj,self.distances))

        # Add dynamic multiplier for the loss so it is not dominating in the learning period
        if self.lower_cost_training:
            loss_cost_multiplier = 1
        else:
            if epoch >= self.lower_cost_training_period:
                loss_cost_multiplier = 1
            else:
                loss_cost_multiplier = self.lower_cost_training_period - epoch
                loss_cost_multiplier = torch.pow(torch.Tensor([loss_cost_multiplier]),
                                                 self.lower_cost_training_power).item()
                loss_cost_multiplier = 1/loss_cost_multiplier


        # Calculate the shortest paths
        shortest_paths = self._shortest_path(soft_adj, distances, _delta=self.delta, _gamma=self.gamma)

        # Choice probability matrix
        exp_ut_rail = torch.exp(self.utility_scale * shortest_paths * self.priority_rail)
        exp_ut_base = torch.exp(self.utility_scale * distances)
        choice_matrix = exp_ut_rail / (exp_ut_rail + exp_ut_base)

print("XDXDXD")

