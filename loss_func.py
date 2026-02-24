from bisect import bisect_left as bisect

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from utils import shortest_path_exact


class RailCostBenefitLoss(nn.Module):
    def __init__(self,
                 delta: float=1e-4,
                 gamma: float=1e-1,
                 utility_scale: float = -1e-2,
                 priority_rail: float = 0.5,
                 loss_component_balance: float = 1.0,
                 alpha_elu: float = 1.0,
                 entropy_thresholds: list | None = None,
                 entropy_levels: list | None = None,
                 mask_level: float = 1e4
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
        :param loss_component_balance: balance between cost and utility gain
        :param alpha_elu: lower values for the exponent function
        :param entropy_thresholds: thresholds for increasing penalties with entropy
        :param entropy_levels: penalty levels with entropy (following thresholds)
        :param mask_level: penalty level for assigning links that do not exist in the original graph
        """
        super(RailCostBenefitLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.utility_scale = utility_scale
        self.priority_rail = priority_rail
        self.loss_component_balance = loss_component_balance
        self.alpha_elu = alpha_elu
        self.elu = nn.ELU(self.alpha_elu)

        if entropy_thresholds is None:
            self.entropy_thresholds = [0]
        else:
            self.entropy_thresholds = entropy_thresholds
        if entropy_levels is None:
            self.entropy_levels = [0]
        else:
            self.entropy_levels = entropy_levels

        self.mask_level = mask_level

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
                epoch: float) -> Tensor:
        """
        Function to calculate all components of the loss
        :param soft_adj: soft adjacency matrix (optimization argument)
        :param original_adj: original adjacency matrix
        :param distances: distance matrix
        :param flow: demand potential matrix (travellers between cities)
        :param epoch: current epoch to tune BCE loss
        :return: total loss (scalar)
        """
        # First loss, cost of the infrastructure
        loss_cost = torch.sum(torch.mul(soft_adj,distances))

        # Calculate the shortest paths
        shortest_paths = self._shortest_path(soft_adj, distances, _delta=self.delta, _gamma=self.gamma)

        # Choice probability matrix
        exp_ut_rail = torch.exp(self.utility_scale * shortest_paths * self.priority_rail)
        exp_ut_base = torch.exp(self.utility_scale * distances)
        choice_matrix = exp_ut_rail / (exp_ut_rail + exp_ut_base)

        # Second loss
        distance_saved = distances - self.priority_rail * shortest_paths
        utility_gain = flow * choice_matrix * distance_saved
        utility_gain = self.elu(utility_gain) + self.alpha_elu

        # Third loss
        inverse_soft = torch.eye(soft_adj.size(0), device=soft_adj.device) - soft_adj
        entropy_loss = (torch.square(torch.mul(soft_adj, inverse_soft))).sum()
        entropy_scale = self.entropy_levels[bisect(self.entropy_thresholds, epoch)]

        # Fourth loss
        mask_loss = (soft_adj * (1 - original_adj)).sum()

        return (
                loss_cost +
                self.loss_component_balance * utility_gain.sum() +
                entropy_loss * entropy_scale +
                self.mask_level * mask_loss
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
        utility_gain = flow * choice_matrix * distance_saved
        utility_gain = self.elu(utility_gain) + self.alpha_elu
        loss_utility = self.loss_component_balance * utility_gain.sum()

        # Illegal edges
        illegal_edges = adj_matrix_hard - original_adj
        illegal_edges = illegal_edges.cpu().apply_(lambda x: x if x == 1 else 0)

        return {
            "cost": loss_cost.item(),
            "utility_gain": loss_utility.item(),
            "total_loss": (loss_cost + loss_utility).item(),
            "illegal_edges": (illegal_edges.sum()/2).item()
        }



