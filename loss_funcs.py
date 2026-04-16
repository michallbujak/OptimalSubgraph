from abc import ABC
from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from utils import LossMultiplier
from loss_parent_func import UtilityBalancerParent


class ShortestPathBalancer(UtilityBalancerParent, ABC):
    def __init__(self,
                 adjacency_matrix: Tensor,
                 distances: Tensor,
                 flow_matrix: Tensor,
                 building_cost_multiplier: LossMultiplier,
                 entropy_multiplier: LossMultiplier,
                 mask_multiplier: LossMultiplier,
                 utility_scale: float = -1e-2,
                 priority_rail: float = 0.5,
                 utility_gain_multiplier: float = 1.0,
                 delta: float = 1e-4,
                 gamma: float = 1e-1,
                 alpha_elu: float = 1.0,
                 **kwargs):
        super().__init__(
            adjacency_matrix=adjacency_matrix,
            distances=distances,
            flow_matrix=flow_matrix,
            building_cost_multiplier=building_cost_multiplier,
            entropy_multiplier=entropy_multiplier,
            mask_multiplier=mask_multiplier,
            utility_scale=utility_scale,
            priority_rail=priority_rail,
            utility_gain_multiplier=utility_gain_multiplier
        )
        """
        Class with loss based on the shortest path.
        :param delta: Avoid division by zero.
        :param gamma: Smoothing parameter for the softmin.
               Smaller values = better accuracy (though, negligible difference).
               Larger values = smoother gradients.
        :param alpha_elu: lower values for the exponent function
        :param **kwargs: Additional arguments passed in config if not required in the loss function.
        """
        self.delta = delta
        self.gamma = gamma
        self.alpha_elu = alpha_elu
        self.elu = nn.ELU(self.alpha_elu)

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

    def loss_utility_func(self, soft_adj: torch.Tensor, epoch: int | None, **kwargs) -> Tensor:
        shortest_paths = self._shortest_path(soft_adj, self.distances, _delta=self.delta, _gamma=self.gamma)

        # Choice probability matrix
        exp_ut_rail = torch.exp(self.utility_scale * shortest_paths * self.priority_rail)
        exp_ut_base = torch.exp(self.utility_scale * self.distances)
        choice_matrix = exp_ut_rail / (exp_ut_rail + exp_ut_base)

        # Distance saved
        distance_saved = self.distances - self.priority_rail * shortest_paths
        distance_saved = self.elu(distance_saved) + self.alpha_elu
        distance_saved = distance_saved * (1 - torch.eye(distance_saved.shape[0], device=distance_saved.device))

        utility_gain = self.flow_matrix * choice_matrix * distance_saved

        return self.utility_gain_multiplier * utility_gain.sum()



class AllPathsBalancer(UtilityBalancerParent, ABC):
    def __init__(self,
                 adjacency_matrix: Tensor,
                 distances: Tensor,
                 flow_matrix: Tensor,
                 building_cost_multiplier: LossMultiplier,
                 entropy_multiplier: LossMultiplier,
                 mask_multiplier: LossMultiplier,
                 utility_scale: float = -1e-2,
                 priority_rail: float = 0.5,
                 utility_gain_multiplier: float = 1.0,
                 **kwargs):
        super().__init__(
            adjacency_matrix=adjacency_matrix,
            distances=distances,
            flow_matrix=flow_matrix,
            building_cost_multiplier=building_cost_multiplier,
            entropy_multiplier=entropy_multiplier,
            mask_multiplier=mask_multiplier,
            utility_scale=utility_scale,
            priority_rail=priority_rail,
            utility_gain_multiplier=utility_gain_multiplier
        )
        self.predefined_paths = self._predefined_paths_foo(adjacency_matrix)

    @staticmethod
    def _predefined_paths_foo(adjacency_matrix):
        n = len(adjacency_matrix)
        adj_list = defaultdict(list)

        def find_all_paths(u, target, path):
            path = path + [u]

            if u == target:
                return [path]

            found_paths = []
            for neighbour in adj_list[u]:
                if neighbour not in path:
                    # Recursively all paths from the neighbour to the target
                    new_results = find_all_paths(neighbour, target, path)
                    # Use extend to add all found path-lists to our collection
                    found_paths.extend(new_results)

            return found_paths

        # Convert adjacency matrix to adjacency list
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i][j] == 1:
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        paths_dict = {}
        for i in range(n):
            for j in range(i + 1, n):
                # Pass an empty list for the initial path
                paths_dict[(i, j)] = find_all_paths(i, j, [])

        return paths_dict

    def loss_utility_func(self, soft_adj: torch.Tensor, epoch: int | None, **kwargs) -> Tensor:
        x = 0