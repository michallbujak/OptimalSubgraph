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
        self.shortest_paths = self._shortest_paths(adjacency_matrix, self.distances)
        self.predefined_paths = self._predefined_paths_foo(
            adjacency_matrix,
            self.shortest_paths,
            kwargs.get("max_distance", 1.5),
        )
        self.extended_paths, self.extended_distances, self.max_paths = self._amend_predefined_paths()
        self.baseline_utility = self._baseline_utility()

    @staticmethod
    def _shortest_paths(adjacency_matrix: Tensor, distances: Tensor, infty=1e7) -> Tensor:
        """ Shortest path just to calculated predefined baseline utility and find feasible paths """
        dist = distances.clone()
        dist[adjacency_matrix == 0] = infty
        dist.diagonal().fill_(0)

        n = dist.size(0)

        for k in range(n):
            dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))

        return dist

    @staticmethod
    def _predefined_paths_foo(adjacency_matrix: Tensor, min_distances: Tensor, max_distance: float) -> dict[tuple[int], list[int]]:
        n = len(adjacency_matrix)
        adj_list = defaultdict(list)

        def find_all_paths(u, target, path):
            path = path + [u]

            if len(path) >= 2:
                total_distance = 0
                for _node_ind in range(len(path) - 1):
                    total_distance += min_distances[path[_node_ind], path[_node_ind + 1]]
                if total_distance > max_distance * min_distances[path[0], target]:
                    return []

            if u == target:
                return [(path, total_distance.item())]

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

        return dict(paths_dict)

    def _amend_predefined_paths(self):
        def _foo(_list, _size):
            out = torch.zeros((_size, _size), device=self.adjacency_matrix.device)
            for i in range(len(_list)-1):
                out[_list[i], _list[i+1]] = 1
            return out
        n = len(self.adjacency_matrix)

        new_dict = {key: [(_foo(i, n), j) for (i, j) in val] for key, val in self.predefined_paths.items()}

        max_paths = max(len(path) for path in self.predefined_paths.values())
        extended_path_tensor = torch.zeros((n, n, max_paths, n, n), device=self.adjacency_matrix.device)
        extended_value_tensor = torch.zeros((n, n, max_paths), device=self.adjacency_matrix.device)
        for key, path_list in new_dict.items():
            for path_no, path in enumerate(path_list):
                extended_path_tensor[*key, path_no] = path[0]
                extended_value_tensor[*key, path_no] = path[1]

        return extended_path_tensor, extended_value_tensor, max_paths


    def _baseline_utility(self):
        return torch.exp(self.utility_scale * self.shortest_paths)

    def loss_utility_func(self, soft_adj: torch.Tensor, epoch: int | None, **kwargs) -> Tensor:
        # Maximum number of predefined paths
        soft_extended = torch.clone(self.extended_paths)
        soft_extended[:, :, :] = soft_adj

        # Check probability of individual links existing
        probs_partial = soft_extended*self.extended_paths

        # Don't count 0's in the product
        probs_partial_ones = torch.where(probs_partial == 0, torch.ones_like(probs_partial), probs_partial)
        probs_agg = torch.prod(probs_partial_ones, dim=-1)
        probs_agg = torch.prod(probs_agg, dim=-1)
        # Prod of nonzero elements
        has_non_zeros = (probs_partial != 0).any(dim=(-2, -1))
        probs_agg = torch.where(has_non_zeros, probs_agg, torch.zeros_like(probs_agg))



        utility_gain = 1

        return self.utility_gain_multiplier * utility_gain.sum()