import torch
import torch.nn as nn
from torch import Tensor


class RailCostBenefitLoss(nn.Module):
    def __init__(self,
                 delta: float=1e-4,
                 gamma: float=1e-1):
        """
        Loss function
        ---
        :param delta: Avoid division by zero.
        :param gamma: Smoothing parameter for the softmin.
               Smaller values = better accuracy (though, negligible difference).
               Larger values = smoother gradients.
        """
        super(RailCostBenefitLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma

    @staticmethod
    @torch.jit.script
    def _shortest_path(
            _soft_adj: torch.Tensor,
            _distances: torch.Tensor,
            _delta: float,
            _gamma: float,
            _max_distance: float,
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
                distances: torch.Tensor,
                demand_potential: torch.Tensor,
                utility_scale: float=-1e-2,
                priority_rail: float=0.5,
                loss_component_balance: float=1.0) -> Tensor:
        """
        Function to calculate all components of the loss
        :param soft_adj: soft adjacency matrix (optimisation argument)
        :param distances: distance matrix
        :param demand_potential: demand potential matrix (travellers between cities)
        :param utility_scale: to scale utilities (e^{ax}/(e^{ax}+e^{ay}) is sensitive to a
        :param priority_rail: scaling of the distance with the new service (e.g. greater speed)
        :param loss_component_balance: balance between cost and utility gain
        :return: loss (scalar)
        """
        # First loss, cost of the infrastructure
        loss_cost = torch.sum(torch.mul(soft_adj,distances))

        # Calculate the shortest paths
        shortest_paths = self._shortest_path(soft_adj, distances, delta=self.delta, gamma=self.gamma)

        # Choice probability matrix
        exp_ut_rail = torch.exp(utility_scale * shortest_paths * priority_rail)
        exp_ut_base = torch.exp(utility_scale * distances)

        choice_matrix = exp_ut_rail / (exp_ut_rail + exp_ut_base)
        utility_gain = demand_potential * choice_matrix * (priority_rail * shortest_paths - distances)

        return loss_cost + loss_component_balance * utility_gain.sum()



