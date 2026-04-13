from abc import abstractmethod, ABC
from bisect import bisect

import torch
import torch.nn as nn
from torch import Tensor

from utils import LossMultiplier

class UtilityBalancerParent(nn.Module, ABC):
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
                 ):
        """
        Loss function
        ---
        :param adjacency_matrix: adjacency matrix to calculate feasible paths
        :param distances: distances between all pairs of nodes
        :param flow_matrix: flow matrix (demand potential) between all pairs of nodes
        :param building_cost_multiplier: dynamic building cost multiplier based on epoch
        :param entropy_multiplier: dynamic entropy multiplier based on epoch
        :param mask_multiplier: dynamic mask multiplier based on epoch
        :param utility_scale: to scale utilities (e^{ax}/(e^{ax}+e^{ay}) is sensitive to a
        :param priority_rail: scaling of the distance with the new service (e.g. greater speed)
        :param utility_gain_multiplier: balance between cost and utility gain
        """
        super(UtilityBalancerParent, self).__init__()
        # Store key graph characteristics
        self.building_cost_multiplier = building_cost_multiplier
        self.adjacency_matrix = adjacency_matrix
        self.distances = distances
        self.flow_matrix = flow_matrix

        # Loss components
        self.utility_scale = utility_scale
        self.priority_rail = priority_rail
        self.utility_gain_multiplier = utility_gain_multiplier

        # Dynamic loss multipliers
        self.building_cost_multiplier = building_cost_multiplier
        self.entropy_multiplier = entropy_multiplier
        self.mask_multiplier = mask_multiplier


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
        loss_cost = self.loss_cost_func(soft_adj, epoch)

        # Second, utility gain
        loss_utility = self.loss_utility_func(soft_adj, epoch)

        # Third, entropy loss
        loss_entropy = self.loss_entropy_func(soft_adj, epoch)

        # Fourth, mask loss
        loss_mask = self.loss_mask_func(soft_adj, epoch)

        return loss_cost + loss_utility + loss_entropy + loss_mask

    def loss_cost_func(self, soft_adj: Tensor, epoch: int | None) -> Tensor:
        """ Costs of building infrastructure """
        loss_cost_multiplier = self.building_cost_multiplier.obtain_multiplier(epoch)

        building_cost = torch.sum(torch.mul(soft_adj, self.distances))

        return loss_cost_multiplier * building_cost

    @abstractmethod
    def loss_utility_func(self, soft_adj: Tensor, epoch: int | None, **kwargs) -> Tensor:
        """ Gains from building infrastructure """
        ...

    def loss_entropy_func(self, soft_adj: Tensor, epoch: int) -> Tensor:
        """ Push values in the solution to a binary matrix """
        loss_entropy_multiplier = self.entropy_multiplier.obtain_multiplier(epoch)

        inverse_soft = torch.eye(soft_adj.size(0), device=soft_adj.device) - soft_adj
        entropy_loss = (torch.square(torch.mul(soft_adj, inverse_soft))).sum()

        return loss_entropy_multiplier * entropy_loss

    def loss_mask_func(self, soft_adj: Tensor, epoch: int) -> Tensor:
        """ Penalise for proposing edges outside of feasible connections"""
        loss_mask_multiplier = self.mask_multiplier.obtain_multiplier(epoch)

        mask_loss = (soft_adj * (1 - self.adjacency_matrix)).sum()

        return loss_mask_multiplier * mask_loss

    def exact_loss(self, soft_adj: Tensor) -> dict:
        # Convert to hard adjacency matrix
        adj_matrix_hard = soft_adj.round()

        # First loss, cost of the infrastructure
        loss_cost = self.loss_cost_func(soft_adj, None)

        # Second, utility gain
        loss_utility = self.loss_utility_func(soft_adj, None)

        # Illegal edges
        illegal_edges = adj_matrix_hard - self.adjacency_matrix
        illegal_edges = illegal_edges.cpu().apply_(lambda x: x if x == 1 else 0)

        # Cover fraction
        cover_fraction = adj_matrix_hard.sum()/self.adjacency_matrix.sum()

        return {
            "Building_cost": loss_cost.item(),
            "Utility_increase": loss_utility.item(),
            "Total_loss": (loss_cost + loss_utility).item(),
            "Illegal_edges": (illegal_edges.sum()/2).item(),
            "Cover_fraction": round(cover_fraction.item(), 4)
        }
