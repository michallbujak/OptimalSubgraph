from abc import abstractmethod, ABC
from bisect import bisect

import torch
import torch.nn as nn
from torch import Tensor


class UtilityBalancerParent(nn.Module, ABC):
    def __init__(self,
                 adjacency_matrix: Tensor,
                 distances: Tensor,
                 flow_matrix: Tensor,
                 utility_scale: float = -1e-2,
                 priority_rail: float = 0.5,
                 utility_gain_multiplier: float = 1.0,
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
        :param entropy_thresholds: thresholds for increasing penalties with entropy
        :param entropy_levels: penalty levels with entropy (following thresholds)
        :param mask_thresholds: thresholds for increasing penalties with mask
        :param mask_levels: penalty levels for assigning links that do not exist in the original graph (following thresholds)
        :param lower_cost_training_power: during training, scale building cost B as B/[((N-n)^+)^p], where p is the power (parameter) and N is lower_cost_period
        :param lower_cost_training_period: scaling period for lower_cost_training_power
        """
        super(UtilityBalancerParent, self).__init__()
        # Store key graph characteristics
        self.adjacency_matrix = adjacency_matrix
        self.distances = distances
        self.flow_matrix = flow_matrix

        # Loss components
        self.utility_scale = utility_scale
        self.priority_rail = priority_rail
        self.utility_gain_multiplier = utility_gain_multiplier

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

    def loss_cost_func(self, soft_adj: Tensor, epoch: int | None = None) -> Tensor:
        """ Costs of building infrastructure """
        if epoch is None or not self.lower_cost_training:
            loss_cost_multiplier = 1
        else:
            if epoch >= self.lower_cost_training_period:
                loss_cost_multiplier = 1
            else:
                loss_cost_multiplier = self.lower_cost_training_period - epoch
                loss_cost_multiplier = torch.pow(torch.Tensor([loss_cost_multiplier]),
                                                 self.lower_cost_training_power).item()
                loss_cost_multiplier = 1 / loss_cost_multiplier

        building_cost = torch.sum(torch.mul(soft_adj, self.distances))

        return building_cost * loss_cost_multiplier

    @abstractmethod
    def loss_utility_func(self, soft_adj: Tensor, epoch: int | None = None) -> Tensor:
        """ Gains from building infrastructure """
        ...

    def loss_entropy_func(self, soft_adj: Tensor, epoch: int | None = None) -> Tensor:
        """ Push values in the solution to a binary matrix """
        if epoch is None:
            return Tensor([0])

        inverse_soft = torch.eye(soft_adj.size(0), device=soft_adj.device) - soft_adj
        entropy_loss = (torch.square(torch.mul(soft_adj, inverse_soft))).sum()
        entropy_scale = self.entropy_levels[bisect(self.entropy_thresholds, epoch)]

        return entropy_loss * entropy_scale

    def loss_mask_func(self, soft_adj: Tensor, epoch: int | None = None) -> Tensor:
        """ Penalise for proposing edges outside of feasible connections"""
        if epoch is None:
            return Tensor([0])

        mask_loss = (soft_adj * (1 - self.adjacency_matrix)).sum()
        mask_scale = self.mask_levels[bisect(self.mask_thresholds, epoch)]

        return mask_loss * mask_scale

    def exact_loss(self, soft_adj: Tensor) -> dict:
        # Convert to hard adjacency matrix
        adj_matrix_hard = soft_adj.round()

        # First loss, cost of the infrastructure
        loss_cost = self.loss_cost_func(soft_adj)

        # Second, utility gain
        loss_utility = self.loss_utility_func(soft_adj)

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
