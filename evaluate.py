import pandas as pd
import matplotlib.pyplot as plt
import torch

from loss_func import RailCostBenefitLoss


def evaluate(
        soft_adj: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        distances: torch.Tensor,
        flow: torch.Tensor,
        loss_calculator: RailCostBenefitLoss,
) -> None:
    loss_dict = loss_calculator.exact_loss(soft_adj, adjacency_matrix, distances, flow)

    return None


