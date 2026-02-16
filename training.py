import torch

from model import OptimalSubgraphGNN
from loss_func import RailCostBenefitLoss

def training(
        model: OptimalSubgraphGNN,
        adjacency_matrix: torch.Tensor,
        distances: torch.Tensor,
        demand_potential: torch.Tensor,
        parameters: dict,
        optimizer: torch.optim.Optimizer | None = None
) -> None:
    num_epochs = parameters['num_epochs']
    model.train()

    loss_calculator = RailCostBenefitLoss(*parameters.get('loss_args', {}))

    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            *parameters.get('optimizer_args', {})
        )

    x = adjacency_matrix

    for epoch in range(num_epochs):
        soft_adj = model(x, adjacency_matrix, parameters.get('final_activation', 'sigmoid'))
        loss = loss_calculator(soft_adj, distances, demand_potential)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return None