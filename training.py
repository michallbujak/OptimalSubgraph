import torch

from model import OptimalSubgraphGNN
from loss_func import RailCostBenefitLoss

def train(
        model: OptimalSubgraphGNN,
        adjacency_matrix: torch.Tensor,
        distances: torch.Tensor,
        demand_potential: torch.Tensor,
        parameters: dict,
        optimizer: torch.optim.Optimizer | None = None,
        **kwargs,
) -> None:
    num_epochs = parameters['num_epochs']
    model.train()

    loss_calculator = RailCostBenefitLoss(**parameters.get('loss_args', {}))

    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            *parameters.get('optimizer_args', {})
        )

    loss_progress = []

    x = adjacency_matrix

    for epoch in range(num_epochs):
        soft_adj = model(x, adjacency_matrix)
        loss = loss_calculator(soft_adj, distances, demand_potential)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch % 50 == 0) & (kwargs.get("show_training_progress", False)):
            loss_progress.append(loss.item())
            print(f'Epoch {epoch}, Loss: {loss_progress[-1]}')

    if kwargs.get("show_training_output", True):
        import matplotlib.pyplot as plt
        import pandas as pd

        pd.DataFrame(soft_adj.cpu().detach().numpy()).to_csv('results/training_output.csv')

        plt.plot(loss_progress)
        plt.savefig('results/loss_progress.png')


    return None