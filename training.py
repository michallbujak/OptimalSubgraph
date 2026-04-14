import numpy as np
import torch

from model import OptimalSubgraphGNN
from loss_parent_func import UtilityBalancerParent

def train(
        model: OptimalSubgraphGNN,
        adjacency_matrix: torch.Tensor,
        loss_calculator: UtilityBalancerParent,
        parameters: dict,
        optimizer: torch.optim.Optimizer | None = None
) -> np.ndarray:
    num_epochs = parameters['num_epochs']
    model.train()

    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            **parameters.get('optimizer_args', {})
        )

    loss_progress = np.array([])

    x = torch.eye(adjacency_matrix.size(0)).to(adjacency_matrix.device)
    # x = adjacency_matrix.clone()

    for epoch in range(num_epochs):
        soft_adj = model(x, adjacency_matrix)
        loss = loss_calculator(soft_adj, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_progress = np.append(loss_progress, loss.item())

        if (epoch % 50 == 0) & (parameters.get("show_training_progress", False)):
            print(f'Epoch {epoch}, Loss: {loss_progress[-1]}')

    if parameters.get("show_training_output", True):
        import os
        import matplotlib.pyplot as plt
        import pandas as pd

        if not os.path.exists('results'):
            os.makedirs('results')

        pd.DataFrame(soft_adj.cpu().detach().numpy()).to_csv('results/training_output.csv')

        plt.plot(loss_progress)
        plt.savefig('results/loss_progress.png')

        np.save('results/loss_progress.npy', loss_progress)


    return soft_adj


def train_old(
        model: OptimalSubgraphGNN,
        adjacency_matrix: torch.Tensor,
        distances: torch.Tensor,
        flow: torch.Tensor,
        loss_calculator: UtilityBalancerParent,
        parameters: dict,
        optimizer: torch.optim.Optimizer | None = None
) -> np.ndarray:
    num_epochs = parameters['num_epochs']
    model.train()

    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            **parameters.get('optimizer_args', {})
        )

    loss_progress = np.array([])

    x = torch.eye(adjacency_matrix.size(0)).to(adjacency_matrix.device)
    # x = adjacency_matrix.clone()

    for epoch in range(num_epochs):
        soft_adj = model(x, adjacency_matrix)
        loss = loss_calculator(soft_adj, adjacency_matrix, distances, flow, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_progress = np.append(loss_progress, loss.item())

        if epoch == 50:
            ooo = 0

        if (epoch % 50 == 0) & (parameters.get("show_training_progress", False)):
            print(f'Epoch {epoch}, Loss: {loss_progress[-1]}')

    if parameters.get("show_training_output", True):
        import os
        import matplotlib.pyplot as plt
        import pandas as pd

        if not os.path.exists('results'):
            os.makedirs('results')

        pd.DataFrame(soft_adj.cpu().detach().numpy()).to_csv('results/training_output.csv')

        plt.plot(loss_progress)
        plt.savefig('results/loss_progress.png')

        np.save('results/loss_progress.npy', loss_progress)


    return soft_adj.cpu().detach().numpy()