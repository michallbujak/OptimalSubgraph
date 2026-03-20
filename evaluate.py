from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

from loss_func import UtilityInfrastructureBalancer


def evaluate(
        soft_adj: torch.Tensor | np.ndarray,
        adjacency_matrix: torch.Tensor,
        distances: torch.Tensor,
        flow: torch.Tensor,
        loss_calculator: UtilityInfrastructureBalancer,
        label_coordinates_path: str | Path,
        **kwargs
) -> None:
    loss_dict = loss_calculator.exact_loss(soft_adj, adjacency_matrix, distances, flow)
    for k, v in loss_dict.items():
        print(f'Metric "{k}": {v}')

    labels_coordinates = pd.read_csv(label_coordinates_path).set_index('id').to_dict('index')

    fig, ax = plt.subplots(figsize=tuple(kwargs.get('figure_size', [8, 8])))
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i+1, adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] > 0:
                ax.plot(
                    (labels_coordinates[i]['y_coord'], labels_coordinates[j]['y_coord']),
                    (labels_coordinates[i]['x_coord'], labels_coordinates[j]['x_coord']),
                    color=kwargs.get('base_line_color', 'gray'),
                    ls=kwargs.get('ls', '-'),
                    lw=kwargs.get('lw', 5),
                    alpha=kwargs.get('alpha', 0.3),
                    zorder=kwargs.get('zorder', 1),
                )
            if soft_adj[i, j] > 0.5:
                ax.plot(
                    (labels_coordinates[i]['y_coord'], labels_coordinates[j]['y_coord']),
                    (labels_coordinates[i]['x_coord'], labels_coordinates[j]['x_coord']),
                    color=kwargs.get('model_line_color', 'red'),
                    ls=kwargs.get('ls', '-'),
                    lw=kwargs.get('lw', 1),
                    alpha=kwargs.get('alpha', 0.7),
                    zorder=kwargs.get('zorder', 1),
                )

        annotate_city(
            ax,
            labels_coordinates[i]['name'],
            (labels_coordinates[i]['y_coord'], labels_coordinates[i]['x_coord'])
        )

    ax.set_title(kwargs.get('title', ""), fontsize=16, pad=20)

    for direction in ['left', 'right', 'top', 'bottom']:
        ax.spines[direction].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.tight_layout()

    if not os.path.exists('results'):
        os.makedirs('results')

    plt.savefig('results/output.png', dpi=kwargs.get('dpi', 300))

    return None


def annotate_city(ax, label, coords):
    ax.annotate(
        label,
        coords,
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

