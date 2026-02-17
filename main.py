import torch
import json
import argparse

from example_graph_generation import generate_sample_graph
from model import OptimalSubgraphGNN
from training import train

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="parameters.json")
args = parser.parse_args()
print(args)

params = json.load(open(args.config))

torch.manual_seed(params.get("seed", 123))
torch.cuda.manual_seed(params.get("seed", 123))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data (to be changed for the actual data)
adj_matrix, distances, demand_potential = generate_sample_graph(**params['random_graph'])

adj_matrix = adj_matrix.to(device)
distances = distances.to(device)
demand_potential = demand_potential.to(device)

# Prepare model
model = OptimalSubgraphGNN(in_channels=adj_matrix.shape[0],
                           adjacency_matrix=adj_matrix,
                           **params.get("model_args", {}))
model.to(device)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    **params.get('optimizer_args', {})
)

# Train
train(
    model=model,
    adjacency_matrix=adj_matrix,
    distances=distances,
    demand_potential=demand_potential,
    parameters=params,
    optimizer=optimizer
)

x = 0


