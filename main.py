import torch
import json
import argparse

from utils import generate_sample_graph, load_data
from model import OptimalSubgraphGNN
from training import train
from loss_func import UtilityInfrastructureBalancer
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="parameters.json")
args = parser.parse_args()
print(args)

params = json.load(open(args.config))

torch.manual_seed(params.get("seed", 123))
torch.cuda.manual_seed(params.get("seed", 123))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data (to be changed for the actual data)
if params.get('input', {}).get('exact_data', False):
    adj_matrix, distances, flow = load_data(**params['input'])
else:
    adj_matrix, distances, flow = generate_sample_graph(**params['random_graph'])

adj_matrix = adj_matrix.to(device)
distances = distances.to(device)
flow = flow.to(device)

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

# Loss calculator
loss_calculator = UtilityInfrastructureBalancer(**params.get('loss_args', {}))

# Train
soft_adj = train(
    model=model,
    adjacency_matrix=adj_matrix,
    distances=distances,
    flow=flow,
    loss_calculator=loss_calculator,
    parameters=params,
    optimizer=optimizer
)

# Evaluate results
evaluate(
    soft_adj=soft_adj,
    adjacency_matrix=adj_matrix,
    distances=distances,
    loss_calculator=loss_calculator,
    flow=flow,
    **params['visualization']
)



