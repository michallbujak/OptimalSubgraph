import torch
import json
import argparse

from utils import generate_sample_graph, load_data, LossMultiplier
from model import OptimalSubgraphGNN
from training import train
from loss_funcs import ShortestPathBalancer
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

# Loss calculation
loss_args = params.get('loss_args', {})
CostMultiplier = LossMultiplier(
    method=loss_args['cost_multiplier_method'],
    thresholds=loss_args.get("cost_multiplier_thresholds", None),
    levels=loss_args.get("cost_multiplier_levels", None),
    period=loss_args.get("cost_multiplier_period", None),
    power=loss_args.get("cost_multiplier_power", None),
)

EntropyMultiplier = LossMultiplier(
    method=loss_args['entropy_multiplier_method'],
    thresholds=loss_args.get("entropy_multiplier_thresholds", None),
    levels=loss_args.get("entropy_multiplier_levels", None),
    period=loss_args.get("entropy_multiplier_period", None),
    power=loss_args.get("entropy_multiplier_power", None),
)

MaskMultiplier = LossMultiplier(
    method=loss_args['mask_multiplier_method'],
    thresholds=loss_args.get("mask_multiplier_thresholds", None),
    levels=loss_args.get("mask_multiplier_levels", None),
    period=loss_args.get("mask_multiplier_period", None),
    power=loss_args.get("mask_multiplier_power", None),
)

LossFunction = ShortestPathBalancer(
    adjacency_matrix=adj_matrix,
    distances=distances,
    flow_matrix=flow,
    building_cost_multiplier=CostMultiplier,
    entropy_multiplier=EntropyMultiplier,
    mask_multiplier=MaskMultiplier,
    alpha_elu=loss_args['alpha_elu'],
)

# Train
soft_adj = train(
    model=model,
    adjacency_matrix=adj_matrix,
    loss_calculator=LossFunction,
    parameters=params,
    optimizer=optimizer
)

# Evaluate results
evaluate(
    soft_adj=soft_adj,
    adjacency_matrix=adj_matrix,
    distances=distances,
    loss_calculator=LossFunction,
    flow=flow,
    **params['visualization']
)



