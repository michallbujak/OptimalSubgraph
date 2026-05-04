import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, Sequential, DenseGCNConv
from torch_geometric.utils import dense_to_sparse

class OptimalSubgraphGNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 adjacency_matrix: torch.Tensor | None = None,
                 mp_units: tuple=(64,64,48),
                 mp_activation: str='ReLU',
                 mlp_units: tuple=(32, 16),
                 mlp_activation: str='ReLU',
                 final_activation: str='Sigmoid',
                 prior_logit_shift: float=3.,
                 **kwargs):
        super().__init__()

        mp_act_class = getattr(nn, mp_activation)
        mlp_act_class = getattr(nn, mlp_activation)

        self.sparse_formulation = kwargs.get("sparse_GCNN", False)

        # Message passing (nodes)
        if len(mp_units) > 0:
            if self.sparse_formulation:
                header = "x, edge_index, edge_weight -> x"
                mp_architecture = []
            else:
                self.mp_architecture = nn.ModuleList()
            prev = in_channels
            for mp_unit in mp_units:
                if self.sparse_formulation:
                    gcn = GCNConv(prev, mp_unit, normalize=False)
                    mp_architecture.append((gcn, header))
                    mp_architecture.append(mp_act_class())
                else:
                    self.mp_architecture.append(DenseGCNConv(prev, mp_unit))
                    self.mp_architecture.append(mp_act_class())
                prev = mp_unit

            if self.sparse_formulation:
                self.mp_architecture = Sequential("x, edge_index, edge_weight", mp_architecture)

            self.embedding_dim = mp_units[-1]
        else:
            self.mp_architecture = nn.Identity()
            self.embedding_dim = in_channels

        # Parametrise (make learnable) the adjacency matrix
        if adjacency_matrix is not None:
            self.adapt_adj = nn.Parameter(adjacency_matrix.clone().detach().requires_grad_(True))
        else:
            self.adapt_adj = None

        # Edges: using node embeddings and original edge weights
        mlp_in_channels = 2 * self.embedding_dim + 1 # two node representations + link weight
        # mlp_in_channels = self.embedding_dim
        self.edge_mlp_architecture = nn.Sequential()
        for mlp_unit in mlp_units:
            self.edge_mlp_architecture.append(nn.Linear(mlp_in_channels, mlp_unit))
            self.edge_mlp_architecture.append(mlp_act_class(inplace=True))
            mlp_in_channels = mlp_unit
        self.edge_mlp_architecture.append(nn.Linear(mlp_in_channels, 1))

        if final_activation == 'gumbel_softmax':
            self.final_activation = GumbelSoftmax(**kwargs.get('final_activation_parameters', {}))
        else:
            self.final_activation = getattr(nn, final_activation)()

        # Force higher initial values to not get stuck at local optimum in 0
        if kwargs.get("force_initial_value", False):
            final_linear_layer = self.edge_mlp_architecture[-1]
            nn.init.constant_(final_linear_layer.bias, kwargs.get("initial_value", 3.0))
            nn.init.xavier_uniform_(final_linear_layer.weight, gain=0.01)

        self.prior_logit_shift = prior_logit_shift

    def forward(self,
                x: torch.Tensor,
                original_adj: torch.Tensor):
        # Use learnable adjacency matrix or original
        if self.adapt_adj is not None:
            lr_adj_matrix = torch.relu(self.adapt_adj)
            lr_adj_matrix = (lr_adj_matrix + lr_adj_matrix.T) / 2.0
        else:
            lr_adj_matrix = torch.relu(original_adj)

        # First, node embeddings
        if self.sparse_formulation:
            edge_index, edge_weight = dense_to_sparse(lr_adj_matrix)
            x = self.mp_architecture(x, edge_index, edge_weight)

        else:
            x_in = x.unsqueeze(0) if x.dim() == 2 else x
            adj_in = lr_adj_matrix.unsqueeze(0) if lr_adj_matrix.dim() == 2 else lr_adj_matrix

            for layer in self.mp_architecture:
                if isinstance(layer, DenseGCNConv):
                    x_in = layer(x_in, adj_in)
                else:
                    x_in = layer(x_in)

            x = x_in.squeeze(0)

        # Second, prepare pairwise features for every possible edge
        N = x.shape[0]
        h_i = x.unsqueeze(1).expand(N, N, -1)
        h_j = x.unsqueeze(0).expand(N, N, -1)
        edge_feat = lr_adj_matrix.unsqueeze(-1)  # (N, N, 1)
        pair_input = torch.cat([h_i, h_j, edge_feat], dim=-1)  # (N, N, 2*emb + 1)

        # Third, apply mlp part
        s = self.edge_mlp_architecture(pair_input).squeeze(-1)

        # Fourth, stay close to the original adj matrix
        prior_logits = (original_adj * 2.0 - 1.0) * self.prior_logit_shift
        s = s + prior_logits

        o = self.final_activation(s)

        # Finally, enforce symmetry and remove self-loops
        o = (o + o.T) / 2.0
        o = o * (1 - torch.eye(N, device=o.device))

        return o


class GumbelSoftmax(nn.Module):
    """Gumbel softmax as nn.module"""
    def __init__(self, tau=1.0, hard=False, dim=-1):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.dim = dim

    def forward(self, x):
        return F.gumbel_softmax(x, tau=self.tau, hard=self.hard, dim=self.dim)



