import torch

def generate_sample_graph(
        no_nodes: int = 16,
        edge_prob: float = 0.3,
        **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a sample graph with given number of nodes and edge probabilities.
    :param no_nodes: number of nodes in the graph
    :param edge_prob: probability of each edge
    :param kwargs: set nodes and edges parameters and seed
    :return: weighted adjacency matrix and node features
    """
    if kwargs.get("seed", None) is not None:
        torch.manual_seed(seed=kwargs.get("seed"))

    node_mean_weight = kwargs.get("node_base_weight", 100)
    node_std_weight = kwargs.get("node_std_weight", 10)
    edge_mean_weight = kwargs.get("edge_base_weight", 100)
    edge_std_weight = kwargs.get("edge_std_weight", 10)

    node_dist = torch.distributions.Normal(node_mean_weight, node_std_weight)
    edge_dist = torch.distributions.Normal(edge_mean_weight, edge_std_weight)

    # construct adjacency matrix
    adj_matrix = torch.abs(edge_dist.sample((no_nodes, no_nodes)))
    mask = ((torch.rand(no_nodes, no_nodes) < edge_prob)
            & (torch.triu(torch.ones((no_nodes, no_nodes)), diagonal=1) == 1))
    adj_matrix = adj_matrix * mask
    adj_matrix = adj_matrix + adj_matrix.T

    node_features = node_dist.sample((no_nodes, 1))

    if kwargs.get("print_random_graph", False):
        print(f"Graph generated: {no_nodes} nodes, {int(adj_matrix.sum() / 2)} edges")
        print(f"Total edge weight (full): {0.5 * adj_matrix.sum().item():.1f}")
        print(f"Total node prize (full): {node_features.sum().item():.1f}\n")

    return adj_matrix, node_features
