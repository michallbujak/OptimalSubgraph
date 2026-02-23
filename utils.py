import torch

def generate_sample_graph(
        no_nodes: int = 16,
        edge_prob: float = 0.3,
        connected: bool | int = True,
        **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a sample graph with given number of nodes and edge probabilities.
    :param no_nodes: number of nodes in the graph
    :param edge_prob: probability of each edge
    :param kwargs: set nodes and edges parameters and seed
    :param connected: indicates whether the graph should be connected
    :return: weighted adjacency matrix, distance matrix, and demand potential
    """
    node_mean_weight = kwargs.get("node_base_weight", 200000)
    node_std_weight = kwargs.get("node_std_weight", 50000)
    edge_mean_weight = kwargs.get("edge_base_weight", 100)
    edge_std_weight = kwargs.get("edge_std_weight", 10)
    demand_potential_std = kwargs.get("demand_potential_std", 0.2)

    node_distro = torch.distributions.Normal(node_mean_weight, node_std_weight)
    edge_distro = torch.distributions.Normal(edge_mean_weight, edge_std_weight)
    demand_potential_distro = torch.distributions.Normal(1, demand_potential_std)

    # construct adjacency matrix
    is_connected = False
    while not is_connected:
        base_distances = torch.abs(edge_distro.sample((no_nodes, no_nodes)))
        mask = ((torch.rand(no_nodes, no_nodes) < edge_prob).int()
                & (torch.triu(torch.ones((no_nodes, no_nodes)), diagonal=1) == 1))
        base_distances = base_distances * mask
        base_distances = base_distances + base_distances.T
        min_dist = kwargs.get("minimal_distance", 10)
        base_distances[(base_distances < min_dist) & (base_distances != 0)] = min_dist
        base_distances = base_distances + base_distances.T
        adj_matrix = (base_distances > 0).int()

        if connected:
            is_connected = check_if_connected(adj_matrix)
        else:
            is_connected = True

    distances = shortest_path_exact(base_distances)

    # create demand potential, start from number of people within a node
    node_features = node_distro.sample((no_nodes, no_nodes))
    randomisers = demand_potential_distro.sample((no_nodes, no_nodes))
    randomisers[randomisers < 0.1] = 0.1
    demand_potential = randomisers * node_features/torch.mul(distances, distances)
    demand_potential.fill_diagonal_(0)

    return adj_matrix.float(), distances, demand_potential



def shortest_path_exact(
        adjacency_matrix: torch.Tensor,
        infty: float = 1e7
) -> torch.Tensor:
    dist = adjacency_matrix.clone()
    dist[dist == 0] = infty
    dist.diagonal().fill_(0)

    n = dist.size(0)

    for k in range(n):
        dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))

    return dist


def check_if_connected(
        adj_matrix: torch.Tensor
) -> bool:
    n = len(adj_matrix)
    visited = [False] * n

    def iter_foo(u):
        visited[u] = True
        for v in range(n):
            if adj_matrix[u][v] == 1 and not visited[v]:
                iter_foo(v)

    iter_foo(0)

    return all(visited)


def load_data(
        adjacency_matrix_path: str,
        distances_matrix_path: str,
        flow_matrix_path: str,
        **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import pandas as pd

    adjacency_matrix = pd.read_csv(adjacency_matrix_path, index_col=0)
    distances = pd.read_csv(distances_matrix_path, index_col=0)
    flow = pd.read_csv(flow_matrix_path, index_col=0)

    adjacency_matrix = torch.from_numpy(adjacency_matrix.to_numpy()).float()
    distances = torch.from_numpy(distances.to_numpy()).float()
    flow = torch.from_numpy(flow.to_numpy()).float()

    return adjacency_matrix, distances, flow