# Overview
To calculate properly loss function in our setting, we need to calculate shortest paths. To do that, we need a differentiable algorithm that yields the solution.


# Lemma
Maximum and minimum values can be approximated with their smooth counterparts. The smooth counterpart is differentiable and therefore applicable for NN applications. [Wikipedia](https://en.wikipedia.org/wiki/Smooth_maximum)
We approximate the softmin function with LogSumExp (LSE) for $x=(x_1, \ldots, x_n)$ as:

$$
\mathrm{softmin}_{\lambda}(x) = -\lambda \mathrm{ln} \sum_{i \leq n} \mathrm{exp}(-x_i/\lambda)
$$

In the limit, $\mathrm{softmin}_{\lambda}(x) \xrightarrow{\lambda\rightarrow 0} \mathrm{min}(x)$. However, for the NN applications the $\lambda$ will not be very close to $0$.

# Lemma
Iterative approaches are differentiable. We can use Long Short-Term Memory (LSTM) network for iterative calculations.
See [PyTorch Tutorial](https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html), [PyTorch Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

# Problem definition
We seek a binary adjacency matrix $A_f$ of the undirected graph $G$. To achieve this, we will work on the the continuous version $A=\[a_{ij}\] {i,j\leq n}: a_{i, j} \in [0, 1]$. Distance between nodes is expressed via the distance (weight) matrix $D$.
To ensure that avoid null division, we introduce $W = [w_{ij}] := \frac{d_{ij}}{a_{ij}+\delta}$. For a large $w_{ij}$ (small $a_{ij}$, the path will be effectively removed from the shortest path calculation. Let $X$ denote a vector with node features (population).

| Symbol | Definition |
| --- | --- |
| G | original, undirected graph, weighted nodes and edges |
| A | soft adjacency matrix (entries in $[0, 1]$) |
| D | distance between nodes - original weight matrix |
| W | amended weight matrix, where unlikely links are assigned huge weights |

# Shortest path
We will apply Floyd–Warshall algorithm ([wiki](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)). Thankfully, due to the nature of our problem, we can easily fix a low maximal number of intermediary points and otherwise return a huge constant.

#### Why this algorithm?
For dense graphs it performs better than the Dijkstra algorithm.

## Base algorithm
We consider the graph $G$ with nodes $V = (1, \ldots, n)$. Consider a function $\mathrm{shortestPath}(i, j, k)$ between nodes $i$ and $j$ using vertices from the set $\{1, \ldots, k\}$. Note, the shortest path is naturally given by $\mathrm{shortestPath}(i, j, N := |V|)$, which we'll find recursively. 
Note also that $\mathrm{shortestPath}(i, j, k) \leq \mathrm{shortestPath}(i, j, k-1)$ (greater flexibility).
We consider two cases:
1. $\mathrm{shortestPath}(i, j, k) = \mathrm{shortestPath}(i, j, k-1)$ that translates to the $k$ node being inactive in the shortest path from $i$ to $j$.
2. $\mathrm{shortestPath}(i, j, k) < \mathrm{shortestPath}(i, j, k-1) \implies \mathrm{shortestPath}(i, j, k) = \mathrm{shortestPath}(i, k, k-1) + \mathrm{shortestPath}(k, j, k-1)$.

Therefore, we can write 
$$\mathrm{shortestPath}(i, j, k) = \mathrm{min}(\mathrm{shortestPath}(i, j, k-1),\ \mathrm{shortestPath}(i, k, k-1) + \mathrm{shortestPath}(k, j, k-1)).$$

The first element is:

$$
\mathrm{shortestPath}(i, j, 0) = 
\begin{cases} 
d_{ij}, & \mathrm{ if \ edge \ (i, j) \ exists,} \\
\infty, & \mathrm{otherwise.}
\end{cases}
$$

Algorithm:
```
let Z be a |V| × |V| array of minimum distances initialized to ∞ (infinity) 
for each edge (u, v) do  
    Z[u][v] = w[u][v]  // The weigh of the edge (u, v) - distance x existence probability
for each node v do
    Z[v][v] = 0 // The diagonal elements of Z are 0
for k from 1 to |V| // Iterate maximal intermediate all nodes
    for i from 1 to |V| // Iterate all initial nodes
        for j from 1 to |V| // Iterate all ending nodes
            if Z[i][j] > Z[i][k] + Z[k][j] // If going through node k saves distance
                Z[i][j] = Z[i][k] + Z[k][j]
            end if
```

## NN-focused algorithm adjustments
To calculate the loss function that is based on the shortest paths, we need to guarantee that our function is differentiable. We will use tools described in the first two lemmas.

#### Step 1: Initialisation
We start with the weighted matrix W, as defined earlier. Let's denote $D^{(0)} = W$. To properly penalise non-existing edges, adjust the parameter $\delta$.

#### Step 2: Min to softmin and recurrence statement 
The last `if` statement in the Algorithm is to be replaced with the `softmin` function. That is, `dist[i][j]` is calculated as 

$$
-\lambda \ \mathrm{ln}(\mathrm{exp}(D^{(k-1)}_{ij}/\lambda) + \mathrm{exp}((D^{(k-1)}_{ik} + D^{(k-1)}_{kj})/\lambda)).
$$

At a higher level, we define the recurrence:

$$D^{(k)}_{ij} = \mathrm{softmax}(D^{(k-1)}_{ij}, D^{(k-1)}_{ik} + D^{(k-1)}_{kj}).$$


