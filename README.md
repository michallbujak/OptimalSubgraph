# Optimal Subgraph: Transportation Application
This repository is dedicated to a framework that finds an optimal subgraph. 
The optimality is defined as a balance between edge preservation (associated with cost) and their contribution (system-level impact).

# Problem

For a given set of nodes $N$, located on a 2D space (long, lat) with a weight (population) 
$N \ni n_i = (x_i, y_i, w_i )$ and a demand matrix spanned between the nodes
$Q_{ij}$.

> Interpret as cities in Poland with their locations and a demand matrix of people willing to travel among them

find an _optimal_ subgraph $G^* \subset G$ connecting it. 

> Interpet as optimal High Speed Rail network between the cities.

Optimal in a sense of _Loss_ , composed of: 

* Benefits
* Costs

Costs are link additive constant, proportional to distance $c_{ij}$ - for instance cost of bulgind 100km railway between Warsaw and Łódź

$C(G')=\sum_{a \in G'}c_{ij}$

Benefits are measured with some classic transport-based metric, for instance accesibility or passenger hours:

$B(G') = Q \times T(G')$

where $Q$ is a fixed demand and $T(G')$ is travel times matrix $NxN$ for a given network $G'$.

Loss either difference between benefits and costs (not classicla, but easier for algos) or ratio $BCR$ (classical but harder probably).

### Remarks, issues:

* in the solution $G'$ shall to be a subset of a complete graph $G$, but in the meantime the continous variable $0 \leq a_{ij} \leq 1$ is accepted.
* we need gradient $dL\dG$


## TL;DR Script Launch
1) create virtual environment with requirements.txt.
2) `python3 main.py`.

## Input manipulation
All relevant parameters are stored in a .json configuration file.
To navigate to your custom file (particularly, when experimenting with various parameters selection), launch script as 
`python3 main.py --config <path_to_your_config>`.
By default, it uses `parameters.json` file.

# Methodology
Detailed methodology, including description of invididual components, is stored in the folder **Documentation**.



