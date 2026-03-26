# Model and Loss
This document contains details regarding structural details of the model and loss function.

## Model 
The model comprises three learnable components:
- message passing on nodes (GCNN)
- edge embeddings (MLP)
- learnable adjacency matrix (optional)

#### Model structure details
- GCNN is build as alternating `GCNConv` and activation (`ReLU` by default).
- MLP Architecture is multilayered alternation of `Linear` and activation (`ReLU` by default). 
- Learnable adjacency matrix is passed as a tunable parameter inside the model.

#### Forward structure
Each `forward` method follows those steps:
1. Obtain edge indexes from the learned adjacency matrix or strictly from the adjacency matrix.
2. Apply GCNN part.
3. Prepare tensor for all potential edges; concatenating NxNxd (d features of node i), NxNxd (d features of node j), NxNx1 (original adjacency matrix) intro NxNx(2d+1) tensor.
4. Apply the MLP part.
5. Add obtain coefficients on top of the preexisting logits (based on the adjacency matrix).
6. Apply final activation (`sigmoid` by default).
7. Add to the output its transposition and divide by 2 (ensure symmetry), nullify diagonal.

## Loss
Loss comprises four components:
- Connectedness cost (cost of builiding the infrastructure)
- Utility gain (socioeconomic benefits)
- Entropy (penalise values distant from 0 and 1)
- Mask (penalise connections not allowed by the adjacency matrix)

### Loss 1: Connectedness
The loss is a direct sum of elements of the adjacency matrix

### Loss 2: Utility
**Note:** multipliation between matrices refers to elementwise matrix multiplication.

**Part 1:** `shortest_paths`:
1. Calculate weighted shortest paths `shortest_paths` (details on the implementation are in document _shortest_path_algorithm_).
   
**Part 2:** `choice_matrix`:
1. New mode utility matrix `exp_ut_rail` exponens as $\exp($`utility_scale`\*`shortest_paths`\*`utility_rail`).
2. Base utility matrix `exp_ut_base` as $\exp($`utility_scale`\*`distances`, where `distances` is predifined distance between nodes (**Note:** distance can be different (shorter) than the shortest paths on the adjacency matrix)
3. Choice matrix as `choice_matrix = exp_ut_rail / (exp_ut_rail + exp_ut_base)`

**Part 3:** `distance_saved`:
1. Calculate distance difference with new mode, applying parameter `priority_rail`, that is, `distances - self.priority_rail * shortest_paths`
2. Make it non-negative with `elu` shifted up by `elu` parameter.
3. Nullify diagonal.

**Utility formulation:** <br>
Using `flow` input data, calculate `utility_gain = flow * choice_matrix * distance_saved`. 
Utility gain is then multiplied by `utility_gain_multiplier`.

### Loss 3: Entropy
Calculate entropy loss (`soft_adj * (1 - soft_adj`).
Depending on epoch, there are different multipliers for the entropy loss, based on parameters `entropy_thresholds` and `entropy_levels`.

### Loss 4: Mask
Sum all not feasible edges (all enteries of `soft_adj` where the original adjacency matrix has $0$s). Loss depends on `mask_thresholds` and `mask_levels`



