# Parameter description

This document explains all parameters used in the `parameters.json` file.

## Input parameters
Section described under `input`

| Name                    | Description                                                                                                           |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------|
| exact_data              | Choose whether to load actual data (`true`) or sample data (`false`)                                                  |
| adjacency_matrix_path   | Loads if `exact_data = true`: path to adjacency matrix. Adjacency matrix indicates which connections are allowed in the graph.<br>**Note:** the matrix requires precomputation of shortest path. The reason is that for the base mode the computed distance can differ from the distance via the feasible network (adjacency matrix) |
| distances_matrix_path   | Loads if `exact_data = true`: path to distance matrix. Distance matrix indicates potential direct distance between nodes in the network |
| flow_matrix_path        | Loads if `exact_data = true`: path to flow matrix. Flow matrix indicates the potential flow (demand) between each pair of nodes |
| no_nodes                | Loads if `exact_data = false`: number of nodes in the generated graph                                                 |
| edge_prob               | Loads if `exact_data = false`: following random graph generated proposed by Erdős–Rényi, probability that an edge exists |
| connected               | Loads if `exact_data = false`: ensure that the generated graph is connected                                           |

## Model arguments
Section described under `model_args`

| Name              | Description                                                                                 |
|-------------------|---------------------------------------------------------------------------------------------|
| mp_units          | Message passing numbers of neurons in consecutive layers of GCNN part                       |
| mp_activation     | Activation function at the end of each GCNN layer                                           |
| mlp_units         | Multilayered perceptron numbers of neurons in each layer                                   |
| mlp_activation    | Activation function for each MLP layer → new soft adjacency matrix                         |
| final_activation  | Final activation when calculating model output – new soft adjacency matrix                 |

## Loss calculation arguments
Section under `loss_args`. The asterisk * denotes parameters vital for the model.

| Name                     | Description                                                                                                                                                                                                                                                                                                    |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| delta                    | Small float ensuring that there is no division by 0 in the shortest path algorithm                                                                                                                                                                                                                             |
| gamma                    | Smoothing parameter used in the shortest path algorithm to introduce differentiability                                                                                                                                                                                                                         |
| * utility_scale          | When calculating utility of the base method (a.k.a.), and new (rail) method (b.k.), the model derives probability that a fraction of demand (flow) chooses new method following the utility                                                                                                                    |
| * priority_rail          | Utility for base mode is computed as distance multiplied by the `utility_scale` and for rail is computed as product of shortest distance (via new network), `utility_scale` and priority factor. E.g. `priority_rail = 0.5` implies that distance with rail of 2 is perceived as distance with base method of 1. |
| * loss_component_balance | Multiplier for _utility gain_. **Note:** Needs to be negative for proper minimisation formulation                                                                                                                                                                                                              |
| alpha_elu                | Distance reduction following introduction of the rail (including `priority_rail`) is moved to be non-negative. To avoid hard cut at 0 of ReLU method, the model applies elu with parametr `alpha_elu` and adds `alpha_elu`                                                                                     | 
| entropy_thresholds       | Thresholds at which new levels of entropy loss are introduced. It helps to increasingly push values to 0 or 1                                                                                                                                                                                                  |
| entropy_levels           | Levels of entropy loss at the respective threshold                                                                                                                                                                                                                                                             |
| mask_thresholds          | Thresholds for multiplier for penalty for each edge which is not allowed according to the _adjacency matrix_                                                                                                                                                                                                   |
| mask_levels              | Levels of the mask penalty at the respetive threshold                                                                                                                                                                                                                                                          |

## Optimiser arguments
Section under `optimizer_args`
| Name | Description |
| ---- | ------------- |
| lr | Learning rate |
| weight_decay | Weight decay for the Adam optimiser |

## Visualisation
Section under `visualization`
| Name | Description |
| ---- | ------------- |
| show_training_progress | Indicate whether to show learning progress each 50 epochs |
| show_training_output | At the end of training, show the output |
| save_training_output | Save the training output |
| label_coordinates_path | To plot the map when using actual geospatial data, provide dataframe with node features triples: name, x coord., y coord. |
| dpi | Pixel density for saved plots |

## Miscellaneous
Remaining parameters
| Name | Description |
| ---- | ------------- |
| num_epochs | Number of training epochs |
| seed | Seed for reproducibility | 


