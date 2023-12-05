import re

import jax.numpy as jnp


def parameters_partition_fn(
    module_name: str,
    _name: str,
    _value: jnp.array,
    first_trainable_gnn_layer: int,
    gnn_layer_name: str,
    model_name: str,
    train_esm_from: int,
) -> bool:
    """This function is passed to haiku.datastructures.partition(here, param_dict)
    to separate the parameters based on the condition given below.

    Args:
        module_name (str): The module name, follows the haiku convention of the modules
        _name (str): Name indicating the type of parameter.
        _value (jnp.array): Leaf node, the parameter array.
        first_trainable_gnn_layer (int): Must be baked in.

    Returns:
        bool: Whether to partition the current parameter into the first argument.
    """
    if "graph_neural_network" in module_name:
        if "gnn_final_mlp" in module_name:
            return True
        if "embed" in module_name and first_trainable_gnn_layer == 0:
            return True
        # this regex returns None if it is not in a GNN module, we therefore know it
        # belongs to a module after the GNN.
        regex = re.compile(".+?" + gnn_layer_name + "[_]?([0-9]*)\/.*")  # noqa: W605
        m = re.match(regex, module_name)
        if m is not None:
            (layer,) = m.groups()
            if layer == "":  # the first GNN layer has no layer number.
                layer = 0
            valid_layer = int(layer) >= first_trainable_gnn_layer
            rescaling = module_name.endswith("_rescaler")
            return valid_layer or rescaling
        else:
            return False
    else:
        if "esm2" in module_name:
            trainable = False
            if "attention_layer_" in module_name:
                layer = int(module_name.split("attention_layer_")[-1].split("/")[0])
                trainable = layer > train_esm_from
            return trainable
        else:
            return True
