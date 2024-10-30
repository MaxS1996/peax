from tensorflow import keras
import tensorflow as tf
import numpy as np
import logging as log
from typing import Dict, List, Tuple
import math

import networkx as nx
from . import architecture as arch

from . import graph_tools as gt


def _mac_conv3d(layer: keras.layers.Conv3D) -> int:
    """default function to estimate the MAC operations required to execute the given Conv3D layer

    Args:
        layer (keras.layers.Conv3D): the Conv3D layer of which the MAC count should be estimated

    Returns:
        int: the number of MAC operations as integer
    """
    # Get the number of filters, kernel size and strides
    F, d, h, w, _ = layer.get_weights()[0].shape
    S = layer.strides[1:-1]

    # Get the output shape
    D_out, H_out, W_out = layer.output_shape[2:]

    # Calculate the MACs
    MAC = D_out * H_out * W_out * d * h * w
    total_macs = F * MAC

    return total_macs


def _mac_conv2d(layer: keras.layers.Conv2D) -> int:
    """default function to estimate the MAC operations required to execute the given Conv2D layer

    Args:
        layer (keras.layers.Conv2D): the Conv2D layer of which the MAC count should be estimated

    Returns:
        int: the number of MAC operations as integer
    """
    filters = layer.filters
    groups = layer.groups
    kernel_size = layer.kernel_size
    out_shape = layer.output_shape
    inp_shape = layer.input_shape
    channel_pos = layer._channels_first

    if not channel_pos or channel_pos is "channels_last":
        inp_c = inp_shape[-1]
    else:
        inp_c = inp_shape[1]

    macs = np.prod(out_shape[1::]) * np.prod(kernel_size) * np.int64(inp_c // groups)

    return macs

def _mac_dw_conv2d(layer: keras.layers.DepthwiseConv2D) -> int:
    """default function to estimate the MAC operations required to execute the given Depthwise Conv2D layer

    Args:
        layer (keras.layers.DepthwiseConv2D): the DepthwiseConv2D layer of which the MAC count should be estimated

    Returns:
        int: the number of MAC operations as integer
    """
    kernel_size = layer.kernel_size
    out_shape = layer.output_shape

    macs = np.prod(out_shape[1::]) * np.prod(kernel_size)
    return macs

def _mac_conv1d(layer: keras.layers.Conv1D) -> int:
    """default function to estimate the MAC operations required to execute a Conv1D layer

    Args:
        layer (keras.layers.Conv1D): the layer of wich the MAC count should be estimated

    Returns:
        int: the number of MAC operations as integer
    """
    kernel_size = layer.kernel_size
    output_size = layer.output_shape

    mac_count = np.prod(output_size[1::]) * np.prod(kernel_size)
    return int(mac_count)

def _mac_dense(layer: keras.layers.Dense) -> int:
    """default function to estimate the MAC operations required to execute the given Dense layer

    Args:
        layer (keras.layers.Dense): the Dense layer of which the MAC count should be estimated

    Returns:
        int: the number of MAC operations as integer
    """
    return np.prod(layer.get_weights()[0].shape)


def _mac_max_pool2d(layer: keras.layers.MaxPooling2D) -> int:
    """default function to estimate the MAC operations required to execute the given Max Pooling 2D layer

    Args:
        layer (keras.layers.MaxPooling2D): the Max Pooling 2D layer of which the MAC count should be estimated

    Returns:
        int: the number of MAC operations as integer
    """
    pool_size = layer.pool_size
    out_shape = layer.output_shape

    macs = np.prod(pool_size) * np.prod(out_shape[1::])
    return macs

def _mac_max_pool1d(layer: keras.layers.MaxPooling1D) -> int:
    pool_size = layer.pool_size
    out_shape = layer.output_shape

    macs = np.prod(pool_size) * np.prod(out_shape[1::])
    return macs

def _mac_global_avg_pool1d(layer: keras.layers.GlobalAveragePooling1D) -> int:
    """default function to estimate the number of MAC oeprations to execute the given global average pooling 1D layer

    Args:
        layer (keras.layers.GlobalAveragePooling2D): the layer whose cost should be estimated

    Returns:
        int: the number of MAC operations as integer
    """
    pool_size = layer.input_shape[1::]
    out_shape = layer.output_shape[1::]

    macs = np.prod(pool_size) * np.prod(out_shape)
    return macs

def _mac_global_avg_pool2d(layer: keras.layers.GlobalAveragePooling2D) -> int:
    """default function to estimate the number of MAC oeprations to execute the given global average pooling 2D layer

    Args:
        layer (keras.layers.GlobalAveragePooling2D): the layer whose cost should be estimated

    Returns:
        int: the number of MAC operations as integer
    """
    out_shape = layer.output_shape[1::]
    inp_shape = layer.input_shape[1::]
    channel_pos = layer.data_format

    if channel_pos == "channels_last":
        n = out_shape[-1]
    else:
        n = out_shape[0]

    macs = np.prod(inp_shape)# * n
    return macs


def _mac_batch_norm(layer: keras.layers.BatchNormalization) -> int:
    """default function to estimate the MAC operations required to execute the given BatchNormalization layer

    Args:
        layer (keras.layers.BatchNormalization): the BatchNormalization layer of which the MAC count should be estimated

    Returns:
        int: the number of MAC operations as integer
    """
    input_shape = layer.input_shape[1::]
    values = np.prod(input_shape)
    macs = 4 * values

    return macs


def _mac_merge(layer: keras.layers.Multiply) -> int:
    """default function to estimate the MAC operations required to execute the given Multiply Merge layer"""
    output_shape = layer.output_shape[1::]
    factor_tensors = len(layer.input_shape)

    macs = np.prod(output_shape) * (factor_tensors - 1)

    return macs


def _mac_stub(layer: keras.layers.Layer) -> int:
    """a stub that can be used to ignore less relevant layers during the full network MAC estimate

    Args:
        layer (keras.layers.Layer): the layer that will be ignored
    Returns:
        int: always 0
    """
    return 0


def _lambda_warn(layer: keras.layers.Lambda) -> int:
    """warns user about Lambda layers in model, as their footprint cannot be estimated

    Args:
        layer (keras.layers.Lambda): the input layer

    Returns:
        int: always 0
    """

    log.warn(
        f"layer {layer} is a Lambda layer, its footprint cannot be estimated!"
    )
    return 0


_default_mac_estimators = {
    keras.layers.Conv3D: _mac_conv3d,
    keras.layers.Conv2D: _mac_conv2d,
    keras.layers.DepthwiseConv2D: _mac_dw_conv2d,
    keras.layers.Conv1D: _mac_conv1d,
    keras.layers.Dense: _mac_dense,
    keras.layers.MaxPool2D: _mac_max_pool2d,
    keras.layers.MaxPool1D: _mac_max_pool1d,
    keras.layers.AveragePooling2D: _mac_max_pool2d,
    keras.layers.AveragePooling1D: _mac_max_pool1d,
    keras.layers.GlobalAveragePooling1D: _mac_global_avg_pool1d,
    keras.layers.GlobalAveragePooling2D: _mac_global_avg_pool2d,
    keras.layers.BatchNormalization: _mac_batch_norm,
    keras.layers.Multiply: _mac_merge,
    keras.layers.Add: _mac_merge,
    keras.layers.InputLayer: _mac_stub,
    keras.layers.Flatten: _mac_stub,
    keras.layers.ReLU: _mac_stub,
    keras.layers.Activation: _mac_stub,
    keras.layers.ZeroPadding2D: _mac_stub,
    keras.layers.Reshape: _mac_stub,
    keras.layers.Dropout: _mac_stub,
    keras.layers.Lambda: _lambda_warn,
}
"""the dictionary for the default functions to estimate the MAC footprint of different layer types"""


def get_layer_macs(
    layer: keras.layers.Layer,
    estimation_functions: Dict[tf.keras.layers.Layer, callable] = None,
) -> int:
    """Estimates the MAC operations required to execute the given layer.
    Operates in two steps: 1) look-up of estimation function from estimation_functions
    2) Execute estimation function for given layer
    The estiamtion_functions dict can be None, then the default estimators will be used

    Args:
        layer (keras.layers.Layer): The layer that will be analyzed
        estimation_functions (Dict[tf.keras.layers.Layer, callable], optional): optional dict of specific estimation functions. Defaults to None.

    Returns:
        int: number of MAC operations
    """

    if estimation_functions is None:
        estimation_functions = _default_mac_estimators

    lut_value = type(layer)
    try:
        macs = estimation_functions[lut_value](layer)
    except KeyError as e:
        log.warning(f"no matching estimation function for {e.args[0]}")
        return 0

    return macs


def get_block_macs(
    block, estimation_functions: Dict[tf.keras.layers.Layer, callable] = None
) -> int:
    macs = 0
    for layer in list(block.subgraph.nodes):
        try:
            macs += get_layer_macs(
                layer=layer.keras, estimation_functions=estimation_functions
            )
        except AttributeError as e:
            log.warn(f"{layer} does not contain attribute 'keras', unable to calculate MAC count for it")

    return macs

def get_subgraph_macs(block_graph : nx.DiGraph, start_block : gt.BlockNode = None, end_block : gt.BlockNode = None, estimation_functions: Dict[tf.keras.layers.Layer, callable] = None):
    if start_block is None:
        start_block = gt.get_first_input_node(block_graph)
    if end_block is None:
        end_block = gt.get_first_output_node(block_graph)

    shortest_path = nx.shortest_path(block_graph, source=start_block, target=end_block)
    macs = 0
    for node in shortest_path:
        macs += get_block_macs(node, estimation_functions=estimation_functions)

    return macs

def get_model_macs(
    model: tf.keras.Model,
    estimation_functions: Dict[tf.keras.layers.Layer, callable] = None,
) -> Dict[str, int]:
    """Estimates the MAC footprint of each layer in the model, specialized estimator functions can be passed as estimation_function as a dict

    Args:
        model (tf.keras.Model): the Model that will be analyzed
        estimation_functions (Dict[tf.keras.layers.Layer, callable], optional): optional dict of specific estimation functions. Defaults to None.

    Returns:
        Dict[str, int]: Dict, using layer names as keys, and MAC estimate as integer as value
    """
    mac_estimates = {}
    for layer in model.layers:
        mac_estimates[layer.name] = get_layer_macs(layer)

    return mac_estimates


def get_output_macs(
    model: tf.keras.Model,
    estimation_functions: Dict[tf.keras.layers.Layer, callable] = None,
) -> Dict[str, int]:
    """Estimates the MAC operations that are required to reach each output during the inference,
    specialized estimator functions can be passed as estimation_function as a dict

    TODO: this needs fixing as residual connections throw the identify_routes function of!

    Args:
        model (tf.keras.Model): The model that will be analyzed
        estimation_functions (Dict[tf.keras.layers.Layer, callable], optional): optional dict of specific estimation functions. Defaults to None.

    Returns:
        Dict[str, int]: _description_
    """

    

    return get_output_macs_from_block_graph(arch.identify_blocks(model=model), estimation_functions=estimation_functions)

    '''routes = arch.identify_routes(model)

    macs = {}
    for name, route in routes.items():
        macs[name] = 0
        for layer in route:
            macs[name] += get_layer_macs(layer)

    return macs'''

def get_output_macs_from_block_graph(
    block_graph : nx.DiGraph,
    estimation_functions: Dict[tf.keras.layers.Layer, callable] = None,
) -> Dict[str, int]:
    
    macs = {}
    inp_block = [node for node in block_graph.nodes() if len(list(block_graph.predecessors(node))) == 0][0]
    output_blocks = [node for node in block_graph.nodes() if len(list(block_graph.successors(node))) == 0]

    for out_block in output_blocks:
        mac_sum = 0
        shortest_path = nx.shortest_path(block_graph, source=inp_block, target=out_block)
        for node in shortest_path:
            mac_sum += node.macs

        out_layer = [node for node in out_block.subgraph.nodes() if len(list(out_block.subgraph.successors(node))) == 0][0].keras.name

        macs[out_layer] = mac_sum

    return macs


### Storage and Memory estimations

_datatype_widths = {
    "uint1": 1,
    "uint2": 2,
    "uint4": 4,
    "uint8": 8,
    "uint16": 16,
    "uint32": 32,
    "int1": 1,
    "int2": 2,
    "int4": 4,
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "float8": 8,
    "float16": 16,
    "bfloat": 16,
    "bfloat16": 16,
    "float32": 32,
    "float": 32,
    "float64": 64,
    "double": 64,
}
"""Lookup Table for the bit-width of typically used datatypes"""


def get_layer_weight_count(layer: tf.keras.layers.Layer) -> int:
    """returns the number of weights that the given layer uses

    Args:
        layer (tf.keras.layers.Layer): The layer that will be analyzed

    Returns:
        int: number of weights
    """
    return layer.count_params()


def get_model_weight_count(model: tf.keras.Model) -> int:
    """Returns the number of weights that are used by the given model

    Args:
        model (tf.keras.Model): The model that will be analyzed

    Returns:
        int: number of weights
    """
    return model.count_params()


def get_layer_weight_size(
    layer: tf.keras.layers.Layer,
    datatypes: List[str] = [
        "int8",
        "int16",
        "float16",
        "float16",
        "float32",
        "float64",
    ],
) -> Dict[str, int]:
    """Returns the size of the layer in bytes, the estimate is based on the number of weights and desired datatypes.
    The size of a layer only considers the number of weights for now.
    For sub-byte types, dense packing is assumed

    Args:
        layer (tf.keras.layers.Layer): The layer that will be analyzed
        datatypes (List[str], optional): A list of datatypes that shall be considered. Defaults to ["int8", "int16", "float16", "float16", "float32", "float64"].

    Returns:
        Dict[str, int]: size estimates in byte for the given types as dict
    """
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    total_params = layer.count_params()
    size = {}
    for dtype in datatypes:
        size[dtype] = math.ceil(total_params * _datatype_widths[dtype] / 8)

    return size


def get_model_weight_size(
    model: tf.keras.Model,
    datatypes: List[str] = [
        "int8",
        "int16",
        "float16",
        "float16",
        "float32",
        "float64",
    ],
) -> Dict[str, int]:
    """returns the size of the model in bytes, based on the amount of weights and desired datatypes, assumes dense packing for sub-word types

    Args:
        model (tf.keras.Model): The model that will be analyzed
        datatypes (List[str], optional): A list of datatypes that shall be considered. Defaults to ["int8", "int16", "float16", "float16", "float32", "float64"].

    Returns:
        Dict[str, int]: size estimates in byte for the given types as dict
    """

    total_params = model.count_params()
    size = {}
    for dtype in datatypes:
        size[dtype] = math.ceil(total_params * _datatype_widths[dtype] / 8)

    return size


### Memory
def get_model_IFM_dims(model: tf.keras.Model) -> Dict[str, Tuple[int]]:
    """extracts the shapes of the intermediate feature maps (IFMs) of the model

    Args:
        model (tf.keras.Model): The model that will be analyzed

    Returns:
        Dict[str, Tuple[int]]: the layer input feature map shapes, addressed by the layer names
    """
    ifm_dims = {}

    for layer in model.layers:
        ifm_dims[layer.name] = layer.output_shape[1::]

    return ifm_dims


def get_model_IFM_count(model: tf.keras.Model) -> Dict[str, int]:
    """extracts the number of intermediate elements per layer for the given model

    Args:
        model (tf.keras.Model): The model that will be analyzed

    Returns:
        Dict[str, int]: the layer IFM element count addressed by the layer name
    """
    ifm_counts = {}

    for layer in model.layers:
        ifm_counts[layer.name] = np.prod(layer.output_shape[1::])

    return ifm_counts
