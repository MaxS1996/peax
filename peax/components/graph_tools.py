from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Union
import logging as log
import copy
import json

import tensorflow as tf
import networkx as nx
import numpy as np

input_layers = set(["InputLayer"])
"""class names for Keras layers that can be input layers
"""

support_layers = set(
    [
        "BatchNormalization",
    ]
)
"""class names for Keras layers that are less relevant support layers (i.e. normalization)
"""

reshape_layers = set(
    [
        "Flatten",
        "Reshape",
        "RepeatVector",
        "Permute",
        "ZeroPadding2D",
    ]
)
"""Class names for Keras layers whose main purpose is to change the tensor shape"""

conv_layers = set(
    [
        "Conv1D",
        "Conv2D",
        "Conv3D",
        "SeparableConv1D",
        "SeparableConv2D",
        "DepthwiseConv2D",
        "Conv1DTranspose",
        "Conv2DTranspose",
        "Conv3DTranspose",
    ]
)
"""Class names for Convolutional layers"""

compute_layers = set(
    [
        "Dense",
        "LocallyConnected1D",
        "LocallyConnected2D",
    ]
)
""" Class names for compute layers"""

pooling_layers = set(
    [
        "MaxPooling1D",
        "MaxPooling2D",
        "MaxPooling3D",
        "AveragePooling1D",
        "AveragePooling2D",
        "AveragePooling3D",
        """"GlobalMaxPooling1D",
    "GlobalMaxPooling2D",
    "GlobalMaxPooling3D",
    "GlobalAveragePooling1D",
    "GlobalAveragePooling2D",
    "GlobalAveragePooling3D",""",
    ]
)
""" Class names for pooling layers"""

global_pooling_layers = set(
    [
        "GlobalMaxPooling1D",
        "GlobalMaxPooling2D",
        "GlobalMaxPooling3D",
        "GlobalAveragePooling1D",
        "GlobalAveragePooling2D",
        "GlobalAveragePooling3D",
    ]
)

""" Class names for global pooling layers"""

recurrent_layers = set(
    [
        "LSTM",
        "GRU",
        "SimpleRNN",
        "TimeDistributed",
        "Bidirectional",
        "ConvLSTM1D",
        "ConvLSTM2D",
        "ConvLSTM3D",
        "RNN",
    ]
)
""" Class names for recurrent layers"""

merging_layers = set(
    [
        "Concatenate",
        "Average",
        "Maximum",
        "Minimum",
        "Add",
        "Subtract",
        "Multiply",
        "Dot",
    ]
)
"""Class names for layers that merge two or more layers into one output layer by applying certain operations (+, -, max, min, x, concat, ...)"""

activation_layers = set(
    [
        "ReLU",
        "Softmax",
        "LeakyReLU",
        "PReLU",
        "ELU",
        "ThresholdedReLU",
        "Activation",
    ]
)
""" Class names for activation layers"""

ignored_layers = set(
    [
        "Dropout",
    ]
)
""" Layers that are not needed for inference"""


@dataclass
class Node:
    """Base class for graph nodes"""

    name: str
    attributes: Dict[str, object]

    def __init__(self, name: str, attributes: Dict[str, object] = None):
        self.name = name
        self.attributes = attributes

        if attributes is None:
            self.attributes = dict()

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if not name in self.attributes.keys():
                raise AttributeError(f"attribute {name} does not exist in this node")
            return self.attributes[name]

    def __hash__(self):
        val = frozenset(
            [
                self.name,
                frozenset(self.attributes.items()),
            ]
        )
        return hash(val)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        return False

    def __deepcopy__(self, memo):
        new_instance = Node(self.name, self.attributes)

        return new_instance

    def toJSON(self):
        data = {
            "name": self.name,
            "attributes": self.attributes,
        }

        return json.dumps(data)

    def asdict(self):
        return {"name": self.name, "attributes": self.attributes}

    def node_label(self):
        return self.name


@dataclass
class LayerNode(Node):
    """Node class the represents the invidual layer of the neural network in the Graph representation"""

    name: str
    layer_type: str
    layer_class: str
    macs: int
    weight_count: int
    attributes: Dict[str, object]

    def __init__(
        self,
        name: str,
        layer_type: str,
        macs: int,
        weight_count: int,
        attributes: Dict[str, object] = None,
    ):
        self.name = name
        self.layer_type = layer_type
        self.macs = macs
        self.weight_count = weight_count
        self.attributes = attributes

        self.input_shapes = None
        self.output_shapes = None
        if hasattr(self, "keras"):
            if isinstance(self.keras.input_shape, list):
                self.input_shapes = [x[1::] for x in self.keras.input_shape]
            elif isinstance(self.keras.input_shape, tuple):
                self.input_shapes = [self.keras.input_shape[1::]]
            
            if isinstance(self.keras.output_shape, list):
                self.output_shapes = [x[1::] for x in self.keras.output_shape if x is not None]
            elif isinstance(self.keras.output_shape, tuple):
                self.output_shapes = [self.keras.output_shape[1::]]

        if attributes is None:
            self.attributes = dict()

        self.layer_class = "unknown"

        if layer_type in ["split_dummy"]:
            self.layer_class = "optimization_dummy"
            if "input_shapes" in attributes.keys():
                self.input_shapes = attributes["input_shapes"]
                del self.attributes["input_shapes"]
            if "output_shapes" in attributes.keys():
                self.output_shapes = attributes["output_shapes"]
                del self.attributes["output_shapes"]

        elif layer_type in input_layers:
            self.layer_class = "input"

        elif layer_type in support_layers:
            self.layer_class = "support"

        elif layer_type in conv_layers:
            self.layer_class = "convolution"

        elif layer_type in compute_layers:
            self.layer_class = "compute"

        elif layer_type in pooling_layers:
            self.layer_class = "pooling"

        elif layer_type in global_pooling_layers:
            self.layer_class = "global_pooling"

        elif layer_type in recurrent_layers:
            self.layer_class = "recurrent"

        elif layer_type in merging_layers:
            self.layer_class = "merging"

        elif layer_type in reshape_layers:
            self.layer_class = "reshape"

        elif layer_type in activation_layers:
            self.layer_class = "activation"

        elif layer_type in ignored_layers:
            self.layer_class = "training_layer"

        if self.layer_class == "unknown":
            log.warning(f"layer {name} has a unknown layer class")

    def __hash__(self):
        val = frozenset(
            [
                self.name,
                self.layer_type,
                self.layer_class,
                self.macs,
                self.weight_count,
                frozenset(self.attributes.items()),
            ]
        )
        return hash(val)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, LayerNode):
            return (self.name == other.name) and (
                self.layer_type == other.layer_type
            )
        return False

    def __deepcopy__(self, memo):
        new_instance = LayerNode(
            self.name,
            self.layer_type,
            self.macs,
            self.weight_count,
            self.attributes,
        )
        new_instance.layer_class = self.layer_class

        return new_instance

    def asdict(self):
        config = None
        if hasattr(self, "keras"):
            config = self.keras.get_config()
        return {
            "name": self.name,
            "layer_type": self.layer_type,
            "layer_class": self.layer_class,
            "layer_config" : config,
            "macs": int(self.macs),
            "weight_count": int(self.weight_count),
            "attributes": self.attributes,
        }

    def toJSON(self):
        data = self.asdict()
        del data["attributes"]

        return json.dumps(data)

    def node_label(self):
        return self.name


class BlockNode(Node):
    """A class to represent the block node for the high-level graph representation"""

    name: str
    subgraph: nx.DiGraph
    macs: int
    weight_count: int
    #input_node: Union[Node, Set[Node]]
    input_shape: Tuple[int]
    output_shape: Tuple[int]
    dominant_operation: str
    attributes: Dict[str, object]

    def __init__(
        self,
        subgraph: nx.DiGraph,
        name: str = None,
        attributes: Dict[str, object] = None,
        #input_name: str = None,
    ):
        self.subgraph = subgraph
        self.macs = 0
        self.weight_count = 0
        '''if hasattr(input_name, "__iter__") and not isinstance(input_name, str):
            self.input_node = set()
            for name in input_name:
                self.input_node.add(get_node_for_layer(subgraph, name))
            self.input_node = frozenset(self.input_node)
        else:
            self.input_node = get_node_for_layer(subgraph, input_name)'''

        self.input_shape = None
        subgraph_input_node = get_first_input_node(subgraph)
        try:
            self.input_shape = subgraph_input_node.input_shapes[0]
        except AttributeError as e:
            self.input_shape = None
        except TypeError as e:
            self.input_shape = None

        self.output_shape = None
        subgraph_output_node = get_first_output_node(subgraph)
        try:
            self.output_shape = subgraph_output_node.output_shapes[0]
        except AttributeError as e:
            self.output_shape = None
        except TypeError as e:
            self.output_shape = None

        layer_classes = dict()
        for node in subgraph.nodes():
            if hasattr(node, "layer_class"):
                if node.layer_class in layer_classes.keys():
                    layer_classes[node.layer_class] += 1
                else:
                    layer_classes[node.layer_class] = 1

            if hasattr(node, "macs"):
                self.macs += node.macs
            if hasattr(node, "weight_count"):
                self.weight_count += node.weight_count

        if len(layer_classes.keys()) != 0:
            self.dominant_operation = max(layer_classes, key=layer_classes.get)
            if "convolution" in layer_classes.keys():
                self.dominant_operation = "convolution"
        else:
            self.dominant_operation = "unknown"

        self.attributes = attributes
        if attributes is None:
            self.attributes = dict()

        self.name = name
        if name == None:
            self.name = f"{self.dominant_operation}_{len(list(subgraph.nodes))}_{hash(subgraph)}"

        return

    def __hash__(self):
        val = frozenset(
            (
                self.name,
                self.subgraph,
                self.dominant_operation,
                #self.input_node,
                self.input_shape,
                self.output_shape,
                self.macs,
                self.weight_count,
                frozenset(self.attributes.items()),
            )
        )
        return hash(val)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, BlockNode):
            return (
                (self.name == other.name)
                and (self.subgraph == other.subgraph)
                and (self.dominant_operation == other.dominant_operation)
            )
        return False

    def __deepcopy__(self, memo):
        new_instance = BlockNode(
            self.subgraph, self.name, self.attributes, None
        )
        #new_instance.input_node = copy.deepcopy(self.input_node)
        new_instance.subgraph = copy.deepcopy(self.subgraph)
        new_instance.input_shape = self.input_shape
        new_instance.output_shape = self.output_shape
        new_instance.dominant_operation = self.dominant_operation

        return new_instance

    def asdict(self):
        return {
            "name": self.name,
            "subgraph": self.subgraph,
            "dominant_operation": self.dominant_operation,
            #"input_node": self.input_node,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "macs": int(self.macs),
            "weight_count": int(self.weight_count),
            "attributes": self.attributes,
        }

    def toJSON(self):
        from . import visualize_graph as viz
        data = self.asdict()
        del data["attributes"]
        data["subgraph"] = viz.transform_to_json(data["subgraph"])

        return json.dumps(data)

    def node_label(self):
        return self.name


def convert_to_graph(
    model: tf.keras.Model, store_keras_layer=True
) -> nx.DiGraph:
    """Converts a Keras Model to a layer-level DiGraph representation

    Args:
        model (tf.keras.Model): The Keras model that will be converted
        store_keras_layer (bool, optional): if True, the represented Keras layer will be attached to its LayerNode in this new graph representation. Defaults to True.

    Returns:
        nx.DiGraph: the Layer-level Graph representation of the NN
    """
    from . import resource as res
    from . import architecture as arch

    graph = nx.DiGraph()
    routes = arch.identify_routes(model=model)
    edges = arch.extract_edges(routes=routes)
    nodes = dict()

    for layer in model.layers:
        l_name = layer.name
        l_type = layer.__class__.__name__
        l_macs = res.get_layer_macs(layer)
        l_weights = 0
        for weight in layer.weights:
            l_weights += np.prod(weight.shape)

        attr = dict()
        if store_keras_layer:
            attr = {"keras": layer}

        node = LayerNode(
            name=l_name,
            layer_type=l_type,
            macs=l_macs,
            weight_count=l_weights,
            attributes=attr,
        )
        nodes[l_name] = node

    graph.add_nodes_from(nodes.values())

    for layer_name, layer_node in nodes.items():
        inbound_nodes = layer_node.keras._inbound_nodes

        for inbound in inbound_nodes:
            if isinstance(inbound.inbound_layers, list):
                for pred in inbound.inbound_layers:
                    try:
                        graph.add_edge(nodes[pred.name], nodes[layer_name])
                    except KeyError:
                        log.warn("one node of this edge is not a layer in the model, most likely a shared layer")

            else:
                pred = inbound.inbound_layers
                try:
                    graph.add_edge(nodes[pred.name], nodes[layer_name])
                except:
                    log.warn("one node of this edge is not a layer in the model, most likely a shared layer")

    """for start_node, end_node in edges:
        graph.add_edge(nodes[end_node], nodes[start_node])"""

    # graph.add_edges_from(edges)

    return graph


def get_node_for_layer(
    graph: nx.Graph, layer_name: str = "input_1"
) -> LayerNode:
    """Return the LayerNode that represents the given Keras Layer.
    The Keras layer is identified by its name

    Args:
        graph (nx.Graph): The layer-level graph representation in which the search will take place
        layer_name (str, optional): the name of the searched LayerNode. Defaults to "input_1".

    Returns:
        LayerNode: the found LayerNode that corresponds to the searched Keras layer
    """

    nodes = graph.nodes
    for node in nodes:
        if node.name == layer_name:
            return node


def get_first_input_node(graph: nx.DiGraph) -> Node:
    """Returns the first input node of the given Graph.
    Works for layer-level and high-level Graph representations.
    InputNodes are the Nodes in the directed graph that do not have predecessors

    Args:
        graph (nx.DiGraph): the graph to be analyzed

    Returns:
        Node: the first input node of the graph representation
    """
    return [node for node, in_degree in graph.in_degree() if in_degree == 0][0]


def get_first_output_node(graph: nx.DiGraph) -> Node:
    """Returns the first output node of the given Grpah.
    Works for layer-level and high-level Graph representations.
    OutputNodes are the Nodes in the directed graph that do not have sucessors

    Args:
        graph (nx.DiGraph): the graph to be analyzed

    Returns:
        Node: the first output node of the graph representation
    """
    return [
        node for node, out_degree in graph.out_degree() if out_degree == 0
    ][0]


def get_input_nodes(graph: nx.DiGraph) -> List[Node]:
    """
    Returns the first input node of the given Graph.
    Works for layer-level and high-level Graph representations.
    InputNodes are the Nodes in the directed graph that do not have predecessors

    Args:
        graph (nx.DiGraph): the graph to be analyzed

    Returns:
        List[Node]: the first input node of the graph representation
    """
    return [node for node, in_degree in graph.in_degree() if in_degree == 0]


def get_output_nodes(graph: nx.DiGraph) -> List[Node]:
    """
    Returns the output nodes of the given Graph.
    Works for layer-level and high-level Graph representations.
    Output Nodes are the Nodes in the directed graph that do not have sucessors.

    Args:
        graph (nx.DiGraph): the graph to be analyzed

    Returns:
        List[Node]: the output nodes of the graph representation
    """
    return [node for node, out_degree in graph.out_degree() if out_degree == 0]

def get_branching_nodes(graph: nx.DiGraph) -> List[Node]:
    """
    Returns the branching nodes of the given Graph.
    Works for layer-level and high-level Graph representations.
    Branching Nodes are the Nodes in the directed graph that do have more than one sucessor.

    Args:
        graph (nx.DiGraph): the graph to be analyzed

    Returns:
        List[Node]: the branching nodes of the graph representation
    """


    return [node for node, out_degree in graph.out_degree() if out_degree > 1]
