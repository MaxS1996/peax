import tensorflow as tf
import numpy as np
import networkx as nx

import pytest
from copy import deepcopy
from typing import Dict
from dataclasses import asdict

from peax.components.graph_tools import (
    Node,
    LayerNode,
    BlockNode,
    get_node_for_layer,
    get_first_input_node,
    get_first_output_node,
)


class TestNode:
    def test_init(self):
        node = Node("node1")
        assert node.name == "node1"
        assert node.attributes == {}

    def test_init_with_attributes(self):
        attributes = {"color": "blue", "shape": "circle"}
        node = Node("node1", attributes=attributes)
        assert node.name == "node1"
        assert node.attributes == attributes

    def test_getattribute(self):
        node = Node("node1", {"color": "blue"})
        assert node.color == "blue"

    def test_hash(self):
        node1 = Node("node1")
        node2 = Node("node1")
        assert hash(node1) == hash(node2)

    def test_str(self):
        node = Node("node1")
        assert str(node) == "node1"

    def test_eq(self):
        node1 = Node("node1")
        node2 = Node("node1")
        node3 = Node("node2")
        assert node1 == node2
        assert node1 != node3

    def test_deepcopy(self):
        node1 = Node("node1", {"color": "blue"})
        node2 = deepcopy(node1)
        assert node1 is not node2
        assert node1.name == node2.name
        assert node1.attributes == node2.attributes

    def test_to_dict(self):
        node = Node("layer1", {"foo": "bar"})
        expected_dict = {"name": "layer1", "attributes": {"foo": "bar"}}
        assert asdict(node) == expected_dict

    def test_node_label(self):
        node = Node("node1")
        assert node.node_label() == "node1"


class TestLayerNode:
    def test_init(self):
        layer_node = LayerNode(
            name="node1",
            layer_type="Conv3D",
            macs=1024,
            weight_count=32,
            attributes={"foo": "bar"},
        )

        assert layer_node.name == "node1"
        assert layer_node.layer_type == "Conv3D"
        assert layer_node.macs == 1024
        assert layer_node.weight_count == 32
        assert layer_node.attributes == {"foo": "bar"}

    def test_node_label(self):
        node = LayerNode("layer1", "Conv2D", 10, 20, {"foo": "bar"})
        assert node.node_label() == "layer1"

    def test_eq(self):
        node1 = LayerNode("layer1", "Conv2D", 10, 20, {"foo": "bar"})
        node2 = LayerNode("layer1", "Conv2D", 10, 20, {"foo": "baz"})
        node3 = LayerNode("layer2", "Conv2D", 10, 20, {"foo": "bar"})

        assert node1 == node2
        assert node1 != node3

    def test_hash(self):
        node1 = LayerNode("layer1", "Conv2D", 10, 20, {"foo": "bar"})
        node2 = LayerNode("layer1", "Conv2D", 10, 20, {"foo": "baz"})
        node3 = LayerNode("layer2", "Conv2D", 10, 20, {"foo": "bar"})

        nodes_dict = {node1: "node1", node2: "node2", node3: "node3"}

        assert nodes_dict[node1] == "node1"
        assert nodes_dict[node2] == "node2"
        assert nodes_dict[node3] == "node3"

    def test_to_dict(self):
        node = LayerNode("layer1", "Conv2D", 10, 20, {"foo": "bar"})
        expected_dict = {
            "name": "layer1",
            "layer_type": "Conv2D",
            "layer_class": "convolution",
            "macs": 10,
            "weight_count": 20,
            "attributes": {"foo": "bar"},
        }
        assert asdict(node) == expected_dict


class TestBlockNode:
    def test_init(self):
        subgraph = nx.DiGraph()
        node1 = Node("node1")
        node2 = Node("node2")

        subgraph.add_node(node1)
        subgraph.add_node(node2)

        subgraph.add_edge(node1, node2)
        block_node = BlockNode(subgraph=subgraph)

        assert block_node.subgraph == subgraph
        assert block_node.macs == 0
        assert block_node.weight_count == 0
        #assert block_node.input_node is None
        assert block_node.input_shape is None
        assert block_node.output_shape is None
        assert block_node.dominant_operation == "unknown"

    def test_hash(self):
        subgraph = nx.DiGraph()
        node1 = LayerNode("layer1", "Conv2D", 100, 200)
        node2 = LayerNode("layer2", "MaxPooling2D", 50, 100)
        subgraph.add_node(node1)
        subgraph.add_node(node2)
        subgraph.add_edge(node1, node2)

        block_node1 = BlockNode(subgraph)

        subgraph = nx.DiGraph()
        node3 = LayerNode("layer3", "Conv2D", 200, 400)
        node4 = LayerNode("layer4", "MaxPooling2D", 100, 200)
        subgraph.add_node(node3)
        subgraph.add_node(node4)
        subgraph.add_edge(node3, node4)

        block_node2 = BlockNode(subgraph)

        assert hash(block_node1) != hash(block_node2)

    def test_str(self):
        subgraph = nx.DiGraph()
        node1 = LayerNode("layer1", "Conv2D", 100, 200)
        node2 = LayerNode("layer2", "MaxPooling2D", 50, 100)
        subgraph.add_node(node1)
        subgraph.add_node(node2)
        subgraph.add_edge(node1, node2)

        block_node = BlockNode(subgraph)

        assert str(block_node) == block_node.name

    def test_eq(self):
        subgraph = nx.DiGraph()
        node1 = LayerNode("layer1", "Conv2D", 100, 200)
        node2 = LayerNode("layer2", "MaxPooling2D", 50, 100)
        subgraph.add_node(node1)
        subgraph.add_node(node2)
        subgraph.add_edge(node1, node2)

        block_node1 = BlockNode(subgraph)

        subgraph = nx.DiGraph()
        node3 = LayerNode("layer3", "Conv2D", 100, 200)
        node4 = LayerNode("layer4", "MaxPooling2D", 50, 100)
        subgraph.add_node(node3)
        subgraph.add_node(node4)
        subgraph.add_edge(node3, node4)

        block_node2 = BlockNode(subgraph)

        assert not (block_node1 == block_node2)

    # TODO: something is wrong here
    """def test_to_dict(self):
        # Create a BlockNode instance
        subgraph = nx.DiGraph()
        node1 = LayerNode('layer1', 'Conv3D', 100, 200)
        node2 = LayerNode('layer2', 'Dense', 50, 100)
        subgraph.add_node(node1)
        subgraph.add_node(node2)
        subgraph.add_edge(node1, node2)
        block_node = BlockNode(subgraph, 'block1', {'attr1': 'value1'})

        # Call the asdict function
        block_node_dict = asdict(block_node)

        print(block_node_dict)

        # Check if the returned dictionary has the expected keys and values
        expected_dict = {'subgraph': subgraph, 'macs': 150, 'weight_count': 300, 'input_node': None, 
                        'input_shape': None, 'output_shape': None, 'dominant_operation': 'compute', 
                        'attributes': {'attr1': 'value1'}, 'name': 'block1'}
        assert block_node_dict == expected_dict"""


def test_get_node_for_layer():
    # Create a graph to search
    graph = nx.DiGraph()
    layer1 = LayerNode("layer1", "Dense", 100, 200)
    layer2 = LayerNode("layer2", "Conv2D", 150, 300)
    graph.add_node(layer1)
    graph.add_node(layer2)

    # Test that a node can be found by name
    node = get_node_for_layer(graph, "layer1")
    assert node == layer1

    # Test that None is returned if node is not found
    node = get_node_for_layer(graph, "layer3")
    assert node is None


class TestGetInputNodes:
    def test_single_node_graph(self):
        # Test for a graph with a single input node
        graph = nx.DiGraph()
        graph.add_node("input_1")
        input_node = get_first_input_node(graph)
        assert input_node == "input_1"

    def test_multiple_node_graph(self):
        # Test for a graph with multiple input nodes
        graph = nx.DiGraph()
        graph.add_node("input_1")
        graph.add_node("input_2")
        input_node = get_first_input_node(graph)
        assert input_node == "input_1"


class TestGetOutputNodes:
    def test_single_node_graph(self):
        # Create a single node graph and get the output node
        graph = nx.DiGraph()
        layer1 = LayerNode("layer1", "Dense", 100, 200)
        graph.add_node(layer1)
        output_nodes = get_first_output_node(graph)

        # Check that the output node is the same as the only node in the graph
        assert output_nodes == layer1

    def test_multiple_node_graph(self):
        # Create a graph with multiple nodes and one output node
        graph = nx.DiGraph()
        graph.add_nodes_from(["node1", "node2", "node3"])
        graph.add_edges_from([("node1", "node2"), ("node2", "node3")])
        output_nodes = get_first_output_node(graph)

        # Check that the output node is the same as the node with no successors
        assert output_nodes == "node3"

    def test_multiple_output_nodes(self):
        # Create a graph with multiple output nodes
        graph = nx.DiGraph()
        graph.add_nodes_from(["node1", "node2", "node3"])
        graph.add_edges_from([("node1", "node2"), ("node1", "node3")])
        output_nodes = get_first_output_node(graph)

        # Check that both output nodes are returned
        assert "node2" == output_nodes

    # TODO: need to fix get_nodes functions to return more than one value
