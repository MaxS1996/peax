from tensorflow import keras as keras
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

import networkx as nx

from dataclasses import dataclass
from typing import List, Dict, Union
import pathlib
from datetime import datetime

from . import architecture as arch
from . import resource as comp
from . import predictive as prd
from . import tflm_support as tflm

from . import graph_tools as gt


def block_graph(graph: nx.Graph, path: pathlib.Path) -> None:
    """visualizes a given block graph and stores the figure at the given path

    Args:
        graph (nx.Graph): the graph
        path (pathlib.Path): the path + file name of the destination
    """
    fig, ax = plt.subplots()

    pos = {}
    labels = {}
    for i, node in enumerate(graph.nodes()):
        pos[node] = (i * 1500, i * (-1))
        labels[node] = str(node.name)
    nx.draw_networkx(
        graph, pos, with_labels=True, labels=labels, node_size=500, ax=ax
    )

    fig.savefig(path)
    return


def graph(graph: nx.Graph, path: pathlib.Path) -> None:
    """visualizes a given NetworkX graph and stores the figure at the given path

    Args:
        graph (nx.Graph): the graph
        path (pathlib.Path): the path + file name of the destination
    """
    fig, ax = plt.subplots()

    pos = {}
    labels = {}
    for i, node in enumerate(graph.nodes()):
        pos[node] = (i * 1500, i * (-1))
        labels[node] = str(node.name)
    nx.draw_networkx(
        graph, pos, with_labels=True, labels=labels, node_size=500, ax=ax
    )

    fig.savefig(path)
    return

def transform_to_json(graph: nx.DiGraph) -> dict:
    """Transforms an nx.DiGraph into a dict representation that can written to disk as json file

    Args:
        graph (nx.DiGraph): the graph that needs to be converted

    Returns:
        dict: the dict-based representation that will be JSON seriable
    """

    transformed_graph = nx.DiGraph()
    edges = set()

    for node in graph.nodes:
        data = node.asdict()
        del data["attributes"]

        if isinstance(node, gt.BlockNode):
            data["subgraph"] = transform_to_json(data["subgraph"])

        transformed_graph.add_node(node.name, **data)

    for start, end in graph.edges:
        edges.add((start.name, end.name))

    transformed_graph.add_edges_from(edges)
    transformed_graph = nx.node_link_data(transformed_graph)

    return transformed_graph