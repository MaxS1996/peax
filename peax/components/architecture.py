from tensorflow import keras as keras
import tensorflow as tf
import numpy as np
import networkx as nx

from typing import Dict, List, Tuple, Union
import logging as log

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from . import graph_tools as gt


def extract_edges(
    routes: Dict[str, List[tf.keras.layers.Layer]]
) -> List[Tuple[str, str]]:
    """Extracts the edges from the neural network

    Args:
        routes (Dict[str, List[tf.keras.layers.Layer]]): possible routes throughout the neural network (input to output)

    Returns:
        List[Tuple[str, str]]: directed edges within the neural network
    """

    edges = list()
    for out, route in routes.items():
        node_current = None
        node_last = None
        for layer in route:
            node_last = node_current
            node_current = layer.name

            if not node_last is None:
                edges.append((node_last, node_current))

    return edges


def get_predecessor(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """Returns the preceeding layer for a given Keras layer.
    Does not work correctly, if a layer has multiple direct predecessors

    Args:
        layer (tf.keras.layers.Layer): the layer, whose predecessor should be extraced

    Returns:
        tf.keras.layers.Layer: predecessor of input layer
    """
    """
    TODO: adapt to handle layers with multiple predecessors
    """

    try:
        inp_layer = layer._inbound_nodes[0].inbound_layers
        if len(layer._inbound_nodes) > 1:
            log.warning(
                f"more than one inbound node detected in {layer.name}. Extracted routes might be incomplete!"
            )

        if isinstance(inp_layer, tf.keras.layers.Layer):
            return inp_layer
    except Exception:
        return None

    return None


def identify_output_layers(
    model: tf.keras.Model,
) -> Dict[str, tf.keras.layers.Layer]:
    """Extracts all output layers from Keras Model

    Args:
        model (tf.keras.Model): the Keras model that will be analyzed

    Returns:
        Dict[str, tf.keras.layers.Layer]: the extracted output layers, in a dict, the keys are the output names
    """
    output_layers = {}

    for output_tensor in model.outputs:
        output_layer_name = output_tensor.name.split("/")[0]
        output_layer = model.get_layer(output_layer_name)
        output_layers[output_layer_name] = output_layer

    return output_layers


def identify_routes(
    model: tf.keras.Model, reverse: bool = True
) -> Dict[str, List[tf.keras.layers.Layer]]:
    """Identifies routes through the Keras model.
    A route is a possible connection between an input and an output layer.
    The implementation currently only supports nodes with one input and multiple outputs

    Args:
        model (tf.keras.Model): The keras model that will be analyzed
        reverse (bool, optional): if the order should be reversed (starting with the output layer). Defaults to True.

    Returns:
        Dict[str, List[tf.keras.layers.Layer]]: output layers, in a dict, with the output names as keys
    """
    """
    identifies routes through network, starts from final layer,
    standard feed forward NN: single route with one input and one output,
    implementation currently only supports nodes with one input and multiple outputs
    """
    output_layers = identify_output_layers(model)

    routes = {}
    for name, layer in output_layers.items():
        routes[name] = []
        while layer != None:
            routes[name].append(layer)
            layer = get_predecessor(layer)

    # algorithm starts from output layers, if not desired, the lists need to be inversed
    if reverse == False:
        routes = {key: value[::-1] for key, value in routes.items()}
        log.info(
            f"Routes will be returned in correct order, starting from the input layers."
        )

    return routes


def is_branching(model: Union[tf.keras.Model, nx.Graph]) -> bool:
    """Checks if branches are present within the model.
    (A branch is a parallel subgraph, that does not fuse to the main subgraph again)

    Args:
        model (Union[tf.keras.Model, nx.Graph]): the Keras model that will be analyzed

    Returns:
        bool: true, if the model contains branches, false if not
    """
    if isinstance(model, tf.keras.Model):
        network_graph = gt.convert_to_graph(model=model)
    else:
        network_graph = model

    splits = 0
    merges = 0
    branching = False
    for node in network_graph.nodes:
        successors = list(network_graph.successors(node))
        if len(successors) > 1:
            branching = True
            splits += 1
        if node.layer_class == "merging":
            merges += 1

    if splits == merges:
        return branching

    return branching


def is_feed_forward(model: Union[tf.keras.Model, nx.Graph]) -> bool:
    """Checks, if the network is a simple feed forward architecture

    Args:
        model (Union[tf.keras.Model, nx.Graph]): the model that will be analyzed

    Returns:
        bool: True, if ffed_forward, False if not
    """

    if isinstance(model, tf.keras.Model):
        network_graph = gt.convert_to_graph(model=model)
    else:
        network_graph = model

    splits = 0
    merges = 0
    for node in network_graph.nodes:
        successors = list(network_graph.successors(node))
        if len(successors) > 1:
            splits += 1
        if node.layer_class == "merging":
            merges += 1
        if node.layer_class == "recurrent":
            return False

    if splits == merges:
        return True

    return False


def is_recurrent(model: Union[tf.keras.Model, nx.Graph]) -> bool:
    """Checks , if the given network is a recurrent neural network (RNN)

    Args:
        model (Union[tf.keras.Model, nx.Graph]): the model that will be analyzed

    Returns:
        bool: True, if it is an RNN, False otherwise
    """

    if isinstance(model, tf.keras.Model):
        network_graph = gt.convert_to_graph(model=model)
    else:
        network_graph = model

    for node in network_graph.nodes:
        if node.layer_class == "recurrent":
            return True

    return False


def clean_hybrid_graph(block_graph: nx.DiGraph) -> nx.DiGraph:
    """the new way to collapse branching sections leaves the split node out of the collapsed block.
    This can lead to odd behaviors, where a single activation/reshape/post-processing layer sits between residual blocks.
    This can cause problems with later processing stages, that add convolutions without activation functions or 
    unnecessary early exit branches before and after these acitvation layer blocks.
    To fix this, the activation blocks need to be merged with their predeccesors.

    Args:
        graph (nx.DiGraph): The hybrid or block graph that needs to be cleaned up.

    Returns:
        nx.DiGraph: A cleaned up version of the block-level graph, where acitvation blocks are fused with their predecessors,
        if they are not part of the final classification stage.
    """

    ## identify the blocks that need rewriting
    rewrites = list()
    for node in block_graph.nodes:
        if node.dominant_operation in ["activation"]: #, "reshape"]:
            rewrites.append(node)

    for node in rewrites:
        # we can already assume a single predecessor, as this is supposed to operate on cleaned block/hybrid graphs
        pred = list(block_graph.predecessors(node))[0]

        ## need to merge subgraphs of predecessor and current node into new layer-level subgraph
        new_subgraph = nx.DiGraph()
        new_subgraph.add_nodes_from(pred.subgraph)
        #new_subgraph.add_edges(pred.subgraph.edges)
        for u, v in pred.subgraph.edges:
            new_subgraph.add_edge(u, v)

        new_subgraph.add_nodes_from(node.subgraph)
        #new_subgraph.add_edges_from(node.subgraph)
        for u, v in node.subgraph.edges:
            new_subgraph.add_edge(u, v)

        pred_out = gt.get_first_output_node(pred.subgraph)
        succ_out = gt.get_first_input_node(node.subgraph)

        new_subgraph.add_edge(pred_out, succ_out)

        ## need to replace both block nodes with new single node
        attr = {**pred.attributes, **node.attributes}
        new_node = gt.BlockNode(new_subgraph, name=pred.name, attributes=attr)#, input_name=pred.input_node.name)
        block_graph.add_node(new_node)

        # might encounter the first block after the input
        if len(list(block_graph.predecessors(pred))) > 0:
            start_block = list(block_graph.predecessors(pred))[0]
            block_graph.add_edge(start_block, new_node)

        # might encounter the last block of the graph
        if len(list(block_graph.successors(node))) > 0:
            finish_block = list(block_graph.successors(node))[0]
            block_graph.add_edge(new_node, finish_block)
        
        block_graph.remove_nodes_from([pred, node])

    return block_graph

def get_residual_merge(graph: nx.DiGraph, split_node: gt.Node) -> gt.Node:
    """extracts the merge node for a residual split point in a given graph

    Args:
        graph (nx.DiGraph): the graph in which the residual block and the split and merge nodes exist
        split_node (gt.Node): the initial node of the residual block (node with multiple successors)

    Returns:
        gt.Node: the node where the branches merge again (node with multiple predecessors)
    """
    direct_successors = list(graph.successors(split_node))

    successors_dicts = []
    for successor in direct_successors:
        successors_dicts.append(nx.dfs_successors(graph, successor))

    intersection = set(successors_dicts[0])
    for successors_dict in successors_dicts[1:]:
        intersection.intersection_update(successors_dict)

    # Find the first node in the intersection that comes after the split node
    visited = set()
    queue = [split_node]
    while queue:
        node = queue.pop(0)
        visited.add(node)
        if node in intersection:
            return node
        for successor in graph.successors(node):
            if successor not in visited:
                queue.append(successor)

    return None

def insert_collapse_split_dummies(layer_graph: nx.DiGraph) -> nx.DiGraph:
    """additional pass for residual networks where some or all split nodes are also the merge node of the previous block.
    This function inserts a dummy layer_node at each of these positions, to split these two tasks across two layer_nodes, for better
    collapsability of the layer_node and a better structured block_node

    Args:
        layer_graph (nx.DiGraph): Layer-level graph representation of the model

    Returns:
        nx.DiGraph: Layer-level graph representation of the model with added dummy nodes that do not correspond to any Keras layer and do not perform any computation
    """

    split_nodes = [
        node
        for node in layer_graph.nodes()
        if len(list(layer_graph.successors(node))) > 1
    ]
    merge_nodes = [
        node
        for node in layer_graph.nodes()
        if len(list(layer_graph.predecessors(node))) > 1
    ]

    # these are the nodes that are split and merge nodes simultaneously
    common_nodes = set(split_nodes) & set(merge_nodes)

    # if there are no common nodes, we do not need to insert any dummies
    if len(common_nodes) == 0:
        return layer_graph

    for node in common_nodes:
        attrs = {
            "input_shapes": node.output_shapes,
            "output_shapes":node.output_shapes
            }
        dummy = gt.LayerNode(name=f"split_{node.name}", layer_type="split_dummy", macs=0, weight_count=0, attributes=attrs)
        layer_graph.add_node(dummy)
        succs = list(layer_graph.successors(node))

        for succ in succs:
            layer_graph.add_edge(dummy, succ)
            layer_graph.remove_edge(node, succ)

        layer_graph.add_edge(node, dummy)

    return layer_graph

    

def collapse_residuals(layer_graph: nx.DiGraph) -> nx.DiGraph:
    """converts the layer-level graph into a hybrid form, where the residual/skip connection subgraphs are converted to BlockNodes.

    Args:
        layer_graph (nx.DiGraph): Layer-level graph representation of the model

    Returns:
        nx.DiGraph: hybrid graph representation of the model
    """
    split_nodes = [
        node
        for node in layer_graph.nodes()
        if len(list(layer_graph.successors(node))) > 1
    ]
    merge_nodes = [
        node
        for node in layer_graph.nodes()
        if len(list(layer_graph.predecessors(node))) > 1
    ]

    common_nodes = set(split_nodes) & set(merge_nodes)
    if len(common_nodes) > 0:
        layer_graph = insert_collapse_split_dummies(layer_graph=layer_graph)
        split_nodes = [
        node
        for node in layer_graph.nodes()
        if len(list(layer_graph.successors(node))) > 1
        ]
        merge_nodes = [
            node
            for node in layer_graph.nodes()
            if len(list(layer_graph.predecessors(node))) > 1
        ]

        common_nodes = set(split_nodes) & set(merge_nodes)
        
    # split_nodes = [node for node in split_nodes if node not in common_nodes]
    # merge_nodes = [node for node in merge_nodes if node not in common_nodes]

    # cheapest way to find all residual blocks:
    hybrid_graph = nx.DiGraph()
    hybrid_graph.add_nodes_from(layer_graph.nodes())

    covered_nodes = set()

    block_count = 0
    for split in split_nodes:
        merge_node = get_residual_merge(layer_graph, split)

        residual_graph = nx.DiGraph()
        for path in nx.all_simple_paths(layer_graph, split, merge_node):
            residual_graph = nx.compose(
                residual_graph, layer_graph.subgraph(path)
            )
        
        covered_nodes.update(list(residual_graph.nodes()))
        block = gt.BlockNode(
            residual_graph,
            name=f"residual_{block_count}-{split.name}-{merge_node.name}",
        )

        hybrid_graph.add_node(block)
        inp = list(layer_graph.predecessors(split))[0]
        if len(list(hybrid_graph.predecessors(split))) > 0:
            inp = list(hybrid_graph.predecessors(split))[0]

        outps = list(layer_graph.successors(merge_node)) + list(hybrid_graph.successors(merge_node))
        for outp in outps:
            if not outp in covered_nodes:
                hybrid_graph.add_edge(block, outp)
        
        hybrid_graph.add_edge(inp, block)
        block_count += 1

    hybrid_graph.remove_nodes_from(covered_nodes)

    for source, target in layer_graph.edges():
        if (
            #source in split_nodes or
            source in merge_nodes
            or source in covered_nodes
        ):
            continue
        if (
            #target in split_nodes or
            target in merge_nodes
            or target in covered_nodes
        ):
            continue
        hybrid_graph.add_edge(source, target)

    #nx.draw(hybrid_graph, with_labels=True)
    #plt.savefig(f"hybrid.png")
    #plt.close()

    return hybrid_graph

def identify_blocks(model: tf.keras.Model) -> nx.DiGraph:
    """Identifies the high-level compute blocks in the network architecture

    Args:
        model (tf.keras.Model): The Keras model

    Returns:
        nx.DiGraph: a nx.DiGraph based representation of the network, where Nodes are the high-level compute blocks
    """
    network_graph = gt.convert_to_graph(model=model)

    input_layer_names = model.input_names
    if len(input_layer_names) > 1:
        log.warn("untested for multi-input models")

    split_nodes = [
        node
        for node in network_graph.nodes()
        if len(list(network_graph.successors(node))) > 1
    ]
    merge_nodes = [
        node
        for node in network_graph.nodes()
        if len(list(network_graph.predecessors(node))) > 1
    ]

    if len(split_nodes) > 0 and len(split_nodes) == len(merge_nodes):
        log.info("residual architecture detected")
        network_graph = collapse_residuals(network_graph)

    return convert_to_blocks(network_graph=network_graph)

def subgraph2block(network_graph: nx.DiGraph) -> nx.DiGraph:
    """TODO
    """

    block_graph = nx.DiGraph()
    block_node = None

    node = gt.get_first_input_node(network_graph)
    block = [node]
    block_input = node.name
    block_op = node.layer_class #"input"
    op_counter = {}
    prev = None

    sucessors = list(network_graph.successors(node))

    while len(sucessors) > 0:
        prev = node
        is_new_block = False
        node = list(network_graph.successors(node))[0]
        sucessors = list(network_graph.successors(node))
        

        if len(block) == 0:
            if isinstance(node, gt.LayerNode):
                is_new_block = False
                block = [node]
                block_input = node.name
                block_op = node.layer_class
            else:
                # add residual block
                block_graph.add_node(node)
                block_graph.add_edge(prev, node)

                block_node = node
                block = []
            continue

        if isinstance(node, gt.BlockNode):
            log.debug(f"{node} is a branching node and already a BlockNode")

            # put previous block into graph
            if not block_op in op_counter.keys():
                op_counter[block_op] = 0

            op_counter[block_op] += 1
            subgraph = nx.subgraph(network_graph, block)
            new_block_node = gt.BlockNode(
                subgraph=subgraph,
                name=f"{block_op}_{op_counter[block_op]-1}",
                #input_name=block_input,
            )
            block_graph.add_node(new_block_node)
            if block_node != None:
                block_graph.add_edge(block_node, new_block_node)

            # add residual block
            block_graph.add_node(node)
            block_graph.add_edge(new_block_node, node)

            # TODO: prepare for next node/block?
            block_node = node
            block = []

            continue

        if prev.layer_class == "input":
            is_new_block = True

        if block_op == node.layer_class and prev.layer_class in [
            "pooling",
            "merging",
            "reshape",
        ]:
            is_new_block = True

        if block_op != node.layer_class and not node.layer_class in [
            "support",
            "pooling",
            "activation",
        ]:
            is_new_block = True

        if node.layer_type == "Flatten":
            is_new_block = True

        if node.layer_type == "DepthwiseConv2D":
            is_new_block = True

        ## added for Conv1D network for binary ECG classification
        if block_op == 'convolution' and node.layer_class == "convolution" and node.layer_type == "Conv1D":
            is_new_block = True

        if is_new_block:
            if not block_op in op_counter.keys():
                op_counter[block_op] = 0

            op_counter[block_op] += 1
            subgraph = nx.subgraph(network_graph, block)
            new_block_node = gt.BlockNode(
                subgraph=subgraph,
                name=f"{block_op}_{op_counter[block_op]-1}",
                #input_name=block_input,
            )
            block_graph.add_node(new_block_node)
            if block_node != None:
                block_graph.add_edge(block_node, new_block_node)

            block_node = new_block_node
            block = [node]
            block_input = node.name
            block_op = node.layer_class

        else:
            block.append(node)
            continue

    if len(block) != 0: # need to handle case of branches that are not related to previously processed block node
        if not block_op in op_counter.keys():
            op_counter[block_op] = 0

        op_counter[block_op] += 1
        subgraph = nx.subgraph(network_graph, block)
        new_block_node = gt.BlockNode(
            subgraph=subgraph, name=f"{block_op}_{op_counter[block_op]-1}"
        )
        block_graph.add_node(new_block_node)
        if block_node != None:
            block_graph.add_edge(block_node, new_block_node)

    # cleaning up rough edges and fusions
    block_graph = clean_hybrid_graph(block_graph)

    return block_graph

def input_to_first_branch(network_graph : nx.DiGraph) -> nx.DiGraph:

    start = gt.get_first_input_node(network_graph)
    outputs = gt.get_output_nodes(network_graph)
    branching_nodes = gt.get_branching_nodes(network_graph)

    visited = set([start])
    stack = list(network_graph.successors(start))

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if node in branching_nodes:
                return network_graph.subgraph(visited)
            stack.extend(list(network_graph.successors(node)))
    
    return network_graph
            
def convert_to_blocks(network_graph : nx.DiGraph) -> nx.DiGraph:


    is_branching = False
    # check if network graph contains branches
    split_nodes = [
        node
        for node in network_graph.nodes()
        if len(list(network_graph.successors(node))) > 1
    ]
    merge_nodes = [
        node
        for node in network_graph.nodes()
        if len(list(network_graph.predecessors(node))) > 1
    ]

    if len(split_nodes) > len(merge_nodes):
        is_branching = True
        branch_nodes = gt.get_branching_nodes(network_graph)

    
    if is_branching:
        # if branching, we need to identify subgraph from input to first branching point
        # and convert it into block-level representation
        init_subgraph = subgraph2block(input_to_first_branch(network_graph))

        # recursive calling on branch successors
        branches = {}
        for branch_location in branch_nodes:
            branch_successors = list(network_graph.successors(branch_location))
            succ_branches = []
            for succ in branch_successors:
                for out_node in gt.get_output_nodes(network_graph):
                    if nx.has_path(network_graph, source=succ, target=out_node):
                        succ_elements = nx.shortest_path(network_graph, source=succ, target=out_node)
                        succ_branches.append(convert_to_blocks(network_graph.subgraph(succ_elements)))

            branches[branch_location] = succ_branches

        # glue these together?
        for branch_location, branch_subgraphs in branches.items():
            # unify graphs into newly combined graph
            temp_out = gt.get_output_nodes(init_subgraph)
            branch_block = None
            
            for node in temp_out:
                if branch_location in node.subgraph.nodes:
                    branch_block = node
                    break
            
            for branch_subgraph in branch_subgraphs:
                init_subgraph = nx.compose(init_subgraph, branch_subgraph)

                branch_in = gt.get_first_input_node(branch_subgraph)
                init_subgraph.add_edge(branch_block, branch_in)
                # identify block node that contains branch_location layer node
                # add edges between subgraphs

        return init_subgraph
    else:
        return subgraph2block(network_graph=network_graph)