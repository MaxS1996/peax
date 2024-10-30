import itertools
import math
from tensorflow import keras as keras
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

import hashlib
import gc
import os

from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Set
from typing_extensions import Self
import pathlib
from datetime import datetime
import copy
import networkx as nx
import logging as log

import json
import pickle

from jinja2 import Template

# from ..components import visualize_graph as vis
from ..components import architecture as arch
from . import accuracy as acc
from . import dataset as data
from ..components import resource as res
from ..components import graph_tools as gt
from ..components import predictive as pred
from ..hardware import processor as pr

from . import base

import peax.analysis as aNN


class EarlyExitReportSubmitter(base.ReportSubmitter):
    """class that is used for syntactic sugaring.
    Allows developers to create an EarlyExitReport with:
    'EarlyExitReport.submit_to(analysis).with_config(search_config="large")'
    """

    def __init__(self, analysis: aNN.ModelAnalysis, lazy: bool = False):
        super().__init__(analysis, lazy)

    def with_config(self, search_config:str="large") -> base.Report:
        ee_reporter, ee_uid = EarlyExitReport.closure(create_id=True, search_config=search_config)

        self.analysis.submit_reporter(ee_uid, ee_reporter)

        if self.lazy:
            return ee_uid
        else:
            #self.analysis.create_report(ee_uid, ee_reporter)
            return self.analysis.access_report(ee_uid)

class EarlyExitReport(base.Report):
    """Report that analysis the suitability of a Early Exit conversion for this model"""

    exit_search_config: str
    """the desired exit configuration, currently supported: 'large' and 'small'"""

    is_recommended: bool
    """is True, if Early Exits are recommended for the model"""
    is_ee: bool
    """is True, if model already has Early Exits"""
    classifier_subgraph: nx.DiGraph
    """the final classifier subgraph as its own graph object for easier handling"""

    classifier_macs: int
    """the number of MAC operations required to execute the classifier"""
    model_macs: int
    """the number of MAC operations required to execute the entire model"""
    feature_extraction_macs: int
    """the number of MAC operations required to execute the feature extraction subgraph"""

    recommendations: List[gt.BlockNode]
    """list of recommended Early Exit locations"""

    exit_configs: Dict[gt.BlockNode, List[keras.layers.Layer]]
    """optimized EE configurations for the different placement recommendations"""

    exit_precision : Dict[gt.BlockNode, float]

    exit_model_paths : Dict[gt.BlockNode, pathlib.Path]

    def __init__(self, analysis: aNN.ModelAnalysis, exit_search_config : str = "large") -> None:
        super().__init__(analysis)
        self.exit_search_config = exit_search_config
        self.is_recommended = False
        self.is_ee = False

        # check if it not already a EENN
        out_count = len(
            arch.identify_output_layers(model=analysis.keras_model)
        )
        if out_count > 1:
            self.is_recommended = True
            self.is_ee = True
            # need to split model
            return

        # check if it is a FFNN
        is_ff = arch.is_feed_forward(model=analysis.keras_model)
        if not is_ff:
            self.is_recommended = False
            self.is_ee = True
            return

        # graph analysis
        self._graph = (
            self.analysis.architecture.network_graph
        )  # gt.convert_to_graph(model=analysis.keras_model)
        self._block_graph = (
            self.analysis.architecture.block_graph
        )  # arch.identify_blocks(model=analysis.keras_model)

        # identify classifier / output block to reproduce as EE
        self.classifier_subgraph = self._get_block_classifier()
        # print(self.classifier_subgraph)

        self.classifier_macs = 0
        self.model_macs = list(self.analysis.compute.total_mac.values())[0]
        '''self.model_macs = np.sum(
            list(res.get_model_macs(model=analysis.keras_model).values())
        )'''
        for node in list(self.classifier_subgraph.nodes):
            self.classifier_macs += node.macs
        self.feature_extraction_macs = self.model_macs - self.classifier_macs
        # produce list of EENN placement options (based on compute/conv blocks)
        self.recommendations = self._get_exit_recommendations()

        if len(self.recommendations) > 0:
            self.is_recommended = True
        self.exit_configs = self._get_exit_configuration()
        self.exit_precision = {}
        self.exit_model_paths = {}

        self.subgraph_costs = self._get_subgraph_costs()
        self.exit_costs = self._get_early_classifier_cost()
        pass

    def _get_block_classifier(self) -> nx.DiGraph:
        """extracts the blocks that correspond to the final classifier of the original model

        Returns:
            nx.DiGraph: the classifier subgraph as its own nx.DiGraph
        """

        out_block = gt.get_first_output_node(self._block_graph)
        predecessor = list(self._block_graph.predecessors(out_block))[0]

        out_classifier_blocks = [out_block]

        ### check if we have an output activation function or if the model was trained with "from_logits"
        out_layer = gt.get_first_output_node(out_block.subgraph)
        if out_layer.layer_class in ["compute", "convolution"]:
            out_config = out_layer.keras.get_config()
            out_act = out_config["activation"]

            if out_act == "linear":
                log.warn("classifier subgraph appears to not contain a final activation function, trying to fix this...")
                ### select which function type would be appropriate
                if out_config["units"] > 1 and list(self.analysis.tasks.values())[0] == pred.Task.CLASSIFICATION:
                    ### multiclass problem?
                    act_func = tf.keras.activations.softmax # "softmax"
                
                out_layer.keras.activation = act_func

        if not out_block.dominant_operation in ["compute"]:
            out_classifier_blocks.append(predecessor)
            predecessor = list(self._block_graph.predecessors(predecessor))[0]

        while not predecessor.dominant_operation in ["compute", "convolution"]:
            if predecessor.dominant_operation in ["training_layer"]:
                predecessor = list(
                    self._block_graph.predecessors(predecessor)
                )[0]
                continue

            out_classifier_blocks.append(predecessor)
            predecessor = list(self._block_graph.predecessors(predecessor))[0]

        block_classifier_subgraph = self._block_graph.subgraph(
            out_classifier_blocks
        )
        block_classifier_subgraph.name = "classifier"

        return block_classifier_subgraph

    def _get_exit_options(self) -> List[gt.BlockNode]:
        """creates a list of all possible exit locations

        Returns:
            List[gt.BlockNode]: The found list
        """
        block_feature_graph = self._block_graph.subgraph(
            list(
                set(self._block_graph.nodes)
                - set(self.classifier_subgraph.nodes)
            )
        )
        options = [
            n
            for n in block_feature_graph.nodes()
            if n.dominant_operation in ["compute", "convolution"]
        ]

        return options
    
    def get_exit_subgraph(self, end_location : gt.BlockNode) -> tf.keras.models.Model:
        """A function to create a copy of the model up until the attachement point that has been specified by end_location.
        The returned model is not compiled.

        Args:
            end_location (gt.BlockNode): The position up to which the model will be created

        Returns:
            tf.keras.models.Model: The partial model being created
        """
        # get a copy of the backbone model
        orig_model = tf.keras.models.clone_model(self.analysis.keras_model)
        orig_model.build(self.analysis.keras_model.input_shape)
        orig_model.set_weights(self.analysis.keras_model.get_weights())
        input_tensors = orig_model.inputs

        # get the section that will be attached and where to attach it
        attach_layer_name = gt.get_first_output_node(end_location.subgraph).keras.name
        attach_layer = orig_model.get_layer(attach_layer_name)
        x = attach_layer.output

        partial_model = tf.keras.Model(inputs=input_tensors, outputs=[x])
        
        return partial_model

    def _get_exit_recommendations(self) -> List[gt.BlockNode]:
        """Creates a list of all recommended placement options

        Returns:
            List[gt.BlockNode]: The found list
        """
        recom = list()

        block_feature_graph = self._block_graph.subgraph(
            list(
                set(self._block_graph.nodes)
                - set(self.classifier_subgraph.nodes)
            )
        )
        options = self._get_exit_options()

        for option in options:
            is_recommended = True
            if (
                list(self._block_graph.successors(option))[0]
                in self.classifier_subgraph.nodes
            ):
                is_recommended = False
            ## TODO: add other criteria later

            if is_recommended:
                recom.append(option)

        return recom

    def _get_exit_configuration(
        self,
        size : str = None
    ) -> Dict[gt.BlockNode, List[keras.layers.Layer]]:
        """creates optimized Early Exit branches for the recommended positions

        Returns:
            Dict[gt.BlockNode, List[keras.layers.Layer]]: keys are the blocks after which the exits should be placed,
            values are the configured layers in the correct order.
        """
        if size == None:
            size = self.exit_search_config

        block_nodes = list(nx.topological_sort(self.classifier_subgraph))

        exit_num_dims = len(block_nodes[0].input_shape)
        exit_inp_shape = block_nodes[0].input_shape

        has_1d_conv = False
        
        for block in self._block_graph.nodes:
            if any(obj.layer_type == "Conv1D" for obj in block.subgraph.nodes):
                has_1d_conv = True
                break

        exit_layers = []
        for block in block_nodes:
            layer_nodes = list(nx.topological_sort(block.subgraph))
            exit_layers += layer_nodes

        adapter_num_dims = {}
        adapter_out_shapes = {}

        _conv_layer_constrs = {
            2 : keras.layers.Conv1D,
            3 : keras.layers.Conv2D,
            4 : keras.layers.Conv3D,
        }

        _pool_layer_constrs = {
            2 : keras.layers.AveragePooling1D,
            3 : keras.layers.AveragePooling2D,
            4 : keras.layers.AveragePooling3D,
        }

        for recom in self.recommendations:
            adapter_num_dims[recom] = len(recom.output_shape)
            adapter_out_shapes[recom] = recom.output_shape

        recom_exit_configs = dict()
        # recom_exit_costs = dict()

        has_global_pooling = any(
            obj.dominant_operation == "global_pooling" for obj in block_nodes
        )

        if size == "large":
            log.debug("still in testing, proceed with caution")
            ### need to insert conv layers and pooling layers to downsize IFMs
            ### but need to maintain small footprint

            for recom in self.recommendations:
                recom_exit_configs[recom] = []

                pool_size = 2
                strides = 1
                conv_cstr = _conv_layer_constrs[adapter_num_dims[recom]]
                pool_cstr = _pool_layer_constrs[adapter_num_dims[recom]]

                # calculate how much larger the IFM is compared to the exit input shape
                relative_size = np.prod(adapter_out_shapes[recom]) / np.prod(exit_inp_shape)

                #determin # of convolutional blocks that are going to be used
                if exit_num_dims == adapter_num_dims[recom]:
                    min_plane_dim = np.argmin(adapter_out_shapes[recom][0:2])
                    rel_plane_size = int(np.floor(adapter_out_shapes[recom][min_plane_dim] / exit_inp_shape[min_plane_dim]))

                    kernel_size = 3
                    #kernel_size = rel_plane_size
                    if rel_plane_size >= 5:
                        recom_exit_configs[recom].append(pool_cstr(pool_size=pool_size))
                    if  rel_plane_size >= 10:
                        strides = 2
                    conv_block_count = min(int(np.sqrt(adapter_out_shapes[recom][min_plane_dim] - exit_inp_shape[min_plane_dim])) // 2, 3)

                else:
                    kernel_size = 3
                    if relative_size >= 5:
                        recom_exit_configs[recom].append(pool_cstr(pool_size=pool_size))

                    conv_block_count = int(relative_size // 4)

                #conv_block_count = min(int(relative_size // 2), 3)

                filters = int(adapter_out_shapes[recom][-1]) // 4

                ## insert conv blocks according to relative size and dims
                ### Conv Block = Conv + ReLU + AvgPool
                for i in range(conv_block_count):
                    filters = max(filters // 2, 1)
                    recom_exit_configs[recom].append(conv_cstr(filters=filters, kernel_size=kernel_size, strides=strides, activation="relu"))
                    recom_exit_configs[recom].append(pool_cstr(pool_size=pool_size))
                    kernel_size = 3
                    strides = 1

                if exit_num_dims < adapter_num_dims[recom]:
                    recom_exit_configs[recom].append(keras.layers.Flatten())

                for layer in exit_layers:
                    config = layer.keras.get_config()
                    layer_class = type(layer.keras)
                    recom_exit_configs[recom].append(
                        layer_class.from_config(config)
                    )
                    
                #test_model = self.to_keras((recom, recom_exit_configs[recom]))
                #test_model.summary()
                continue


        elif size == "small":
            chronological_layer_nodes = []

            requires_global_pool = False
            if len(block_nodes[0].input_shape) == 2:
                requires_global_pool = block_nodes[0].input_shape[0] == (1)

            elif len(block_nodes[0].input_shape) == 3:
                requires_global_pool = block_nodes[0].input_shape[0:-1] == (1, 1)

            elif len(block_nodes[0].input_shape) == 4:
                requires_global_pool = block_nodes[0].input_shape[0:-1] == (
                    1,
                    1,
                    1,
                )

            # extract the used layers in the correct order
            for block in block_nodes:
                layer_nodes = list(nx.topological_sort(block.subgraph))
                chronological_layer_nodes += layer_nodes

            class_inp_size = np.prod(block_nodes[0].input_shape)
            inp_rel = []
            for recom in self.recommendations:
                adapter_shape = recom.output_shape

                recom_exit_configs[recom] = []
                # recom_exit_costs[recom] = 0
                # inp_rel.append(np.prod(recom.input_shape) / class_inp_size)

                # if IFM plane is too large -> add pooling layer to reduce workload
                if requires_global_pool and not has_global_pooling:
                    if len(adapter_shape) == 3:  # 2D
                        layer_node = keras.layers.GlobalAveragePooling2D

                    elif len(adapter_shape) == 4:  # 3D
                        layer_node = keras.layers.GlobalAveragePooling3D

                    elif len(adapter_shape) == 2:  # 1D
                        layer_node = keras.layers.GlobalAveragePooling1D

                    recom_exit_configs[recom].append(layer_node(keepdims=True))

                if (has_global_pooling or requires_global_pool) and adapter_shape[0] > 25:
                    kernel_size = max(
                        3,
                        adapter_shape[0] // block_nodes[0].input_shape[0] // 2,
                    )

                    strides = kernel_size // 2
                    filters = adapter_shape[-1]*kernel_size
                    if len(adapter_shape) == 3:  # 2D
                        layer_node = keras.layers.DepthwiseConv2D

                    elif len(adapter_shape) == 4:  # 3D
                        layer_node = keras.layers.Conv3D

                    elif len(adapter_shape) == 2:  # 1D
                        layer_node = keras.layers.Conv1D
                        kernel_size = kernel_size = min(
                            32,
                            adapter_shape[0] // block_nodes[0].input_shape[0] // 2,
                        )
                        filters = block_nodes[0].input_shape[-1]

                    #kernel_size = int(min(3, kernel_size))
                    if layer_node != keras.layers.DepthwiseConv2D:
                        recom_exit_configs[recom].insert(0, 
                            layer_node(kernel_size=kernel_size, strides=strides, filters=filters)
                        )
                    else:
                        recom_exit_configs[recom].insert(0, 
                            layer_node(kernel_size=kernel_size, strides=kernel_size)
                        )

                else:
                    if (
                        np.prod(adapter_shape) / class_inp_size > 2
                        and not has_global_pooling
                        and not requires_global_pool
                    ):
                        if len(adapter_shape) == 3:  # 2D
                            layer_node = keras.layers.MaxPooling2D

                        elif len(adapter_shape) == 4:  # 3D
                            layer_node = keras.layers.MaxPooling3D

                        elif len(adapter_shape) == 2:  # 1D
                            layer_node = keras.layers.MaxPooling1D

                        ## need to decide on pooling size
                        pool_size = max(
                            2,
                            np.ceil(
                                (np.prod(adapter_shape) / class_inp_size)
                                / recom.input_shape[-1]
                            ),
                        )
                        pool_size = int(min(8, pool_size))
                        recom_exit_configs[recom].append(
                            layer_node(pool_size=pool_size)
                        )

                    # if channel count is too large, add pointwise convolutional layer to reduce filter counts
                    if not has_1d_conv and adapter_shape[-1] / block_nodes[0].input_shape[-1] > 2:
                        ## need to decide on pooling type - based on dimensionality
                        if len(adapter_shape) == 3:  # 2D
                            layer_node = keras.layers.Conv2D

                        elif len(adapter_shape) == 4:  # 3D
                            layer_node = keras.layers.Conv3D

                        elif len(adapter_shape) == 2:  # 1D
                            layer_node = keras.layers.Conv1D

                        filter_count = block_nodes[0].input_shape[-1] // 2
                        kernel_size = 1
                        recom_exit_configs[recom].append(
                            layer_node(
                                kernel_size=kernel_size, filters=filter_count
                            )
                        )

                # if channel count is too low, add pointwise convolutional layer to increase filter counts
                if not has_1d_conv and block_nodes[0].input_shape[-1] / adapter_shape[-1] >= 2:
                    ## need to decide on pooling type - based on dimensionality
                    if len(adapter_shape) == 3:  # 2D
                        layer_node = keras.layers.Conv2D

                    elif len(adapter_shape) == 4:  # 3D
                        layer_node = keras.layers.Conv3D

                    elif len(adapter_shape) == 2:  # 1D
                        layer_node = keras.layers.Conv1D

                    filter_count = block_nodes[0].input_shape[-1] // 2
                    kernel_size = 1
                    recom_exit_configs[recom].append(
                        layer_node(
                            kernel_size=kernel_size, filters=filter_count
                        )
                    )

                for layer_node in chronological_layer_nodes:
                    config = layer_node.keras.get_config()
                    layer_class = type(layer_node.keras)
                    recom_exit_configs[recom].append(
                        layer_class.from_config(config)
                    )

                continue

        return recom_exit_configs

    def _get_subgraph_costs(self) -> Dict[gt.BlockNode, float]:
        """Estimates the cost of of the subgraph up to the possible attachement point.
        The cost is currently expressed as the estimated number of MAC operations.

        Returns:
            Dict[gt.BlockNode, int]: dict of the costs, keys are the blocks to after which the EEs can be attached, values are the proportional execution cost of the subgraphs
        """
        subgraph_costs = {}
        block_graph = self.analysis.architecture.block_graph

        inp_block = [node for node in block_graph.nodes() if len(list(block_graph.predecessors(node))) == 0][0]

        for recom in self.recommendations:
            mac_sum = 0
            shortest_path = nx.shortest_path(block_graph, source=inp_block, target=recom)
            for node in shortest_path:
                mac_sum += node.macs
            subgraph_costs[recom] = mac_sum #/ list(self.analysis.compute.total_mac.values())[0]
        
        self.subgraph_costs = subgraph_costs

        return subgraph_costs
    
    def _get_early_classifier_cost(self) -> Dict[gt.BlockNode, float]:
        """Estimates the cost of the classifier branches that are suggested for the attachement locations.
        The cost is currently expressed as the estimated number of MAC operations.

        Returns:
            Dict[gt.BlockNode, int]: dict of the costs, keys are the blocks to after which the EEs can be attached, values are the proportional execution cost of the branches
        """

        ee_costs = {}

        #self.exit_configs contains the layers of each EE
        for location, ee_branch in self.exit_configs.items():
            model = self.to_keras((location, ee_branch))

            ee_macs = 0
            for layer in model.layers:
                ee_macs += res.get_layer_macs(layer)

            ee_costs[location] = ee_macs #/ list(self.analysis.compute.total_mac.values())[0]

        return ee_costs

    def to_keras(
        self,
        exit_configuration: Tuple[gt.BlockNode, List[keras.layers.Layer]],
        debug: bool = False,
    ) -> keras.models.Model:
        """function to turn stored layers and their configuration for EE branch into a Keras Model

        Args:
            exit_configuration (Tuple[gt.BlockNode, List[keras.layers.Layer]]): the exit configuration as it is stored in the dictionary of the report
            debug (bool, optional): function prints additional debug info if enabled. Defaults to False.

        Returns:
            keras.models.Model: The keras model for the EE branch
        """
        pos, layers = exit_configuration

        inp = tf.keras.Input(shape=pos.output_shape)

        layer_class = type(layers[0])
        layer_config = layers[0].get_config()
        layer_config["name"] = f"{layer_config['name']}_0"
        x = layer_class.from_config(layer_config)(inp)

        for i, layer in enumerate(layers[1:-1]):
            layer_class = type(layer)
            config = layer.get_config()
            if isinstance(layer, tf.keras.layers.Dense):
                units = config["units"] // 4
                config["units"] = max(units, 16)

            if isinstance(layer, tf.keras.layers.Conv2D):
                filters = config["filters"]
                config["filters"] = max(filters, 8)
            config["name"] = f"{config['name']}_{i+1}"
            x = layer_class.from_config(config)(x)

        # final output layer, not handled in loop, to avoid changing number of classes
        layer_class = type(layers[-1])
        config = layers[-1].get_config()
        config["name"] = f"{pos.name}_{config['name']}_{len(layers)-1}"
        x = layer_class.from_config(config)(x)
        model = tf.keras.Model(inputs=inp, outputs=x, name=pos.name)

        ### TODO: need to handle networks that have been trained with "from_logits" in cost function, as their final activation function in the architecture will be "linear"

        if debug:
            model.summary()

        return model

    def _determine_ee_training_config(
        self, ee_model: keras.models.Model, batch_size : int = 256
    ) -> Dict[str, str]:
        """**experimental** Helper function to suggest EE branch compilation configuration if not provided by user

        Args:
            ee_model (keras.models.Model): _description_

        Returns:
            Dict[str, str]: _description_
        """

        # needs to determine loss, optimizer and metrics for the EE branch
        # these depend on the task, the number of weights and the depth of the branch

        #num_weights = ee_model.count_params()
        #learning_rate = 0.005 * batch_size / math.sqrt(num_weights)
        learning_rate = 0.001
        log.debug("learning rate", learning_rate)

        if (
            list(self.analysis.tasks.values())[0]
            == pred.Task.BINARY_CLASSIFICATION
        ):
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                #weight_decay=None,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                #use_ema=False,
                #ema_momentum=0.99,
                #ema_overwrite_frequency=None,
                #jit_compile=True,
                name="Adam",
            )
            loss = "binary_crossentropy"
            metrics = [tf.keras.metrics.BinaryAccuracy()]

        elif list(self.analysis.tasks.values())[0] == pred.Task.CLASSIFICATION:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                #weight_decay=None,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                #use_ema=False,
                #ema_momentum=0.99,
                #ema_overwrite_frequency=None,
                #jit_compile=True,
                name="Adam",
            )
            loss = "categorical_crossentropy"
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

        elif list(self.analysis.tasks.values())[0] == pred.Task.REGRESSION:
            optimizer = "adam"
            loss = "mean_squared_error"
            metrics = ["mae", "r2"]

        elif list(self.analysis.tasks.values())[0] == pred.Task.SEGMENTATION:
            if len(ee_model.output.shape) > 3:
                classes = ee_model.output.shape[-1]
                optimizer = "adam"
                loss = "categorical_crossentropy"
                metrics = [
                    tf.keras.metrics.MeanIoU(num_classes=classes),
                    tf.keras.metrics.MeanDice(),
                ]
            else:
                optimizer = "adam"
                loss = "binary_crossentropy"
                metrics = [
                    tf.keras.metrics.MeanIoU(num_classes=2),
                    tf.keras.metrics.MeanDice(),
                ]

        else:
            log.error(
                "unable to suggest a compile configuration for branches on {self.analysis.name}"
            )
            return None

        value_dict = {
            "optimizer": optimizer,
            "loss": loss,
            "metrics": metrics,
            # "run_eagerly": True,
        }

        return value_dict
    
    '''def _cache_IR_data(self, intermediate_dataset:tf.data.Datset):
        intermediate_dataset.save(f"{self.analysis.cache_dir}/IR")'''

    def store_trained_EE(self, model : tf.keras.Model, history : Dict) -> bool:
        """Helper function to store the EE branch as Keras model in the cache directory

        Args:
            model (tf.keras.Model): the Keras model that should be saved to disk
            history (Dict): its training history

        Returns:
            bool: True if successful, False otherwise
        """
        path = self.analysis.cache_dir / self.analysis.name / model.name
        log.info(f"storing trained EE model at {path}")

        path.mkdir(parents=True, exist_ok=True)

        try:
            model.save(filepath=str(path), overwrite=True, save_format="keras")
        except:
            return False
        
        if history is None:
            return True
        
        with open(path / "history.log", "wb") as f:
            pickle.dump(history, f)
        
        self.exit_model_paths[model.name] = path
        return True

    def _get_EE_metrics(self, ee_model : tf.keras.Model, test_data : tf.data.Dataset, threshold_range : List[float] = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]) -> Dict[str, float]:
        """Evaluates the EE branch for the threshold-specific accuracies and termination rates

        Args:
            ee_model (tf.keras.Model): the EE branch model that shall be evaluated
            test_data (tf.data.Dataset): the test dataset, must fit to the input of the EE branch model (went through the previous part of the backbone model)
            threshold_range (List[float], optional): _description_. Defaults to [0.0, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99].

        Returns:
            Dict[str, float]: A dict that contains the raw predicition result for the test set and accuracies as well as termination rates for all evaluated thresholds
        """
        
        y_pred = ee_model.predict(test_data)
        y_labels = np.argmax(y_pred, axis=-1)
        y_conf = np.amax(y_pred, axis=-1)

        total_samples = len(y_labels)
        if test_data.element_spec[0].shape[0] is None:
            test_data = test_data.unbatch()
        true_labels = np.stack([y.numpy() for x,y in test_data], axis=0)

        if isinstance(true_labels[0], np.ndarray):
            true_labels = np.argmax(true_labels, axis=-1)

        result = {}
        #result["raw"] = y_pred

        for threshold in threshold_range:
            # TODO: adapt to non-classification tasks

            # get accuracy and termination rate for each threshold
            correct_predictions = 0
            terminated_samples = 0

            for i in range(total_samples):
                if y_conf[i] >= threshold:
                    # Predicted label for the sample exceeds the confidence threshold
                    if y_labels[i] == true_labels[i]:
                        correct_predictions += 1
                    terminated_samples += 1                    

            if terminated_samples > 0:
                result[threshold] = {}
                result[threshold]["accuracy"] = correct_predictions / terminated_samples
                result[threshold]["termination_rate"] = terminated_samples / total_samples
            
        return result

    def evaluate_precision(
        self,
        exit_configuration: Tuple[gt.BlockNode, List[keras.layers.Layer]],
        training_datasetReport: data.DatasetReport,
        test_datasetReport: data.DatasetReport,
        batch_size: int = 1,
        compile_config: Dict = None,
        training_config: Dict = None,
        callbacks : Set[tf.keras.callbacks.Callback] = None
    ) -> float:
        """Experimental function to estimate the accuracy of an early exit.
        It trains the exits separately without changing the backbone and evaluates them individually.
        While this strategy will be different from the final joint training and is unlikely to achieve the same accuracy level for the exits,
        it can give an early preview of the possible accuracies and can be performed faster than the joint training due to the smaller size of the EE branches
        """

        if callbacks is None:
            callbacks = []

        def _get_cache_path(layer:tf.keras.layers.Layer, dataset:tf.data.Dataset, data_name : str) -> pathlib.Path:
            ### need to rewrite this, because this does not work / sucks
            return self.analysis.cache_dir / self.analysis.name / "ee" / layer.name / data_name

        def is_cached(layer:tf.keras.layers.Layer, dataset:tf.data.Dataset, data_name : str):
            cache_path = _get_cache_path(layer, dataset, data_name=data_name) / "dataset_spec.pb"
            return cache_path.exists() and cache_path.is_dir()
        
        def load_cache(layer:tf.keras.layers.Layer, dataset:tf.data.Dataset, data_name : str):
            cache_path = _get_cache_path(layer, dataset, data_name=data_name)
            data = tf.data.Dataset.load(str(cache_path))
            return data
        
        def write_cache(layer:tf.keras.layers.Layer, dataset:tf.data.Dataset, data_name : str):
            cache_path = _get_cache_path(layer, dataset, data_name=data_name)
            cache_path.mkdir(parents=True, exist_ok=True)
            return dataset.cache(str(cache_path))#save(str(cache_path))

        if exit_configuration[0] in self.exit_precision.keys():           
            return self.exit_precision[exit_configuration[0]]

        pos, layers = exit_configuration

        batch_size = max(1, batch_size)

        ### build EE branch as Keras model if necessary
        if isinstance(layers, list):
            model = self.to_keras(exit_configuration)
        else:
            model = layers

        attach_layer = gt.get_first_output_node(pos.subgraph)

        ### create training and test set by adding another output to backbone and storing the IFMs
        ## create keras model to extract IFMs
        ifm_source = attach_layer.keras
        intermediate_layer_model = keras.models.Model(
            inputs=self.analysis.keras_model.inputs,
            outputs=self.analysis.keras_model.get_layer(
                ifm_source.name
            ).output,
        )

        def preprocess_data(x, y):
            with tf.device('/device:CPU:0'):
                x = intermediate_layer_model(x, training=False)
            #y = tf.squeeze(tensor, axis=0)
            ### need to account for not one-hot encoded y here
            return x, y
        
        #training data should be unbatched at this point
        '''
        if training_datasetReport.data.element_spec[0].shape[0] is None:
            training_data = training_datasetReport.data.unbatch()
        else:
            training_data = training_datasetReport.data
        '''
        training_data = training_datasetReport.data
        #training_data = training_data.prefetch(tf.data.AUTOTUNE)

        '''if test_datasetReport.data.element_spec[0].shape[0] is None:
            test_data = test_datasetReport.data.unbatch()
        else:
            test_data = test_datasetReport.data'''
        test_data = test_datasetReport.data
        #test_data = test_data.prefetch(tf.data.AUTOTUNE)
        
        # check cache
        if is_cached(layer=ifm_source, dataset=training_data, data_name=training_datasetReport.name):
            training_data = load_cache(layer=ifm_source, dataset=training_data, data_name=training_datasetReport.name).prefetch(tf.data.AUTOTUNE)
        ## no cache entry found:
        else:
            training_data = training_data.batch(batch_size)#.prefetch(tf.data.AUTOTUNE)
            training_data = training_data.map(
                    preprocess_data, num_parallel_calls=tf.data.AUTOTUNE
                ).prefetch(tf.data.AUTOTUNE)
            
            ###TODO: fix this, seems to be running out of memory for GSCv3 on DS-CNN_L
            #write_cache(layer=ifm_source, dataset=training_data, data_name=training_datasetReport.name)

        if is_cached(layer=ifm_source, dataset=test_data, data_name=test_datasetReport.name):
            test_data = load_cache(layer=ifm_source, dataset=test_data, data_name=test_datasetReport.name)
        else:

            test_data = test_data.batch(batch_size).map(
                    preprocess_data, num_parallel_calls=tf.data.AUTOTUNE
                ).prefetch(tf.data.AUTOTUNE)
            
            ###TODO: fix this, seems to be running out of memory for GSCv3 on DS-CNN_L
            #write_cache(layer=ifm_source, dataset=test_data, data_name=test_datasetReport.name)

        #training_data = training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        #test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        ## training of EE branch
        if compile_config is None:
            compile_config = self._determine_ee_training_config(model, batch_size=batch_size)

        # tf.config.run_functions_eagerly(True)

        model.summary()
        model.compile(**compile_config)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=compile_config["metrics"][0].name,
            patience=1,
            min_delta=0.01,
            baseline=1 / model.output.shape[-1], #baseline should be above 1/len(classes) for classification tasks
            start_from_epoch=2,
        )

        callbacks.append(early_stopping)
        
        try:
            history = model.fit(
                training_data,
                #batch_size=batch_size,
                epochs=7,
                callbacks=callbacks,
            )
        except Exception as e:
            log.error(
                f"training of {model.name} failed due to {e}, skipping evaluation..."
            )
            exit_metrics = self._get_EE_metrics(model, test_data)
            self.exit_precision[exit_configuration[0]] = (0, exit_metrics)
            #self.store_trained_EE(model, None)
            return 0, {}

        if early_stopping.stopped_epoch != 0:
            log.info(
                f"Model stopped training after epoch {early_stopping.stopped_epoch} due to early stopping."
            )
        performance = (
            history.history[compile_config["metrics"][0].name][-1]
        )

        # TODO: need to find a fitting alternative here that works with non-classification tasks
        if str(list(self.analysis.tasks.values())[0]) == "TASK.CLASSIFICATION" and performance < 1 / model.output.shape[-1]:
            log.warn(
                "training of EE branch was unsuccessful, stopping evaluation now!"
            )

            exit_metrics = self._get_EE_metrics(model, test_data)
            self.exit_precision[exit_configuration[0]] = (0, exit_metrics) #performance
            self.store_trained_EE(model, history)
            return 0, {} #self.exit_precision[exit_configuration[0]]

        # return history.history[compile_config["metrics"][0].name][-1] * 100

        eval = model.evaluate(test_data, batch_size=batch_size)
        exit_metrics = self._get_EE_metrics(model, test_data)

        #TODO: implement check if accuracy/whatever is better than randomly guessing the label

        self.exit_precision[exit_configuration[0]] = (eval[-1], exit_metrics)
        self.store_trained_EE(model, history)
        return eval[-1], exit_metrics
    
    def access_id(self) -> str:

        return self.create_unique_id(
            self.exit_search_config
        )

    def render_summary(self, folder_path: Union[str, pathlib.Path] = None):
        """Creates the HTML file for the summary overview

        Args:
            folder_path (Union[str, pathlib.Path], optional): folder, in which the file and auxiliary data will be stored. Defaults to None.
        """
        _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'

        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        file_name = f"report_early_exit_{self.access_id()}.html"

        with open(_template_path / "early_exit_report.html", "r") as file:
            template = Template(file.read())

        summary = self.dump(folder_path=folder_path)
        summary["recommendations"] = self.recommendations
        summary["exit_configs"] = self.exit_configs
        summary["exit_precisions"] = self.exit_precision
        summary["exit_costs"] = self.exit_costs
        summary["subgraph_costs"] = self.subgraph_costs

        block_node_data = [
            {"id": str(node), "label": str(node)}
            for node in self.classifier_subgraph.nodes()
        ]
        block_edge_data = [
            {"from": str(edge[0]), "to": str(edge[1])}
            for edge in self.classifier_subgraph.edges()
        ]
        block_graph_data = {"nodes": block_node_data, "edges": block_edge_data}
        summary["classifier_subgraph"] = block_graph_data

        text = f"{self.analysis.name} is a Keras model with {summary['model_macs']} MAC operations in the backbone \
                and {summary['classifier_macs']} MAC operations in the classifier.<br />"

        if summary["is_early_exit"]:
            text += "<b>The model already is an Early Exit Neural Network (or uses multiple exits)!</b>"
        else:
            text += "The model is not yet an Early Exit Neural Networks.<br/>"
            if summary["is_recommended"]:
                text += "The report <b>recommends</b> a conversion to an Early Exit Neural Network Architecture!"
                text += f"{len(summary['recommendations'])} positions have been found on which an Early Exit could be added."
            else:
                text += "This report does <b>not recommend</b> a conversion to an Early Exit Neural Network Architecture!"
        summary["text"] = text

        # Render the template with the summary data
        try:
            html = template.render(summary=summary)
            # Save the generated HTML to a file
            with open(folder_path / file_name, "w") as file:
                file.write(html)
        except Exception as e:
            log.error(f"error {e} occured while trying to print summary of report")

        return f"Early Exit {self.exit_search_config}", file_name

    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> Dict:
        """Writes its most important information to disk.
        Uses JSON to make the extracted information machine-readable

        Args:
            folder_path (Union[str, pathlib.Path], optional): _description_. Defaults to None.

        Returns:
            Dict: Members of the Report
        """
        from ..components import visualize_graph as viz
        summary = {
            "report_type": "Early Exit",
            "name": self.analysis.name,
            "unique_id" : self.access_id(),
            "creation_date": str(self.analysis.creation_time),
            "recommendations": [x.name for x in self.recommendations],
            "search_config": self.exit_search_config,
            "is_early_exit" : self.is_ee,
            "is_recommended" : self.is_recommended,
            "exit_configs" : [(key.name, [(str(type(layer)), layer.get_config()) for layer in value]) for key, value in self.exit_configs.items()],
            "feature_extraction_macs" : int(self.feature_extraction_macs),
            "classifier_macs" : int(self.classifier_macs),
            "model_macs": int(self.model_macs),
            "classifier_subgraph" : viz.transform_to_json(self.classifier_subgraph),
            "exit_precisions" : [(key.name, precision) for key, precision in self.exit_precision.items()],
            "exit_cost" : [(key.name, int(cost)) for key, cost in self.exit_costs.items()],
            "subgraph_costs" : [(key.name, int(cost)) for key, cost in self.subgraph_costs.items()],
        }

        with open(folder_path / f"report_early_exit_{self.access_id()}.json", "w") as file:
            json.dump(summary, file)

        return summary
    
    @classmethod
    def create_unique_id(cls, search_config : str) -> str:
        """
        Function to create a unique ID for instances of the EEReport class.
        This can be used to create a unqiue ID for instances that have not yet been instantiated

        Args:
            search_config (str): the name of the ModelAnalysis

        Returns:
            str: a unique ID
        """
        descr_str = f"EEReport-{search_config}"

        hashed_str = cls.create_reporter_id(descr_str)

        return hashed_str
    
    @classmethod
    def load(cls, folder_path : Union[str, pathlib.Path], analysis : aNN.ModelAnalysis) -> Union[Self, Set[Self]]:
        #TODO: implement loading
        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)
        
        #find json and pickle file
        json_data = {}
        for option in folder_path.glob("report_early_exit_*.json"):
            with open(option, "rb") as file:
                json_data[option.stem] = json.load(file)

        reports = set()
        for name, data in json_data.items():
            new_report = EarlyExitReport(analysis=analysis, exit_search_config=data["search_config"])
            new_report.is_ee = data["is_early_exit"]
            new_report.is_recommended = data["is_recommended"]

            new_id = new_report.access_id()
            if "unique_id" in data.keys():
                new_id = data["unique_id"]
            analysis.reports[new_id] = new_report
            reports.add(new_report)

        return reports
    
    @classmethod
    def submit_to(cls, analysis:aNN.ModelAnalysis, lazy:bool=False) -> EarlyExitReportSubmitter:
        submitter = EarlyExitReportSubmitter(analysis=analysis, lazy=lazy)
        return submitter

    @classmethod
    def closure(cls, create_id:bool=True, search_config:str = "large"):
        """Closure that should be passed to the ModelAnalysis object"""

        def builder(analysis: aNN.ModelAnalysis):
            return EarlyExitReport(analysis=analysis, exit_search_config=search_config)
        
        if create_id:
            descr_str = f"EEReport_{search_config}"
            hashed_str = cls.create_reporter_id(descr_str)

            return builder, hashed_str

        return builder
