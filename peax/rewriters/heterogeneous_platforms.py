from ctypes import ArgumentError
import json
import pathlib
from typing import Iterable, List, Dict, Optional, Union, Tuple, Set
import logging as log
import numbers
import copy
import math
import hashlib
import os

import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras import backend as K
import networkx as nx
from jinja2 import Template

from .base import Rewriter, Solution

from ..hardware import processor as pr
from ..hardware import connection as con
from ..hardware import power_domain as pd

from ..reports import early_exit as ee
from ..reports import hw_checker as hw_check
from ..reports import accuracy as acc
from ..reports import dataset as data

from ..components import resource as res
from ..components import graph_tools as gt
from ..components import predictive as prd

from ..utils import keras_graph

import peax.analysis as aNN

class MappingOption:
    """A class to represent a possible mapping option"""

    options : List[gt.BlockNode]
    """ locations in the BlockGraph where a branch could be placed """

    processors : List[pr.Processor]
    """ the processors that the subgraphs can ge mapped to """

    mappings: Dict[pr.Processor, gt.BlockNode]
    """ Map that describes which option was mapped to which processor, the processor is the key """

    cost_map: Dict[Tuple[pr.Processor, gt.BlockNode], Dict]
    """ map that assigns the cost of a mapping of a subgraph to a processor with a specific cost """

    cost_keys : Set[str]
    """ names of the metrics that have been evaluated """

    def __init__(
        self, options: List[gt.BlockNode], processors: List[pr.Processor]
    ) -> None:
        self.options: List[gt.BlockNode] = options
        self.processors: List[pr.Processor] = processors

        self.mappings: Dict[pr.Processor, gt.BlockNode] = dict()
        self.cost_map: Dict[Tuple[pr.Processor, gt.BlockNode], Dict] = dict()
        self.cost_keys = set()
        
        pass

    def add_mapping(
        self, processor: pr.Processor, node: gt.BlockNode, costs
    ) -> bool:
        """Adds a mapping of a given exit to a given processor.
        This means that the subgraph since the last inserted exit (or the input, if no previous exit has been mapped yet)
        will be executed by the processor as well as the assigned early exit branch

        Args:
            processor (pr.Processor): The processor that will execute the early exit and the backbone up to the branching point
            node (gt.BlockNode): the branching point after which the exit will be inserted
            costs (_type_): estimated costs of the mapping during the inference

        Returns:
            bool: True, if mapping has been applied successfully, False otherwise
        """
        if node in self.mappings.values():
            log.warning(f"{node} has already been mapped")
            return False

        if node not in self.options:
            log.warning(f"{node} is not known as a possible solutions")
            return False

        if processor not in self.processors:
            log.warning(f"{processor} is not known as a possible solutions")
            return False

        self.mappings[processor] = node
        self.cost_map[(processor, node)] = costs
        for key in costs:
            self.cost_keys.add(key)

        return True

    def get_mappings(self) -> List[Tuple[pr.Processor, gt.BlockNode]]:
        """Returns all currently configured mappings for this MappingOption

        Returns:
            List[Tuple[pr.Processor, gt.BlockNode]]: All stored mappings as List of Tuples
        """
        mappings = list()
        for key, value in self.mappings.items():
            mappings.append((key, value, self.cost_map[(key, value)]))

        return mappings

    def get_mapping(self, processor: pr.Processor) -> gt.BlockNode:
        """Returns the mapped output for a given processor, returns NOne, if no exit has been assigned to the processor yet

        Args:
            processor (pr.Processor): The processor, whose exit we want to access

        Returns:
            gt.BlockNode: The exit that has been assigned to this processor
        """
        if not processor in self.mappings.keys():
            return None

        node = self.mappings[processor]

        return node

    def get_assigned_processors(self) -> List[pr.Processor]:
        """Returns all processors that already have been assigned to exit locations

        Returns:
            List[pr.Processor]: List of assigned processors
        """
        return list(self.mappings.keys())

    def get_unassigned_processors(self) -> List[pr.Processor]:
        """Returns all processors that have not been assigned to an exit location yet,
        they can be used for the open locations or stay unused for this specific MappingOption

        Returns:
            List[pr.Processor]: List of unassigned processors
        """
        return [
            x
            for x in self.processors
            if x not in self.get_assigned_processors()
        ]

    def get_assigned_nodes(self) -> List[gt.BlockNode]:
        """Returns all Exit options that have not yet been assigned.
        Not all Exits need to be assigned for a valid solution

        Returns:
            List[gt.BlockNode]: List of assigned exit options
        """
        return list(self.mappings.values())

    def get_unassigned_nodes(self) -> List[gt.BlockNode]:
        """Return all Exit options that have not been assigned to a processor yet.
        Not all Exits need to be assigned for a valid solution

        Returns:
            List[gt.BlockNode]: List of unassigned exit options
        """
        return [x for x in self.options if x not in self.get_assigned_nodes()]

    def get_open_nodes(self) -> List[gt.BlockNode]:
        """Returns a list of all exits that have not yet been assigned, and are located after the last assigned node.
        This function is used to keep the processors in the required order during the generation of the Mapping option.
        PROBLEM: The open nodes are not in the right order in Python versions < 3.10 (which we have to use for the IFX HPC)

        Returns:
            List[gt.BlockNode]: List of the open exit options
        """
        if len(self.get_assigned_nodes()) == 0:
            return self.get_unassigned_nodes()

        last_assigned = self.get_assigned_nodes()[-1]

        idx = self.options.index(last_assigned)
        if (idx + 1) == len(self.options):
            return []
        return self.options[(idx + 1) :]

    def accumulate_cost(self, cost_key: str):
        """Accumulates the total cost (for the given cost key) for the MappingOption

        Args:
            cost_key (str): which cost metric to use

        Returns:
            _type_: The accumulated cost metric across all layers and mappings
        """
        if len(self.cost_map) == 0:
            return None

        if not cost_key in self.cost_keys:
            log.warning(
                f"unknown cost key, cost_key must be one of {self.cost_keys}"
            )
            return None

        if isinstance(list(self.cost_map.values())[0][cost_key], bool):
            value = True
            for val in self.cost_map.values():
                value &= val[cost_key]

            return value

        value = 0
        if isinstance(list(self.cost_map.values())[0][cost_key], dict):
            value = {}
            for key in list(self.cost_map.values())[0][cost_key].keys():
                value[key] = 0
        for val in self.cost_map.values():
            if isinstance(val[cost_key], dict):
                for key in val[cost_key].keys():
                    value[key] += val[cost_key][key]
            else:
                value += val[cost_key]

        return value

    def __str__(self) -> str:
        """string representation of the object

        Returns:
            str: the string
        """
        string = f"Solution {[tuple(map(str, entry)) for entry in self.mappings.items()]}"
        return string

    def __eq__(self, __o: object) -> bool:
        """Equality checker

        Args:
            __o (object): Object that the MappingOption will be compared to

        Returns:
            bool: True, if equal, False if not
        """
        if not type(__o) == MappingOption:
            return False

        if self.processors == __o.processors and self.options == __o.options:
            if self.get_assigned_nodes() == __o.get_assigned_nodes():
                if (
                    self.get_assigned_processors()
                    == __o.get_assigned_processors()
                ):
                    if (
                        self.mappings == __o.mappings
                        and self.cost_map == __o.cost_map
                    ):
                        return True

        return False
    
    def __hash__(self) -> str:
        """
        Ensures that hashing the MappingOption creates the same hash, if the same parameters were applied to create it.

        Returns:
            str: the hash value
        """
        options_tuple = tuple([x.name for x in self.options])
        processors_tuple = tuple([x.name for x in self.processors])

        # Convert self.mappings to a hashable representation
        mappings_tuple = tuple([(key.name, val.name) for key, val in self.mappings.items()])

        # Convert self.cost_map to a hashable representation
        # TODO FIX THIS
        cost_map_tuple = None # tuple((processor.name, node.name, tuple(list(cost.values()))) for (processor, node), cost in self.cost_map.items())

        cost_keys_tuple = tuple(self.cost_keys)

        hash_value = hash((options_tuple, processors_tuple, mappings_tuple, cost_map_tuple, cost_keys_tuple))

        return hash_value

    
    def toDict(self) -> Dict[str, object]:
        """
        Turns the MappingOption into a dict.

        Returns:
            Dict[str, object]: a dict containing the mapping and cost_maps for the option
        """
        data = {}
        #data = self.__dict__
        #data["options"] = [block.name for block in self.options]
        #data["processors"] = [proc.name for proc in self.processors]
        data["mappings"] = [(proc.name, block.name) for proc, block in self.mappings.items()]
        data["cost_maps"] = [((proc.name, block.name), cost) for (proc, block), cost in self.cost_map.items()]
        return data


    def create_copy(self):
        """Creates a copy of the MappingOption to account for multiple possible solutions that extend the current solution

        Returns:
            MappingOption: A copy of the current object
        """
        new_options = self.options[:]
        new_processors = self.processors[:]

        new_instance = MappingOption(
            options=new_options, processors=new_processors
        )

        for processor, node in self.mappings.items():
            new_instance.add_mapping(
                processor=processor,
                node=node,
                costs=self.cost_map[(processor, node)],
            )
        # new_instance.mappings = copy.deepcopy(new_instance.mappings)

        return new_instance

class ThresholdConfigSpace:
    """
    A class to wrap the exit-wise threshold configuration into an object.
    One ThresholdConfigSpace object per EENN option is enough to represent the task.
    """

    option : MappingOption
    """The mapping option, that describes which exits are used and assigned to which classifier"""

    accuracies : Dict[gt.BlockNode, float]
    """The accuracies achieved by the early classifiers"""

    termination_rates : Dict[gt.BlockNode, float]
    """The termination rates achieved by the mapping option"""
    
    weights : List[float]
    """Weighting between the penalties of accuracy drop and cost increase"""

    MACs : Dict[gt.BlockNode, int]
    """The required MAC ops for each early exit classifier"""

    def __init__(self, option : MappingOption, termination_rates : Dict[gt.BlockNode, float], MACs : Dict[gt.BlockNode, int], weights : List[float] = None) -> None:
        self.option = option
        #self.accuracies = accuracies
        self.termination_rates = termination_rates
        self.MACs = MACs

        if weights == None:
            weights = [1/2, 1/2]

        self.weights = weights

        self.search_graph, self.start_node, self.end_node = self.build_search_graph()
        #self.selection = self.search_for_solution()
        #self.best_config = self.convert_to_config(self.selection)
        pass

    def search_for_solution(self, weights : List[float] = None) -> List[Tuple[gt.BlockNode, float]]:
        """Finds the shortest path through the search space, which should equal the best solution
        in terms of threshold configurations for each Early Exit branch

        Args:
            weights (List[float], optional): weights of the different cost factors. Defaults to None.

        Returns:
            List[Tuple[gt.BlockNode, float]]: A List containing tuples of the BlockNode that describes the exit location and a threshold value of the exits
        """
        if weights is None:
            weights = self.weights
        np_weights = np.array(weights)

        def weight_function(u, v, data):
            costs = 0
            compute_cost = data["MAC"] * data["termination_rate"] # / max(self.MACs.values())
            #termination_rate = 1 - data["termination_rate"]
            accuracy = 1 - data["accuracy"] # * data["termination_rate"]

            costs = np.array([compute_cost, accuracy])
            
            costs = np_weights * costs
            costs = np.sum(costs)
            return costs
        try:
            shortest_path = nx.bellman_ford_path(self.search_graph, self.start_node, self.end_node, weight=weight_function)
            cost = nx.bellman_ford_path_length(self.search_graph, self.start_node, self.end_node, weight=weight_function)
        except Exception as e:
            log.warn(f"something went wrong: {e}")
            #search_graph = self.build_search_graph()
            return []
        return shortest_path[1::], cost
    
    def convert_to_config(self, selection : List[Tuple[gt.BlockNode, float]]) -> List[float]:
        """Converts the found result into a list of the thresholds, in the same order as the Exit branches they correspond to

        Args:
            selection (List[Tuple[gt.BlockNode, float]]): the previously found path through the search space

        Returns:
            List[float]: the thresholds
        """
        result = list()
        for node, threshold in selection:
            result.append(threshold)
        return tuple(result)

    def build_search_graph(self) -> nx.DiGraph:
        """Creates the Search Graph for the embedded mappingOption.
        The graph looks similar to a dense neural network, the "layers" equal the Early Exit positions,
        while the "neurons" represent the different termination thresholds for the confidence score.
        The edges are annotated with dicts that contain the difference in cost metrics between the exit threshold configurations

        Returns:
            nx.DiGraph: the search graph
        """

        # create search graph
        graph = nx.DiGraph()

        # add nodes

        ## exit branches
        for exit_branch, configuration in self.termination_rates.items():
            ## threshold configurations
            for thres_config in configuration.keys():
                if isinstance(thres_config, float):

                    graph.add_node((exit_branch, thres_config))

        ## add start and end node (end node will be the final exit of the backbone, so maybe just the starting node)
        graph.add_node("start")

        # add edges with cost annotation

        ## from start to first exit branch
        first_exit = self.option.get_assigned_nodes()[0]
        for thres_config, costs in self.termination_rates[first_exit].items():
            if isinstance(thres_config, float):
                #print(thres_config)
                # TODO: fix this mess
                costs["MAC"] = self.MACs[first_exit]
                #costs["time"] = [(cpu, node), costs in self.option.cost_map] #self.option.cost_map
                graph.add_edge("start", (first_exit, thres_config), **costs)

        ## add other edges
        for pos, node in enumerate(self.option.get_assigned_nodes()):
            # find next node
            if pos+1 >= len(self.option.get_assigned_nodes()):
                continue
            next_node = self.option.get_assigned_nodes()[pos+1]
            
            log.debug(f"drawing edges from {node} to {next_node}")
            for node_thres, node_costs in self.termination_rates[node].items():
                if not isinstance(node_thres, float):
                    continue
                node_costs["MAC"] = self.MACs[node]
                for next_thres, next_costs in self.termination_rates[next_node].items():
                    if not isinstance(next_thres, float):
                        continue
                    next_costs["MAC"] = self.MACs[next_node]
                    delta = {}
                    for key, val in node_costs.items():
                        delta[key] = node_costs[key] - next_costs[key]
                    log.debug(delta)
                    graph.add_edge((node, node_thres), (next_node, next_thres), **delta)

        end_node = self.option.get_assigned_nodes()[-1]
        end_thres = list(self.termination_rates[end_node].keys())[-1]
        
        return graph, "start", (end_node, end_thres)

class HeterogeneousPlatformSolution(Solution):
    """
    A class to wrap the solution generated by the HeterogenousPlatformOptimizer.
    It is supposed to wrap the extracted information and provide certain extended functionality, like finetuning and evaluation
    """

    mapping : MappingOption
    """the mapping of the subgraphs to the devices"""

    model : tf.keras.Model
    """the full early exit model, containing all classifiers that have been selected for this solution"""

    thresholds : Tuple[float]
    """ the exit-wise confidence thresholds """

    rewriter : Rewriter
    """ the rewriter that created the solution """

    costs : Dict[str, object]
    """ the cost of its inference in early exit model """

    cost_weights : Tuple[float]
    """ the weighting of the penalties """

    def __init__(self, mapping:MappingOption, thresholds:Tuple[float], rewriter:Rewriter, costs:Dict[str, object], cost_weights:Tuple[float]) -> None:
        self.rewriter = rewriter
        self.mapping = mapping
        self.model = rewriter.compile_mapping(mapping)

        self.thresholds = thresholds
        self.costs = costs
        self.cost_weights = cost_weights

        self.macs = {}
        ref_macs = res.get_subgraph_macs(self.rewriter.analysis.architecture.block_graph)
        for exit_block in self.mapping.get_assigned_nodes():
            added_cost = 0
            if exit_block in self.rewriter.EE_MAC.keys():
                added_cost = self.rewriter.EE_MAC[exit_block]
            self.macs[exit_block] = (res.get_subgraph_macs(
                self.rewriter.analysis.architecture.block_graph,
                start_block=None,
                end_block=exit_block) + added_cost) / ref_macs
        pass

    def finetune(self, epochs:int=1, train_config:Dict[str, object]=None) -> None:
        """Finetunes the solution by training the full model for the given epochs and rerunning the search for the confidence thresholds afterwards.
        
        Args:
            epochs (int, optional): the number of epochs for the training of the EENN. Defaults to 1.
            train_config (Dict[str, object], optional): an optional dict that contains parameters for the model.fit process. """
        self.finetune_model(epochs=epochs, train_config=train_config)
        self.finetune_decisions()

        return

    def finetune_model(self, epochs:int=1, train_config : Dict[str, object] = None, enable_eval:bool=True) ->Optional[Dict[str, float]]:
        """
        Finetunes the weights of the model for the given epochs.

        Args:
            epochs (int, optional): the number of epochs for the training of the EENN. Defaults to 1.
            train_config (Dict[str, object], optional): an optional dict that contains parameters for the model.fit process.. Defaults to None.
            enable_eval (bool, optional): Evaluates the model after finetuning. Defaults to True.

        Returns:
            Optional[Dict[str, float]]: returns evaluation results if enable_eval = True
        """

        '''acc_rep_id = self.rewriter._submitted_reports[acc.AccuracyReport]
        acc_report : acc.AccuracyReport = self.rewriter.analysis.access_report(acc_rep_id)'''
        acc_report : acc.AccuracyReport = self.rewriter.acc_report
        batch_size = acc_report.batch_size

        '''ee_rep_id = self.rewriter._submitted_reports[ee.EarlyExitReport]
        ee_report : ee.EarlyExitReport = self.rewriter.analysis.access_report(ee_rep_id)'''
        ee_report : ee.EarlyExitReport = self.rewriter.ee_report

        # acquire training set
        train_dataset = self.rewriter.train_dataset.data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # acquire training config
        if train_config is None:
            train_config = ee_report._determine_ee_training_config(self.mapping, batch_size=batch_size)
            train_config["optimizer"].learning_rate = train_config["optimizer"].learning_rate / 20
        self.model.compile(**train_config)

        # train model
        self.model.fit(train_dataset, epochs=epochs)

        # evaluate on test set
        #test_dataset = self.optimizer.test_dataset.data.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = self.rewriter.test_dataset.data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        eval_data = None
        if enable_eval:
            eval_data = self.model.evaluate(test_dataset, return_dict=True)
        
        return eval_data

    def finetune_decisions(self) -> List[float]:
        """
        Finetunes the confidence decision thresholds.
        Mostly needed after tuning the weights of the EENN architecture.

        Returns:
            List[float]: the new threshold parameters.
        """
        threshold_ranges = [[] for _ in range(len(self.model.outputs))]

        for idx, threshold in enumerate(self.thresholds[:-1]):
            threshold_ranges[idx] = (0.6, 0.99, 0.01)

        threshold_ranges.append((0.00, 0.01, 0.01))

        termination_rates = self.evaluate_thresholds(threshold_ranges=threshold_ranges)

        thres_config = ThresholdConfigSpace(self.mapping, termination_rates=termination_rates, MACs=self.macs, weights=self.cost_weights)
        
        thresholds_config_route, cost = thres_config.search_for_solution()
        thresholds = thres_config.convert_to_config(thresholds_config_route)

        self.thresholds = thresholds
        
        return thresholds

    def evaluate_thresholds(self, threshold_ranges:Union[List[Tuple[float, float, float]],Tuple[float, float, float]] = (0.6,1.0, 0.05)):
        """Helper function to search the ideal thresholds.
        It creates the evaluation data for the used classifiers in combination with different threshold configurations."""
        
        #batch_size = self.rewriter.analysis.access_report(self.rewriter._submitted_reports[acc.AccuracyReport]).batch_size
        batch_size = self.rewriter.acc_report.batch_size
        if self.rewriter.test_dataset.data.element_spec[0].shape[0] is None:
            test_data = self.rewriter.test_dataset.data.unbatch()
        else:
            test_data = self.rewriter.test_dataset.data
        test_data = test_data.batch(batch_size=batch_size)

        num_outputs = len(self.model.outputs)
        y_pred = [[] for _ in range(num_outputs)]
        y_labels = [[] for _ in range(num_outputs)]
        true_labels = []
        for x, y in test_data:
            batch_predictions = self.model.predict(x, verbose=0)
            for out_idx in range(num_outputs):
                y_pred[out_idx].append(batch_predictions[out_idx])
                batch_labels = np.argmax(batch_predictions[out_idx], axis=-1)
                y_labels[out_idx].append(batch_labels)

            batch_true_labels = np.argmax(y, axis=-1)
            true_labels.append(batch_true_labels)

        y_pred = [np.concatenate(preds, axis=0) for preds in y_pred]
        y_labels = [np.concatenate(labels, axis=0) for labels in y_labels]
        y_conf = [np.amax(preds, axis=-1) for preds in y_pred]
        true_labels = np.concatenate(true_labels, axis=0)
        total_samples = len(true_labels)

        if isinstance(threshold_ranges, tuple):
            threshold_ranges = [threshold_ranges] * num_outputs

        result = {}
        for out_idx in range(len(y_labels)):
            thres_range = threshold_ranges[out_idx]
            
            if out_idx == len(y_labels) - 1:
                threshold_range = [0.0]
            else:
                threshold_range = list(np.arange(*thres_range))

            out_result = {}
            #result["raw"] = y_pred
            for threshold in threshold_range:
                # TODO: adapt to non-classification tasks

                # get accuracy and termination rate for each threshold
                correct_predictions = 0
                terminated_samples = 0

                for i in range(total_samples):
                    if y_conf[out_idx][i] >= threshold:
                        # Predicted label for the sample exceeds the confidence threshold
                        if y_labels[out_idx][i] == true_labels[i]:
                            correct_predictions += 1
                        terminated_samples += 1                    

                if terminated_samples > 0:
                    out_result[threshold] = {}
                    out_result[threshold]["accuracy"] = correct_predictions / terminated_samples
                    out_result[threshold]["termination_rate"] = terminated_samples / total_samples
                else:
                    out_result[threshold] = {}
                    out_result[threshold]["accuracy"] = 0
                    out_result[threshold]["termination_rate"] = 0
            
            result[self.mapping.get_assigned_nodes()[out_idx]] = out_result

        return result
    
    def evaluate(self, data_report : data.DatasetReport = None) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Evaluates the performance of the found solution.
        Returns information about the accuracy, the termination locations and the labels created by the classifiers,
        even if they would not be executed due to early termination.

        Returns:
            List[Dict, Dict, Dict, Dict]: the resulting data
        """
        try:
            #batch_size = self.rewriter.analysis.access_report(self.rewriter._submitted_reports[acc.AccuracyReport]).batch_size
            batch_size = self.rewriter.acc_report.batch_size
        except:
            batch_size = 64
            log.warn(f"no batch size information could be found for the current device, falling back to {batch_size}")
        #test_data = self.rewriter.test_dataset.data.unbatch().batch(batch_size=batch_size)

        if data_report is None:
            test_data = self.rewriter.test_dataset.data.batch(batch_size=batch_size)
        else:
            test_data = data_report.data.batch(batch_size=batch_size)

        num_outputs = len(self.model.outputs)
        y_pred = [[] for _ in range(num_outputs)]
        y_labels = [[] for _ in range(num_outputs)]
        true_labels = []
        y_true = []
        for x, y in test_data:
            batch_predictions = self.model.predict(x, verbose=0)
            for out_idx in range(num_outputs):
                if num_outputs > 1:
                    y_pred[out_idx].append(batch_predictions[out_idx])
                    if batch_predictions[out_idx].shape[-1] > 1:
                        batch_labels = np.argmax(batch_predictions[out_idx], axis=-1)
                    else:
                        batch_labels = np.where(batch_predictions[out_idx] > 0.5, 1, 0).flatten()
                    y_labels[out_idx].append(batch_labels)
                else:
                    y_pred[out_idx].append(batch_predictions)
                    batch_labels = np.argmax(batch_predictions, axis=-1)
                    y_labels[out_idx].append(batch_labels)

            if len(y.shape) > 1: 
                batch_true_labels = np.argmax(y, axis=-1)
            else:
                batch_true_labels = y
            true_labels.append(batch_true_labels)
            y_true.append(y.numpy())

        y_pred = np.array([np.concatenate(preds, axis=0) for preds in y_pred])
        y_labels = np.array([np.concatenate(labels, axis=0) for labels in y_labels])
        y_conf = np.array([np.amax(preds, axis=-1) for preds in y_pred])
        true_labels = np.array(np.concatenate(true_labels, axis=0))
        y_true = np.array(np.concatenate(y_true, axis=0))
        total_samples = len(true_labels)

        y_eenn_labels = []
        y_eenn_pred = []
        terminated_at = []
        right = 0
        for idx, outputs in enumerate(y_conf.transpose()):
            for pos, conf in enumerate(outputs):
                if conf >= self.thresholds[pos]:
                    y_eenn_labels.append(y_labels[pos, idx])
                    y_eenn_pred.append(y_pred[pos, idx])
                    terminated_at.append(pos)
                    right += int(y_labels[pos, idx] == true_labels[idx])
                    break

        y_eenn_labels = np.array(y_eenn_labels)
        y_eenn_pred = np.array(y_eenn_pred)

        #acc_rep_id = self.rewriter._submitted_reports[acc.AccuracyReport]
        #acc_report : acc.AccuracyReport = self.rewriter.analysis.access_report(acc_rep_id)
        acc_report : acc.AccuracyReport = self.rewriter.acc_report
        metrics = acc_report.metrics
        other_results = {}

        #there is something wrong in here
        for name, func in metrics.items():
            try:
                other_results[name] = func(y_true, y_eenn_pred)
            except:
                log.warning(f"was unable to calculate {name} for EENN evaluation")

        return right/total_samples, terminated_at, y_eenn_labels, other_results
    
    def split(self) -> List[tf.keras.Model]:
        """
        Returns a list of the subgraphs as keras models that are assigned to each processor of the platform.
        The weights are copied from the original model and do not change with it.

        Returns:
            List[tf.keras.Model]: the keras submodels
        """

        eenn_model = tf.keras.models.clone_model(self.model)
        eenn_model.build(self.model.input_shape)
        eenn_model.set_weights(self.model.get_weights())

        nodes = [] #self.mapping.get_assigned_nodes()[0:-1] # ignore the final exit here
        proc_names = []
        for proc, node, stats in self.mapping.get_mappings():
            nodes.append(node)
            proc_names.append(proc.name)

        nodes = nodes[0:-1]

        connection_layer_names = []
        for node in nodes:
            connection_layer_names.append(gt.get_output_nodes(node.subgraph)[0].keras.name)

        connection_layers = []
        for name in connection_layer_names:
            connection_layers.append(eenn_model.get_layer(name))
            #connection_layers.append(eenn_model.get_layer(name).output)
            
        exit_names = eenn_model.output_names
        final_exit_name = exit_names[-1] # moving the final classifier to the end of the list, for correct ordering
        exit_names = exit_names[:-1]
        exit_names.append(final_exit_name)

        exit_layers = []
        for name in exit_names:
            exit_layers.append(eenn_model.get_layer(name).output)

        inp_layers = eenn_model.inputs # [eenn_model.get_layer(inp.name) for inp in eenn_model.inputs]
        subgraph_models = []
        for idx, (connect, out_classifier) in enumerate(zip(connection_layers, exit_layers)):
            #log.debug([connect.output, out_classifier])
            out_layers = [connect.output, out_classifier]
            sub_model = tf.keras.Model(inputs=inp_layers, outputs=out_layers, name=f"{eenn_model.name}-{proc_names[idx]}")

            new_submodel = tf.keras.models.clone_model(sub_model)
            new_submodel.build(sub_model.input_shape)
            new_submodel.set_weights(sub_model.get_weights())

            log.debug(f"new submodel was created {new_submodel.name}")
            for layer in new_submodel.layers:
                log.debug("\t", layer.name)
            
            subgraph_models.append(new_submodel)

            connection_layer = eenn_model.get_layer(connection_layer_names[idx])
            inp_layers = connection_layer.output #[tf.keras.layers.Input(shape=connection_layer.output_shape, tensor=connection_layer.output)]

        # need to handle final exit subgraph
        sub_model = tf.keras.Model(inputs=inp_layers, outputs=exit_layers[-1], name=f"{eenn_model.name}-{proc_names[-1]}")
        new_submodel = tf.keras.models.clone_model(sub_model)
        new_submodel.build(sub_model.input_shape)
        new_submodel.set_weights(sub_model.get_weights())
        subgraph_models.append(new_submodel)

        log.debug(f"new submodel was created {new_submodel.name}")
        for layer in new_submodel.layers:
            log.debug("\t", layer.name)

        return subgraph_models
    
    def dump(self, path : Union[str, pathlib.Path]):
        """Writes the solution to disk.
        Currently not maintaining any naming scheme to enable writing multiple solutions to the same folder."""

        thres_filename = f"hp_solution.json"
        sub_model_paths = {}

        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        data = self.toDict(include_objects=False)
        with open(path / thres_filename, "w") as file:
            json.dump(data, file)

        self.model.save(path / f"hp_solution_full_{self.model.name}.tf")
        self.model.save(path / f"hp_solution_full_{self.model.name}.h5")
        self.model.save(path / f"hp_solution_full_{self.model.name}.keras")

        sub_model_paths[self.model.name] = f"{self.model.name}.h5"

        split_path = path / f"hp_submodels_{self.model.name}"
        split_path.mkdir(exist_ok=True, parents=True)

        submodels = self.split()
        for sub in submodels:
            sub.save(split_path / f"{sub.name}.tf")
            sub.save(split_path / f"{sub.name}.h5")
            sub.save(split_path / f"{sub.name}.keras")

            sub_model_paths[sub.name] = str(split_path / f"{sub.name}.h5" )
        data["models"] = sub_model_paths

        data["threshold_config_file_path"] = thres_filename
        data["relative_path"] = str(path.parts[-1])

        return data
    
    def to_analysis(self) -> List[object]:
        """
        Converts the solution to a list of new ModelAnalysis instances.
        One instance is created per subgraph.
        Keep in mind that the application of different optimizations to the submodels could prevent them from working together as
        an early exit neural network.

        Returns:
            List[object]: The created ModelAnalysis obejcts.
        """
        submodels = self.split()

        cache_dir = self.rewriter.analysis.cache_dir

        # we need to create one ModelAnalysis for each submodel
        a_list = []
        for sub in submodels:
            print(sub.name)
            new_analysis = aNN.ModelAnalysis(model=sub, name=sub.name, cache_dir=cache_dir)

            a_list.append(new_analysis)

        return a_list

    def toDict(self, include_objects:bool=True) -> Dict[str, object]:
        """
        Converts the solution into a dict.

        Args:
            include_objects (bool, optional): If objects that cannot be serialized by JSON should be included. Defaults to True.

        Returns:
            Dict[str, object]: A dict representation of the solution.
        """
        values = {
            "cost_weights" : self.cost_weights,
            "thresholds" : self.thresholds,
            "mapping" : self.mapping.toDict(),
        }

        if include_objects:
            values["eenn_model"] = self.model
            values["submodels"] = self.split()

        return values
class HeterogeneousPlatformRewriter(Rewriter):
    """Rewriter that uses information from the Early Exit report and the HWChecker report to decide how to split the neural network across multiple processors
    and insert the EE branches to achieve the largest efficiency gains without impacting the accuracy and latency requirements of the application

    Args:
        analysis (aNN.ModelAnalysis): The analysis of the model that should be optimized
        latency (int, optional): the latency requirements in ms. Defaults to 500.
        processors (List[pr.Processor], optional): List of processors that should be used. Defaults to None.
        connections (List[con.Connection], optional): List of connections between the processors. Defaults to None.
        dtypes (Set[str], optional): Datatypes that could be used by different versions of the model. Defaults to ["int8", "float32"].

    Raises:
        ValueError: if no processors have been passed to the model
        ValueError: if no connections have been passed to the model
    """

    analysis: aNN.ModelAnalysis
    """ the full model analysis that is the base for this optimization run """
    _submitted_reports : Dict
    """ mapping from Reporter class to ID of report of that type that was submitted to analysis object"""
    processors: Set[pr.Processor]
    """The processors that can be targeted by this optimization"""
    connections: Set[con.Connection]
    """connections between the processors that can be used to transfer intermediate results between the graph partition executions"""
    dtypes: Set[str]
    """The datatype sizes that will be considered during the evaluation"""
    latency: float
    """The acceptable worst case latency in seconds, will be used to restrict the solution search space to points that fullfill the
    time requirements for the worst execution configuration of running all EEs as well as the final Exit"""
    EE_keras: Dict[gt.BlockNode, tf.keras.Model]
    """the generated Keras models for the Early Exits addressed by their intended locations"""
    EE_MAC: Dict[gt.BlockNode, int]
    """the cost of executing the EE of the given location in MAC operations"""
    # EE_bases: Dict[gt.BlockNode, nx.DiGraph]
    # """the section of the backbone up until the exit location"""
    EE_costs: Dict[gt.BlockNode, Dict]
    """cost map that contains information about the required resources for the execution of each early exit on each processor"""

    parts_costs: Dict
    """cost map that contains information about the required resources for the execution of each possible partition on each processor"""

    transmission_costs: Dict[
        gt.BlockNode, Dict[con.Connection, Dict[str, float]]
    ]
    """Dict that uses the Exit option as key, and contains the connections and dtypes as dicts to access the transmission time in seconds as float"""

    final : Dict[str, HeterogeneousPlatformSolution]
    """stores already found and returned solutions"""

    def __init__(
        self,
        analysis: aNN.ModelAnalysis,
        latency: int = 500,
        processors: Set[pr.Processor] = None,
        connections: Set[con.Connection] = None,
        dtypes: Set[str] = ["int8", "float32"],
        train_dataset : data.DatasetReport = None,
        validation_dataset : data.DatasetReport = None,
        test_dataset : data.DatasetReport = None,
        #trainset_path: Union[str, pathlib.Path] = None,
        #testset_path: Union[str, pathlib.Path] = None,
        #preprocessor: callable = None,
    ):
        """Initializes the Optimizer
        TODO: need to fix due to usage of old report submission syntax

        Args:
            analysis (aNN.ModelAnalysis): The analysis of the model that should be optimized
            latency (int, optional): the latency requirements in ms. Defaults to 500.
            processors (List[pr.Processor], optional): List of processors that should be used. Defaults to None.
            connections (List[con.Connection], optional): List of connections between the processors. Defaults to None.
            dtypes (Set[str], optional): Datatypes that could be used by different versions of the model. Defaults to ["int8", "float32"].
        Raises:
            ValueError: if no processors have been passed to the model
            ValueError: if no connections have been passed to the model
        """
        #self.analysis = analysis
        self._submitted_reports = dict()
        self.processors = processors

        self.dtypes = dtypes
        self.latency = latency / 1000

        if processors is None:
            raise ValueError("processors cannot be None")
        else:
            if len(processors) == 0:
                raise ValueError("processors cannot be empty")

        if connections is None:
            raise ValueError("connections cannot be None")
        else:
            if len(connections) == 0:
                raise ValueError("connections cannot be empty")
            
        if train_dataset is None:
            raise ValueError("training dataset cannot be None")
        else:
            self.train_dataset = train_dataset
            self.train_dataset.shuffle()
        
        if test_dataset is None:
            raise ValueError("test dataset cannot be None")
        else:
            self.test_dataset = test_dataset
            
        
        if validation_dataset is None:
            log.warn("validation dataset is None, will use training dataset for threshold calibrations instead")
            self.validation_dataset = train_dataset
        else:
            self.validation_dataset = validation_dataset
            self.validation_dataset.shuffle()

        self.connections = dict()
        for connection in connections:
            self.connections[
                connection.start_point, connection.end_point
            ] = connection

        super().__init__(analysis)
        # Get the hexadecimal representation of the hash

        # create HWReport
        self.hw_report : hw_check.HWReport = hw_check.HWReport.submit_to(analysis=analysis, lazy=False).with_config(processors=processors, dtypes=dtypes)
        '''hw_report, hw_report_id = hw_check.HWReport.closure(
            processors=processors,
            dtypes=dtypes,
            create_id=True)
        self.analysis.submit_reporter(
            unique_identifier=hw_report_id,
            reporter_constructor=hw_report            
        )
        self._submitted_reports[hw_check.HWReport] = hw_report_id'''
        self._submitted_reports[hw_check.HWReport] = self.hw_report#.access_id()

        # create EEReport
        self.ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        self._submitted_reports[ee.EarlyExitReport] = self.ee_report#.access_id()
        '''ee_report, ee_report_id = ee.EarlyExitReport.closure(create_id=True)
        self.analysis.submit_reporter(
            unique_identifier=ee_report_id,
            reporter_constructor=ee_report
        )
        self._submitted_reports[ee.EarlyExitReport] = ee_report_id'''

        # create the AccuracyReport
        datasets = {self.train_dataset, self.test_dataset}
        if self.train_dataset is not self.validation_dataset:
            datasets.add(self.validation_dataset)
        
        self.acc_report = acc.AccuracyReport.submit_to(analysis=analysis, lazy=False).with_config(datasets=datasets, metrics=None)
        self._submitted_reports[acc.AccuracyReport] = self.acc_report#.access_id()
        '''acc_report, acc_report_id = acc.AccuracyReport.closure(
            datasets=datasets,
            metrics=None,
            create_id=True
        )
        self.analysis.submit_reporter(
                unique_identifier=acc_report_id,
                reporter_constructor=acc_report
            )

        self._submitted_reports[acc.AccuracyReport] = acc_report_id'''

        #analysis.rewriters.add(self)

        self.analysis.create_reports()

        self.EE_keras = self._create_EE_models()
        self.EE_MAC = self._estimate_EE_MAC()

        self.EE_costs = self._prepare_EE_cost_map()
        self.parts_costs = self._prepare_partition_cost_map()

        self.transmission_costs = self._prepare_transmission_costs()

        # stores all options that have been rejected for not meeting constraints
        self.rejects : List[MappingOption] = []
        # self.option_space = self._create_solution_space()
        self.options = self.extract_all_options()

        self.final = {}

        #self = analysis.submit_rewriter(self)
        return
    

    def compile_mapping(self, solution: MappingOption) -> tf.keras.Model:
        """takes the selected solution from the search space and turns it into an Early Exit Neural Network (EEN).
        This network can than be used to train the additional output branches, if they have not been trained yet.

        Args:
            solution (MappingOption): the selected solution that should be used

        Returns:
            tf.keras.Model: The generated EEN version of the original model
        """
        
        K.clear_session()

        original_model = self.analysis.keras_model
        attach_layers = []
        branches = []

        for proc, mapping in solution.mappings.items():

            #skip original classifer
            if mapping == gt.get_first_output_node(
                self.analysis.architecture.block_graph
            ):
                continue

            attach_layers.append(gt.get_first_output_node(mapping.subgraph).keras.name)
            branches.append(self.EE_keras[mapping])

        log.warn("replaced implementation with shared utility function, change has not been tested yet")
        return keras_graph.attach_branches(original_model, attachment_layers=attach_layers, branches=branches, reorder=True)
    
    def finetune_solution(self,
                          solution : Union[MappingOption, tf.keras.Model],
                          thresholds : Iterable[float],
                          epochs: int = 1,
                          train_config : dict = None) -> Tuple[tf.keras.Model, Tuple[int ,...]]:
        """
        DEPRECATED!

        Args:
            solution (Union[MappingOption, tf.keras.Model]): _description_
            thresholds (Iterable[float]): _description_
            epochs (int, optional): _description_. Defaults to 1.
            train_config (dict, optional): _description_. Defaults to None.

        Returns:
            Tuple[tf.keras.Model, Tuple[int ,...]]: _description_
        """
        
        if isinstance(solution, MappingOption):
            # convert solution into keras model with multiple exits
            ee_model = self.compile_mapping(solution=solution)
        elif isinstance(solution, tf.keras.Model):
            ee_model = solution

        finetuned_ee_model = self.finetune_solution_model(solution=solution, epochs=epochs, train_config=train_config)

        return finetuned_ee_model

    def finetune_solution_model(self,
                          solution: Union[MappingOption, tf.keras.Model],
                          epochs: int = 1,
                          train_config : dict = None,
                          enable_eval:bool = False) -> Tuple[tf.keras.Model, Optional[List]]:
        """
        FInetunes the weights of a potential solutions.
        DEPRECATED!

        Args:
            solution (Union[MappingOption, tf.keras.Model]): The mapping that has been selected as solution
            epochs (int, optional): number of epochs for finetuning. Defaults to 1.
            train_config (dict, optional): parameters for compile step. Defaults to None.
            enable_eval (bool, optional): if True, the finetuned model will also be evaluated. Defaults to False.

        Returns:
            Tuple[tf.keras.Model, Optional[List]]: _description_
        """

        if isinstance(solution, MappingOption):
            solution = self.compile_mapping(solution=solution)

        '''acc_rep_id = self._submitted_reports[acc.AccuracyReport]
        acc_report : acc.AccuracyReport = self.analysis.access_report(acc_rep_id)'''
        acc_report = self.acc_report

        batch_size = acc_report.batch_size

        '''ee_rep_id = self._submitted_reports[ee.EarlyExitReport]
        ee_report : ee.EarlyExitReport = self.analysis.access_report(ee_rep_id)'''
        ee_report = self.ee_report

        # we need to identify suitable training configurations, preprocessing steps and the datasets
        
        if self.train_dataset.data.element_spec[0].shape[0] is None:
            train_dataset = self.train_dataset.data.unbatch()
        else:
            train_dataset = self.train_dataset.data
        train_dataset = train_dataset.shuffle(buffer_size=self.train_dataset.size*2).batch(batch_size)

        #TODO: training configurations
        if train_config is None:
            train_config = ee_report._determine_ee_training_config(solution, batch_size=batch_size)
            train_config["optimizer"].learning_rate = train_config["optimizer"].learning_rate / 20

        log.debug(train_config)

        solution.compile(**train_config)
        solution.fit(train_dataset, batch_size=batch_size, epochs=epochs)

        del train_dataset

        if enable_eval:
            test_dataset = self.test_dataset.data.unbatch().batch(batch_size)

            eval_data = solution.evaluate(test_dataset)
            return solution, eval_data
        return solution

    def finetune_solution_decision(self, solution : MappingOption, model :tf.keras.models.Model, decision_mechanism : List[float]) -> List[float]:
        """
        Finetunes the decision parameters, necessary after changing model weights.

        Args:
            solution (MappingOption): The mapping that has been selected as solution.
            model (tf.keras.models.Model): _description_
            decision_mechanism (List[float]): _description_

        Returns:
            List[float]: _description_
        """

        raise NotImplementedError
    
        #TODO: implementation
        search_space = []

        for outp in decision_mechanism:
            search_space.append(set([0.6, 0.7, 0.8, 0.9, 0.95, 0.99, outp *0.99, outp*1.01, outp*0.95]))

        # need to identify outputs, measure their cost and accuracy
        train_dataset = self.train_dataset.data
        validation_dataset = self.validation_dataset.data

        ## predict individual samples of training / validation set
        ## check termination_rates of outputs, adapt using same method as previous search

        ##thres_config = ThresholdConfigSpace(sol, accuracies=accuracies, termination_rates=exit_metrics, MACs=mac_costs, weights=weights)

        return None

    def select_best_option(
        self,
        focus: str = "cloud",
        power_domains: Set[pd.PowerDomain] = None,
        fine_tune : bool = True,
        weights : Tuple[float] = (0.5, 0.5)
    ) -> HeterogeneousPlatformSolution:
        """This function allows for an automatic selection of the most appropriate solution from the solution space

        Args:
            focus (str, optional): metric on which the selection should focus, either "cloud" for cheap cloud cost
            or "speed" for computing speed. Defaults to "cloud".
            power_domains (Set[pd.PowerDomain], optional): required for energy focus, otherwise ignored. Defaults to None.
            fine_tune (bool): if the found solution should be fine-tuned (additional epoch of training and adapting decision mechanism)

        Raises:
            NotImplementedError: focus on energy not implemented yet

        Returns:
            MappingOption: the best solution based on the selected focus
        """

        search_space = self.options
        '''ee_rep_id = self._submitted_reports[ee.EarlyExitReport]
        ee_report : ee.EarlyExitReport = self.analysis.access_report(ee_rep_id)'''
        ee_report : ee.EarlyExitReport = self.ee_report

        '''acc_rep_id = self._submitted_reports[acc.AccuracyReport]
        acc_report : acc.AccuracyReport = self.analysis.access_report(acc_rep_id)'''
        acc_report = self.acc_report
        batch_size = acc_report.batch_size

        _supported_focus = ["speed", "cloud", "mpsoc"]
        if focus not in _supported_focus:
            raise NotImplementedError(f"The search focus {focus} has not been implemented, currently supported are {_supported_focus}")

        if power_domains is not None and focus != "speed":
            representations = list()

            # for each point in solution space, calculate MACs per power domains
            for option in self.options:
                current_distribution = [0 for domain in power_domains]
                input_node = gt.get_first_input_node(
                    self.analysis.architecture.block_graph
                )
                for proc, node in option.mappings.items():
                    # calculate MACs
                    # print(proc, node)
                    cost = self.parts_costs[proc][input_node, node]["macs"]
                    for i, domain in enumerate(power_domains):
                        if proc in domain:
                            current_distribution[i] += cost
                representations.append(current_distribution)

        if focus == "mpsoc":

            ## if focus "mpsoc" - most complicated problem:
            ## need to assign enough ops to early stages to be able to terminate a reasonable share of samples
            ## but should not assign to many ops to keep the average inference cost as low as possible
            ## difficult to decide without trained networks

            ### evaluate the precision of the individual solutions

            accuracies = {}
            exit_metrics = {}
            costs = {}

            mac_costs = self.EE_MAC.copy()
            ref_macs = res.get_subgraph_macs(self.analysis.architecture.block_graph)
            
            for sol in search_space:
                viable_sol = True
                for exit in sol.get_assigned_nodes():
                    if exit in self.EE_keras.keys():
                        ## idea use pretrained backbone and only train + evaluate the EE branch starting from the first option
                        # create the training and test data generators or iterators
                        # some kind of caching could be done here or in the called function

                        if exit not in accuracies.keys():
                            #acc_rep_id = self._submitted_reports[acc.AccuracyReport]
                            accuracies[exit], exit_metrics[exit] = ee_report.evaluate_precision(
                                exit_configuration=(exit, self.EE_keras[exit]),
                                training_datasetReport=self.train_dataset,
                                test_datasetReport=self.validation_dataset,
                                batch_size=batch_size
                            )
                        mac_costs[exit] = res.get_subgraph_macs(self.analysis.architecture.block_graph, start_block=None, end_block=exit) + self.EE_MAC[exit]
                        mac_costs[exit] /= ref_macs
                    else:
                        # need to handle the final exit that came with the backbone
                        if exit not in accuracies.keys():
                            exit_metrics[exit] = {}

                            '''if self.validation_dataset.data.element_spec[0].shape[0] is None:
                                eval_validation_dataset = self.validation_dataset.data.unbatch()
                            else:
                                eval_validation_dataset = self.validation_dataset.data'''
                            #eval_validation_dataset = self.validation_dataset.data

                            acc_res = list(acc_report.results[self.validation_dataset].values())[0] #self.analysis.keras_model.evaluate(eval_validation_dataset.batch(batch_size))[-1]
                            exit_metrics[exit][0.0] = {
                                "accuracy" : acc_res,
                                "termination_rate" : 1.0,
                            }
                            accuracies[exit] = acc_res

                        mac_costs[exit] = res.get_subgraph_macs(self.analysis.architecture.block_graph, start_block=None, end_block=exit)
                        mac_costs[exit] /= ref_macs

                    if len(exit_metrics[exit]) == 0:
                        #skip this point in solution space as exit was unable to learn
                        viable_sol = False
                
                if not viable_sol:
                    # skip further evaluation that are not viable 
                    costs[sol, None] = float("inf")
                    continue

                #build threshold configuration search space
                #weights = [0.9, 0.1]
                thres_config = ThresholdConfigSpace(sol, termination_rates=exit_metrics, MACs=mac_costs, weights=weights)

                ## TODO: evaluate the accuracy for different confidence thresholds / decision mechanism configurations
                thresholds_config_route, cost = thres_config.search_for_solution()
                thresholds = thres_config.convert_to_config(thresholds_config_route)

                costs[(sol, thresholds)] = cost

            if len(costs) == 0:
                log.warn("no possible solution has been found, you might need to increase your latency constraint.")

            best_solution = min(costs, key=costs.get)
            best_mapping, best_thresholds = best_solution
            min_cost = costs[best_solution]

            solution_object = HeterogeneousPlatformSolution(
                mapping=best_mapping,
                thresholds=best_thresholds,
                rewriter=self,
                costs=mac_costs,
                cost_weights=weights)

            if fine_tune:
                solution_object.finetune_model(epochs=1)
                #solution_object.finetune_weights()
            
            # can we add more information here? user might be interested in estimated runtime, accuracy and distribution
            self.final[focus] = solution_object
            return solution_object

        elif focus == "cloud" and power_domains is not None:
            ## if focus "cloud" - find mapping that assigns as many operations as possible to earlier domains, to reduce the workload on cloud instances
            ## supposed to reduce billing for cloud services by doing as much as possible locally, while still finding the fastest solution

            np_rep = np.array(representations)
            min_cloud_cost = np.min(np_rep[:, -1])

            search_space = [
                self.options[idx]
                for idx, x in enumerate(representations)
                if x[-1] == min_cloud_cost
            ]

        if focus in ["speed", "cloud"]:
            ## if focus on speed - find solution that is the fastest

            fastest = math.inf
            selected_option = None
            # find fastest time
            for option in search_space:
                if option.accumulate_cost("delay") < fastest:
                    fastest = option.accumulate_cost("delay")
                    selected_option = option
            
        accuracies = {}
        exit_metrics = {}
        for exit in selected_option.get_assigned_nodes():
            if exit in self.EE_keras.keys():
                #acc_rep_id = self._submitted_reports[acc.AccuracyReport]
                
                accuracies[exit], exit_metrics[exit] = ee_report.evaluate_precision(
                    exit_configuration=(exit, self.EE_keras[exit]),
                    training_datasetReport=self.train_dataset,
                    test_datasetReport=self.validation_dataset,
                    batch_size=acc_report.batch_size
                )
            else:              
                exit_metrics[exit] = {}
                if self.validation_dataset.data.element_spec[0].shape[0] is None:
                    eval_validation_dataset = self.validation_dataset.data.unbatch()
                else:
                    eval_validation_dataset = self.validation_dataset.data

                acc_res = self.analysis.keras_model.evaluate(eval_validation_dataset.batch(batch_size))[-1]
                exit_metrics[exit][0.0] = {
                    "accuracy" : acc_res,
                    "termination_rate" : 1.0,
                }
                accuracies[exit] = acc_res

        solution_object = HeterogeneousPlatformSolution(
            mapping=selected_option,
            thresholds=None,
            optimizer=self, 
            costs=None,
            cost_weights=None)
        self.final[focus] = solution_object

        return solution_object

    def _create_EE_models(self) -> Dict[gt.BlockNode, tf.keras.Model]:
        """Creates Keras models for the Early Exit branches and stores them in a Dict with the intended
        location in the block graph representation as key

        Returns:
            Dict[gt.BlockNode, tf.keras.Model]: the EE keras models, addressable by their intended location
        """
        '''ee_rep_id = self._submitted_reports[ee.EarlyExitReport]
        ee_report = self.analysis.access_report(ee_rep_id)'''
        ee_report = self.ee_report
        ee_keras = dict()
        for pos, layers in ee_report.exit_configs.items():
            ee_keras[pos] = ee_report.to_keras((pos, layers))
            """inp = tf.keras.Input(shape=pos.output_shape)

            layer_class = type(layers[0])
            x = layer_class.from_config(layers[0].get_config())(inp)

            for layer in layers[1:-1]:
                layer_class = type(layer)
                config = layer.get_config()
                if isinstance(layer, tf.keras.layers.Dense):
                    units = config["units"] // 4
                    config["units"] = max(units, 16)

                if isinstance(layer, tf.keras.layers.Conv2D):
                    filters = config["filters"]
                    config["filters"] = max(filters, 8)
                x = layer_class.from_config(config)(x)

            # final output layer, not handled in loop, to avoid changing number of classes
            layer_class = type(layers[-1])
            x = layer_class.from_config(layers[-1].get_config())(x)
            model = tf.keras.Model(inputs=inp, outputs=x, name=pos.name)
            # model.summary()
            ee_keras[pos] = model"""

        return ee_keras

    def _estimate_EE_MAC(self) -> Dict[gt.BlockNode, int]:
        """calculates the necessary MAC operations to run the EE branches.
        Stored as Dict, uses BlockNode to which EEs would attach as keys

        Returns:
            Dict[gt.BlockNode, int]: Dict that uses the attaching location in BlockGraph as key and required MAC as values
        """
        ee_macs = dict()
        for pos, model in self.EE_keras.items():
            ee_macs[pos] = np.sum(
                list(
                    res.get_model_macs(
                        model=model,
                        estimation_functions=self.analysis.compute.mac_estimators,
                    ).values()
                )
            )

        return ee_macs

    def _prepare_subgraphs(
        self,
    ) -> Dict[gt.BlockNode, nx.DiGraph]:
        """Returns subgraphs from the input node to the exit of the EE locations.
        Not sure why this is needed

        Returns:
            Dict[gt.BlockNode, nx.DiGraph] : the section of the backbone up until the exit location
        """
        ee_base_sections = dict()

        # get starting point of block graph, TODO: currently limited to single input node
        input_node = gt.get_first_input_node(
            self.analysis.architecture.block_graph
        )

        # find the cost of only running the exit, see it as combination of backbone model + early exit classifier
        for pos in self.EE_keras.keys():
            con = nx.shortest_path(
                self.analysis.architecture.block_graph, input_node, pos
            )
            base_section = self.analysis.architecture.block_graph.subgraph(con)
            ee_base_sections[pos] = base_section

        # TODO: should we add the final exit here?

        return ee_base_sections

    def _prepare_EE_cost_map(
        self,
    ) -> Dict[pr.Processor, Dict[gt.BlockNode, Dict]]:
        """Calculates cost metrics for Early Exits based on the available processors.

        Returns:
            Dict[pr.Processor, Dict[gt.BlockNode, Dict]]: the cost map for the early exits
        """

        ee_proc_recom = dict()

        for proc in self.processors:
            ee_proc_recom[proc] = dict()

            # generate support, latency and weight allocation (in %) for the individual layers of the EE branches
            for pos, keras in self.EE_keras.items():
                # get EE cost on different processors
                ee_proc_recom[proc][pos] = dict()

                ee_recom = True
                ee_delay = 0
                ee_macs = 0
                ee_mem = dict()
                for dtype in self.dtypes:
                    ee_mem[dtype] = 0.0

                for layer in keras.layers:
                    recom, delay, mem, macs = proc.check(
                        layer=layer,
                        estimation_functions=self.analysis.compute.mac_estimators,
                        dtypes=self.dtypes,
                    )
                    ee_recom &= recom
                    ee_delay += delay
                    ee_macs += macs
                    for dtype in self.dtypes:
                        ee_mem[dtype] += mem[dtype]

                ee_proc_recom[proc][pos]["support"] = ee_recom
                ee_proc_recom[proc][pos]["delay"] = ee_delay
                ee_proc_recom[proc][pos]["mem_util"] = ee_mem
                ee_proc_recom[proc][pos]["macs"] = ee_macs

        return ee_proc_recom

    def _prepare_partition_cost_map(
        self,
    ) -> Dict[pr.Processor, Dict[Tuple[gt.BlockNode, gt.BlockNode], Dict]]:
        """Calculates cost metrics for all partitions based on the available processors.

        Returns:
            Dict[pr.Processor, Dict[Tuple[gt.BlockNode, gt.BlockNode], Dict]]: the cost map for the partitions
        """

        '''hw_rep_id = self._submitted_reports[hw_check.HWReport]
        hw_info = self.analysis.access_report(hw_rep_id)'''
        hw_info = self.hw_report

        part_proc_recom = dict()
        for proc in self.processors:
            part_proc_recom[proc] = dict()

        # extract partitions between EE
        partitions = dict()
        start_node = gt.get_first_input_node(
            self.analysis.architecture.block_graph
        )
        end_node = gt.get_first_output_node(
            self.analysis.architecture.block_graph
        )

        shortest_path = nx.shortest_path(
            self.analysis.architecture.block_graph, start_node, end_node
        )

        ordered_locations = list(shortest_path)
        ordered_locations = [
            x for x in ordered_locations if x in self.EE_keras.keys()
        ]
        ordered_locations = [start_node] + ordered_locations + [end_node]

        # done = []
        for idx, start in enumerate(ordered_locations):
            start = ordered_locations[idx]
            # done.append(start)

            for end in ordered_locations[idx:]:
                # if end in done:
                #    continue
                key = (start, end)

                for proc in self.processors:
                    part_proc_recom[proc][key] = dict()
                    part_proc_recom[proc][key]["support"] = True
                    part_proc_recom[proc][key]["delay"] = 0
                    part_proc_recom[proc][key]["mem_util"] = dict()
                    part_proc_recom[proc][key]["macs"] = 0

                    for dtype in self.dtypes:
                        part_proc_recom[proc][key]["mem_util"][dtype] = 0.0

                # calcuate cost
                con = list(
                    nx.shortest_path(
                        self.analysis.architecture.block_graph, start, end
                    )
                )
                for block in con:
                    recom, delay, mem, macs = hw_info.check_block_support(
                        block
                    )

                    for proc in recom.keys():
                        part_proc_recom[proc][key]["support"] &= recom[proc]
                        part_proc_recom[proc][key]["delay"] += delay[proc]
                        part_proc_recom[proc][key]["macs"] += macs[proc]
                        for dtype in hw_info.dtypes:
                            part_proc_recom[proc][key]["mem_util"][
                                dtype
                            ] += mem[proc][dtype]

        return part_proc_recom

    def _prepare_transmission_costs(
        self,
    ) -> Dict[gt.BlockNode, Dict[con.Connection, Dict[str, float]]]:
        """Accounts for the transmission of data between the processors by calculating the cost of each IFM for each connection

        Returns:
            Dict[gt.BlockNode, Dict[con.Connection, Dict [str, float]]]: Dict that uses the Exit option as key, and contains the connections and dtypes as dicts to access the transmission time in seconds as float
        """
        transmission_options = dict()
        for pos in self.EE_keras.keys():
            transmission_options[pos] = dict()
            for connection in self.connections.values():
                transmission_options[pos][con] = dict()
                for dtype in self.dtypes:
                    ifm_size = self.analysis.memory.IFM_count[
                        gt.get_first_output_node(pos.subgraph).name
                    ]
                    transmission_options[pos][con][
                        dtype
                    ] = connection.calculate_latency(ifm_size)

        return transmission_options

    def find_initial_mappings(self) -> List[MappingOption]:
        """Creates mapping options for the first processor that fit into given constraints

        Returns:
            List[MappingOption]: MappingOptions for the first processor
        """

        '''hw_rep_id = self._submitted_reports[hw_check.HWReport]
        hw_info = self.analysis.access_report(hw_rep_id)'''
        hw_info = self.hw_report

        partial_solutions: List[MappingOption] = list()
        p0 = self.processors[0]

        # the following lines up to placements = ... were necessary to account for different behavior when Python version is < 3.10
        block_graph = self.analysis.architecture.block_graph
        
        # Get the set of reachable nodes from the source node
        filtered_nodes = set(nx.descendants(block_graph, source=gt.get_first_input_node(block_graph)))

        # Filter the input list to include only reachable nodes
        filtered_nodes = [node for node in list(self.EE_keras.keys()) if node in filtered_nodes]

        # Get the list of nodes in the graph in the order they are visited
        ordered_nodes = list(nx.dfs_preorder_nodes(block_graph, source=gt.get_first_input_node(block_graph)))

        # Sort the filtered list of nodes by their order of appearance
        sorted_nodes = sorted(filtered_nodes, key=lambda node: ordered_nodes.index(node))

        placements = list(sorted_nodes) + [
            gt.get_first_output_node(self.analysis.architecture.block_graph)
        ]

        inp_block = gt.get_first_input_node(
            self.analysis.architecture.block_graph
        )
        for place in placements:
            if place in self.EE_keras.keys():
                # EE
                ee_cost = self.EE_costs[p0][place]
                supported = ee_cost["support"]
                delay = ee_cost["delay"]
                memory_util = ee_cost["mem_util"]

                # out_block = gt.get_first_output_node(self.EE_bases[place]) # that is probably wrong
                out_block = place
            else:
                # Final Exit
                supported = True
                delay = 0.0
                memory_util = dict()
                for dtype in self.dtypes:
                    memory_util[dtype] = 0

                out_block = gt.get_first_output_node(
                    self.analysis.architecture.block_graph
                )

            shortest_path = nx.shortest_path(
                self.analysis.architecture.block_graph,
                source=inp_block,
                target=out_block,
            )[1::]

            for block_node in shortest_path:
                for layer_node in list(block_node.subgraph.nodes()):
                    delay += hw_info.latency[p0][layer_node]
                    supported &= hw_info.supported[p0][layer_node]
                    for dtype in self.dtypes:
                        memory_util[dtype] += hw_info.memory_util[p0][
                            layer_node
                        ][dtype]

            costs = {
                    "delay": delay,
                    "mem_util": memory_util,
                    "support": supported,
                }
            new_solution = MappingOption(
                options=placements, processors=self.processors
            )
            new_solution.add_mapping(processor=p0, node=place, costs=costs)
            
            
            if (
                supported
                and delay <= self.latency
                and max(list(memory_util.values())) <= 1.0
            ):
                partial_solutions.append(new_solution)
                
            else:
                self.rejects.append(new_solution)

        return partial_solutions

    def find_inter_mappings(
        self, partial_solutions: List[MappingOption]
    ) -> List[MappingOption]:
        """uses list of possible mappings for the first processor to generate possible mappings for following processors
        except the last one, which needs to be mapped to final node (for now)

        Args:
            partial_solutions (List[MappingOption]): mapping options for the first processor

        Returns:
            List[MappingOption]: mapping options for all processors except the last
        """
        
        '''hw_rep_id = self._submitted_reports[hw_check.HWReport]
        hw_info = self.analysis.access_report(hw_rep_id)'''
        hw_info = self.hw_report

        for proc in self.processors[1:-1]:
            for solution in partial_solutions:
                # check if this processor has already been assigned in this option
                if proc not in solution.get_unassigned_processors():
                    continue

                # get open mappings in current option, where the remaining processors could be placed
                options = solution.get_open_nodes()

                # create a new mapping option for each possible partial solution in the current mapping option
                for option in options:
                    new_solution = solution.create_copy()

                    last_proc = list(new_solution.mappings.keys())[-1]
                    inp_block = list(
                        self.analysis.architecture.block_graph.successors(
                            new_solution.mappings[last_proc]
                        )
                    )[
                        0
                    ]  # currently limited to single successor

                    # extract cost of mapping option (i.e. the execution of its early exit branch, if existing)
                    if option in self.EE_keras.keys():
                        ee_cost = self.EE_costs[proc][option]
                        supported = ee_cost["support"]
                        delay = ee_cost["delay"]
                        memory_util = ee_cost["mem_util"]

                    else:
                        # Final Exit
                        # TODO: check if this works out
                        supported = True
                        delay = 0.0
                        memory_util = dict()
                        for dtype in self.dtypes:
                            memory_util[dtype] = 0

                    out_block = option

                    # extract all nodes that will be mapped to proc for this option
                    try:
                        shortest_path = nx.shortest_path( # crashes on mobilenetv2_1.00_224 - No path between residual_0 and convolution_0.
                            self.analysis.architecture.block_graph,
                            source=inp_block,
                            target=out_block,
                        )[1::]
                    except nx.NetworkXNoPath as no_path:
                        log.error("tried to map to previous EE")

                    # calculate their cost for this mapping
                    for block_node in shortest_path:
                        for layer_node in list(block_node.subgraph.nodes()):
                            delay += hw_info.latency[proc][layer_node]
                            supported &= hw_info.supported[proc][layer_node]
                            for dtype in self.dtypes:
                                memory_util[dtype] += hw_info.memory_util[
                                    proc
                                ][layer_node][dtype]

                    ## we also need to account for transmission latency
                    if (last_proc, proc) in self.connections.keys():
                        connection = self.connections[(last_proc, proc)]
                        delay += connection.calculate_latency(
                            np.prod(inp_block.input_shape) * 4
                        )  # assuming 4 byte for now
                    else:
                        log.debug(
                            f"mapping for {proc} on {option} was not added, as no connection exits to transport intermediate results from {last_proc}"
                        )
                        continue

                    # check if created option fits constraints before adding to solution space
                    costs = {
                        "delay": delay,
                        "mem_util": memory_util,
                        "support": supported,
                    }

                    new_solution.add_mapping(
                        processor=proc, node=option, costs=costs
                    )
                
                    if (
                        supported
                        and delay + solution.accumulate_cost("delay")
                        <= self.latency
                        and max(list(memory_util.values())) <= 1.0
                    ):
                        delay = new_solution.accumulate_cost("delay")
                        supported = new_solution.accumulate_cost("support")
                        if delay <= self.latency and supported:
                            partial_solutions.append(new_solution)
                            log.info(
                                "SolutionOption was added, as it does not exceed latency requirements with {delay} instead of {latency}"
                            )
                        else:
                            self.rejects.append(new_solution)
                            log.info(
                                "SolutionOption was rejected as it would exceed the worst-case latency with {delay} instead of {latency}"
                            )
                    else:
                        self.rejects.append(new_solution)

        return partial_solutions

    def find_final_mappings(
        self, partial_solutions: List[MappingOption]
    ) -> List[MappingOption]:
        """takes a list of possible mappings for the previous processors and returns a list of complete viable mapping options.
        These are options where the full NN graph has been assigned to the available processors and the solution is estimated to be within the given constraints.
        WARNING: this could return an empty list, if no solution for the given resource constraints exists

        Args:
            partial_solutions (List[MappingOption]): partial solutions of previous processors to previous options

        Returns:
            List[MappingOption]: list of full mappings
        """

        solution_space = list()
        out_block = gt.get_first_output_node(
            self.analysis.architecture.block_graph
        )
        proc = self.processors[-1]

        for solution in partial_solutions:
            # need to check, if the solution already is complete
            if len(solution.get_open_nodes()) == 0:
                # if last node has already been assigned, just add solution to complete solution space
                solution_space.append(solution)
                continue

            # solution is not complete yet, we need to map the last processor to the last node

            # get last mapped processor and last mapped placement
            last_proc, last_mapping, mapping_cost = solution.get_mappings()[-1]

            # check, if connection between last and current processor exists, get transmission cost
            if (last_proc, proc) in self.connections.keys():
                connection = self.connections[(last_proc, proc)]
                delay = connection.calculate_latency(
                    np.prod(last_mapping.input_shape) * 4
                )  # assuming 4 byte for now
            else:
                log.debug(
                    f"mapping for {proc} on {solution} was not added, as no connection exits to transport intermediate results from {last_proc}"
                )
                continue

            memory_util = dict()
            for dtype in self.dtypes:
                memory_util[dtype] = 0
            supported = True

            # get cost of mapping the remaining nodes to last processor
            shortest_path = nx.shortest_path(
                self.analysis.architecture.block_graph,
                source=last_mapping,
                target=out_block,
            )[1::]

            '''hw_rep_id = self._submitted_reports[hw_check.HWReport]
            hw_info = self.analysis.access_report(hw_rep_id)'''
            hw_info = self.hw_report

            for block_node in shortest_path:
                for layer_node in list(block_node.subgraph.nodes()):
                    delay += hw_info.latency[proc][layer_node]
                    supported &= hw_info.supported[proc][layer_node]
                    for dtype in self.dtypes:
                        memory_util[dtype] += hw_info.memory_util[proc][layer_node][dtype]

            costs = {
                "delay": delay,
                "mem_util": memory_util,
                "support": supported,
            }
            solution.add_mapping(
                processor=proc, node=out_block, costs=costs
            )

            # check if created solution fits constraints, add to solution_space if fitting
            if (
                supported
                and delay <= self.latency
                and max(list(memory_util.values())) <= 1.0
            ):
                costs = {
                    "delay": delay,
                    "mem_util": memory_util,
                    "support": supported,
                }

                delay = solution.accumulate_cost("delay")
                supported = solution.accumulate_cost("support")
                if delay <= self.latency and supported:
                    solution_space.append(solution)
                    log.info(
                        f"SolutionOption was added, as it does not exceed latency requirements with {delay} instead of {self.latency}"
                    )
                else:
                    self.rejects.append(solution)
                    log.info(
                        f"SolutionOption was rejected as it would exceed the worst-case latency with {delay} instead of {self.latency}"
                    )
            else:
                self.rejects.append(solution)

        return solution_space

    def extract_all_options(self) -> List[MappingOption]:
        """Creates solution space, that contains all viable solutions and can be used to find the most suitable solutions

        Returns:
            List[MappingOption]: List of possible solutions
        """
        # start with options to place the first processors
        partial_solutions = self.find_initial_mappings()
        partial_solutions = self.find_inter_mappings(
            partial_solutions=partial_solutions
        )

        '''out_block = gt.get_first_output_node(
            self.analysis.architecture.block_graph
        )'''

        solution_space = self.find_final_mappings(
            partial_solutions=partial_solutions
        )

        return solution_space
    
    def create_identifier(self) -> str:
        """
        Creates a unique identfier for the report to enable checking for duplicates.

        Returns:
            str: the unique identifier
        """
        descriptor = f"HeterogeneousPlatforms:{self.analysis.keras_model.name}, {self.latency}, {self.processors}, {self.connections}, {self.dtypes}"
        descriptor += f"{self.train_dataset.access_id()}, {self.validation_dataset.access_id()}, {self.test_dataset.access_id()}"
        sha256_hash = hashlib.sha256()
        
        # Update the hash object with the input string
        sha256_hash.update(descriptor.encode('utf-8'))
        hashed_representation = sha256_hash.hexdigest()

        return hashed_representation
    
    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> Dict[str, object]:
        """
        Writes the rewriter to disk.

        Args:
            folder_path (Union[str, pathlib.Path], optional): The folder in which the dump will be stored. Uses current working directory if None. Defaults to None.

        Returns:
            Dict[str, object]: dict representation of the rewriter.
        """

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)
        
        solutions_path = folder_path / f"rewriter_hetero_platforms_{self.create_identifier()}"
        solutions_path.mkdir(parents=True, exist_ok=True)
        solutions = {}

        if hasattr(self, "final"):
            for name, sol in self.final.items():
                serial_solution = sol.dump(solutions_path) #sol.toDict(include_objects=False)

                '''sub_model_paths = {}
                for submodel in sol.split():
                    sub_filename = f"{name}-{submodel.name}.keras"
                    submodel.save(solutions_path /sub_filename)
                    sub_model_paths[submodel.name] = sub_filename
                eenn_filename = f"{name}-{sol.model.name}-full.keras"
                sol.model.save(solutions_path / eenn_filename)
                sub_model_paths[sol.model.name] = eenn_filename
                serial_solution["models"] = sub_model_paths

                thres_filename = f"{name}-thresholds.json"
                with open(solutions_path / thres_filename, "w") as file:
                    json.dump(sol.thresholds, file)
                serial_solution["threshold_config_file_path"] = thres_filename
                serial_solution["relative_path"] = f"./{solutions_path.parts[-1]}/"'''

                solutions[name] = serial_solution
        else:
            log.info("no solutions have been found yet")
                        
        summary = {
            "report_type": "Heterogeneous Platform",
            "name": self.analysis.name,
            "creation_date": str(self.analysis.creation_time),
            "options": [option.toDict() for option in self.options],
            "rejects": [option.toDict() for option in self.rejects],
            "solutions": solutions,
            "processors": [proc.toDict() for proc in self.processors],
            "connections": [connection.toDict() for connection in self.connections.values()],
        }

        with open(folder_path / f"rewriter_hetero_platforms_{self.create_identifier()}.json", "w") as file:
            json.dump(summary, file)

        return summary

    def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> Tuple[str, str]:
        """
        Creates the HTML file for the summary overview

        Args:
            folder_path (Union[str, pathlib.Path], optional): folder, in which the file and auxiliary data will be stored. Defaults to None.
        
        Returns:
            Tuple[str, str]: The name of the rewriter and the path to its html summary.
        """
        _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'

        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)

        with open(_template_path / "hetero_rewriter.html", "r") as file:
            template = Template(file.read())

        summary = self.dump(folder_path=folder_path)

        summary["options"] = self.options
        summary["rejects"] = self.rejects
        summary["processors"] = self.processors
        summary["connections"] = list(self.connections.values())
        #summary["solutions"] = self.final

        text = f"{self.analysis.name} is a Keras model. \
                {len(self.options)} solutions have been found that lie within the given constraints."

        summary["text"] = text
        # Render the template with the summary data
        html = template.render(summary=summary)
        # Save the generated HTML to a file
        html_filename = f"optimization_heterogeneous_platform_{self.create_identifier()}.html"
        html_path = folder_path / html_filename
        with open(html_path, "w") as file:
            file.write(html)

        return (
            "Heterogeneous Platforms",
            html_filename,
        )
    
    @classmethod
    def create_pass(cls, latency_constraint:float, processors, connections, dtypes, train_data:tf.data.Dataset, valid_data:tf.data.Dataset, test_data:tf.data.Dataset, focus:str="mpsoc", finetune:bool=True) -> Tuple[str, callable]:
        """
        Creates a pass that can be added to the optimization queue of a model analysis object.
        This pass contains all steps necessary to produce a solution of this rewrite from the ModelAnalysis of the submitted model.

        Returns a name for the pass as well as the function that needs to be called to apply the rewrite to the analysis

        Args:
            latency_constraint (float): The worst case latency search constraint.
            processors (_type_): The processors to which the inference workload will be mapped.
            connections (_type_): The connections between the processors.
            dtypes (_type_): The considered datatypes for the memory utilization analysis.
            train_data (tf.data.Dataset): The training data report wrapper.
            valid_data (tf.data.Dataset): The validation data report wrapper.
            test_data (tf.data.Dataset): The test data report wrapper.
            focus (str, optional): The search flow that will be used. Defaults to "mpsoc".
            finetune (bool, optional): If the found solution should be finetuned. Defaults to True.

        Returns:
            Tuple[str, callable]: The pass name as well as its callable implementation.
        """

        str_id:str = f"HeterogeneousPlatformsRewrite_{latency_constraint}, {processors}, {connections}, {dtypes}, {focus}"

        def rewrite(analysis:aNN.ModelAnalysis) -> List[HeterogeneousPlatformSolution]:

            # create dataset reports
            train_data_report = data.DatasetReport.submit_to(analysis, lazy=False).with_config(name="train", modality=None).from_source(tf_dataset=train_data)

            valid_data_report = None
            if valid_data is not None:
                valid_data_report = data.DatasetReport.submit_to(analysis, lazy=False).with_config(name="valid", modality=None).from_source(tf_dataset=valid_data)

            test_data_report = None
            if test_data is not None:
                test_data_report = data.DatasetReport.submit_to(analysis, lazy=False).with_config(name="test", modality=None).from_source(tf_dataset=test_data)

            # create rewriter
            rewriter = HeterogeneousPlatformRewriter(analysis=analysis,
                                                     latency=latency_constraint,
                                                     processors=processors,
                                                     connections=connections,
                                                     dtypes=dtypes,
                                                     train_dataset=train_data_report,
                                                     validation_dataset=valid_data_report,
                                                     test_dataset=test_data_report
                                                     )
            
            # execute pass
            sol = rewriter.select_best_option(focus=focus, fine_tune=finetune)
            # collect solution
            return [sol]
        
        return str_id, rewrite
