import copy
import shutil
import logging as log
import json

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Union, Type
from typing_extensions import Self
import pathlib
from datetime import datetime
import time
import os
from abc import ABC, abstractmethod
import pickle as pkl

from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.use('Agg')

import networkx as nx

from jinja2 import Template

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as keras

from .components import architecture as arch
from .components import resource as res
from .components import predictive as prd
from .components import tflm_support as tflm
from .components import visualize_graph as vis
from .components import graph_tools as gt

from .rewriters import base

class RewriterQueue():
    """
    This class represents the queue of rewriters that will be applied, if solutions from the current ModelAnalysis instance are converted back into
    new ModelAnalysis instances.
    One Rewriter Queue is associated with each ModelAnalysis and consists of multiple stages.
    Each stage can contain multiple rewriters which will be automatically applied when a new ModelAnalysis instance is created from a solution.
    TODO: evaluate if the submission of closures, that do not only perform the instanciation but also the execution of the rewrites to generate a solution.
    """

    stages : List[Dict[str, callable]]
    """The stages that are part of the queue.
    Each stage can contain multiple different passes and each pass is identified by a unique string."""

    history : List[base.Solution]
    """The solutions created by already executed stages."""

    def __init__(self) -> None:
        self.stages = []
        self.history = []

    def add_stage(self) -> None:
        """
        Adds a new stage to the queue.
        """
        self.stages.append({})

    @property
    def stage_count(self) -> int:
        """
        Returns the current number of stages in the queue.

        Returns:
            int: The stage count
        """
        return len(self.stages)

    def add_pass(self, rewriting_pass:callable, str_id:str, new_stage:bool=False, overwrite:bool=False) -> None:
        """
        Adds a new rewriting pass to the queue.
        The pass is a callable function that should produce a Solution object.
        Each pass is identified by a unique ID to perform each operation only once per stage.
        new_stage creates a new stage in the queue, if True, otherwise the pass is added to the latest stage.

        Args:
            rewriting_pass (callable): The function that performs the pass
            str_id (str): The unique identifier of the pass
            new_stage (bool, optional): If True, the pass will be added to a new stage in the queue. Defaults to False.
            overwrite (bool, optional): If True, an existing pass with the same ID in the same stage will be overwritten. Defaults to False.
        """
        if new_stage or self.stage_count == 0:
            self.add_stage()

        last_stage :dict = self.stages[-1]

        if str_id in last_stage.keys() and not overwrite:
            log.warn("Did not add pass to queue, as its unique ID already exists in the stage.")
            return None

        last_stage[str_id] = rewriting_pass

    def add_pass_at(self, index:int, rewriting_pass:callable, str_id:str, overwrite:bool=False) -> None:
        """
        Adds a new rewriting pass to the queue.
        The pass is a callable function that should produce a Solution object.
        Each pass is identified by a unique ID to perform each operation only once per stage.
        This function allows you to add the pass at a stage that is not the last stage in the queue.

        Args:
            index (int): the stage to which the pass should be added.
            rewriting_pass (callable): The function that performs the pass
            str_id (str): The unique identifier of the pass
            overwrite (bool, optional): If True, an existing pass with the same ID in the same stage will be overwritten. Defaults to False.

        Raises:
            IndexError: If the index is larger than the queue length.
        """

        if index >= self.stage_count:
            raise IndexError("the index is larger than the depth of the queue.")

        if str_id in self.stages[index].keys() and not overwrite:
            log.warn("Did not add pass to queue, as its unique ID already exists in the stage.")
            return None

        self.stages[index][str_id] = rewriting_pass

    def get_first_stage(self) -> Dict[str, callable]:
        """
        Returns the first stage in the queue.
        This stage is the next to be processed.

        Returns:
            Dict[str, callable]: The passes within the stage
        """

        return self.stages[0]

    def iterate(self) -> Self:
        """
        Pops the first stage of the queue and returns an updated queue

        Returns:
            Self: An updated instance of the RewriterQueue.
        """

        new_stages = self.stages[1::]
        new_queue = RewriterQueue()

        new_queue.history = copy.copy(self.history)
        new_queue.stages = new_stages

        return new_queue

    def step(self, analysis, return_analysis:bool=True) -> Dict[str, object]:
        """
        Performs the next queue iteration.
        Pops the first stage, executes it and returns the created solutions as new ModelAnalysis objects.

        Args:
            analysis (ModelAnalysis): the ModelAnalysis to which the stage will be applied.
            return_analysis (bool): If True, ModelAnalysis objects will be created from the found Solutions to perform the next stage of the RewriteQueue.

        Returns:
            Dict[str, object]: The newly created solutions.
        """

        # get first stage
        exec_stage = self.get_first_stage()
        new_queue = self.iterate()

        # execute them
        sols = {}
        for pass_id, pass_func in exec_stage.items():
            print(pass_id)
            new_sol = pass_func(analysis=analysis) # do we need to pass anything here? like the ModelAnalysis object?
            
            # collect the solutions and convert them to ModelAnalysis'
            if return_analysis:
                new_anas = new_sol.to_analysis()
                results = []
                for part_id, new_ana in enumerate(new_anas):
                    #new_ana = part.to_analysis()
                    if return_analysis:
                        # Update Queue
                        new_ana.optimization_queue = new_queue
                        new_ana.name = f"{new_ana.name}-{pass_id}-{part_id}"
                    results.append(new_ana)

                sols[pass_id] = results
            else:
                sols[pass_id] = new_sol

        return sols

class BaseAnalysis(ABC):
    """
    Abstract Base class for Analysis classes.
    Contains a version string that references the version of the Analysis object,
    which is supposed to aid with (de)serialization of stored or cached instances.
    """

    keras_model : tf.keras.models.Model
    _version : str = "0.3"

    @abstractmethod
    def dump(self, folder_path : Union[str, pathlib.Path]) -> None:
        """
        Writes the BaseAnalysis to disk by creating the necessary file within the given folder

        Args:
            folder_path (Union[str, pathlib.Path]): the folder in which the files should be created
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, folder_path : Union[str, pathlib.Path]) -> Self:
        """
        Creates a new instance of the BaseAnalysis class from information that was previously written into the given folder

        Args:
            folder_path (Union[str, pathlib.Path]): the folder in which the files are stored

        Returns:
            Self: the BaseAnalysis instance
        """
        pass

class PartialAnalysis(BaseAnalysis):
    """The PartialAnalysis class represents the subcontainers of the ModelAnalysis class.
    These subcontainers maintain the information about the model regarding one specific topic
    (i.e. information about its architecture, etc.).
    """

    def __init__(self, analysis) -> None:
        self.analysis = analysis
        self.keras_model = self.analysis.keras_model

    @classmethod
    @abstractmethod
    def load(cls, folder_path : Union[str, pathlib.Path], analysis : Self) -> Self:
        """
        Creates a new instance of the PartialAnalysis class from information that was previously written into the given folder

        Args:
            folder_path (Union[str, pathlib.Path]): the folder in which the files are stored
            analysis (Self): The ModelAnalysis instance to which the PartialAnalysis will be associated

        Returns:
            Self: the PartialAnalysis instance
        """
        pass


@dataclass
class ArchitectureAnalysis(PartialAnalysis):
    """PartialAnalysis class that contains all analysis data regarding the architecture of the submitted Neural Network"""

    __json_dump_name = "architecture.json"

    __block_dump_name = "block.peaxgraph"

    __layer_dump_name = "layer.peaxgraph"

    is_feed_forward : bool
    """boolean, is True if it is a feed forward architecture"""

    is_recurrent : bool
    """boolean, is True if it is an RNN/LSTM/GRU or similar"""

    is_branching : bool
    """bool, is True, if there are parallel branches in the architecture"""

    inputs : List[str]
    """list of the names of the input tensors of the architecture"""

    outputs : List[str]
    """list of the names of the output layers"""

    network_graph : nx.DiGraph
    """representation of the network at the layer-level as directed graph"""

    block_graph : nx.DiGraph
    """representation of the network at the block-level as directed graph"""

    def __init__(self, analysis, deserialize:bool=False) -> None:

        if analysis is not None:
            super().__init__(analysis)
        else:
            self.keras_model = None
            self.analysis = None
            self._version = "UNKNOWN"

        if not deserialize:

            network_graph = gt.convert_to_graph(self.keras_model)
            # test for typical architectures TODO:expand in the future
            self.is_feed_forward = arch.is_feed_forward(network_graph)
            self.is_recurrent = arch.is_recurrent(network_graph)
            self.is_branching = arch.is_branching(network_graph)

            self.inputs = self.keras_model.input_names
            self.outputs = list(arch.identify_output_layers(model=self.keras_model).keys())

            self.network_graph = network_graph
            self.block_graph = arch.identify_blocks(model=self.keras_model)
        else:
            self.is_feed_forward = None
            self.is_recurrent = None
            self.is_branching = None

            self.inputs = None
            self.outputs = None

            self.network_graph = None
            self.block_graph = None

        if self.is_recurrent:
            log.warn("not tested for recurrent architectures yet")

    def visualize(self, folder_path: Union[str, pathlib.Path] = None) -> None:
        """Creates png files for the layer- and block-level graph representation.
        The files are written into the folder at folder_path.

        Args:
            folder_path (Union[str, pathlib.Path], optional): Where the image files are supposed to be stored. Defaults to None.
        """
        """  """
        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)

        folder_path.mkdir(exist_ok=True, parents=True)

        vis.block_graph(
            graph=self.block_graph, path=folder_path / "block_graph.png"
        )
        vis.graph(
            graph=self.network_graph, path=folder_path / "network_graph.png"
        )

        return

    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> None:
        """Dumps all of the information to disk.
        Can be used to reuse its output in another tool.
        Creates three files:\n
        - an 'architecture.json' file that contains information about the network graph properties
        - a 'block.peaxgraph' file that contains the block-level graph representation
        - a 'layer.peaxgraph' file that contains the layer-level graph representation
        WARNING: pickle is used to create the peaxgraph files!

        Args:
            folder_path (Union[str, pathlib.Path], optional): the folder where it should be stored. Defaults to None.
        """
        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        '''from networkx.readwrite import json_graph

        block_graph_dict = nx.node_link_data(self.block_graph)

        network_graph_dict = nx.node_link_data(self.network_graph)

        # TODO: fix
        with open(folder_path / "network_graph.json", "w") as file:
            json.dump(network_graph_dict, file)

        with open(folder_path / "block_graph.json", "w") as file:
            json.dump(block_graph_dict, file)'''

        summary = {
            "feed-forward": self.is_feed_forward,
            "branching": self.is_branching,
            "recurrent": self.is_recurrent,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "version": self._version,
        }

        with open(folder_path / self.__json_dump_name, "w") as file:
            json.dump(summary, file)

        '''with open(folder_path / self.__layer_dump_name, "wb") as file:
            pkl.dump(self.network_graph, file)'''

        ''' with open(folder_path / self.__block_dump_name, "wb") as file:
            pkl.dump(self.block_graph, file)''' # cannot be pickled due to storing subgraphs within block nodes

        return
    
    def identify_feature_extraction_subgraph(self, layer_level:bool=False) -> nx.DiGraph:
        """extracts the blocks that correspond to the hidden layers of the original model

        Returns:
            nx.DiGraph: the feature extraction subgraph as its own nx.DiGraph
        """

        graph = self.block_graph if not layer_level else self.network_graph
        classifier = self.identify_classifier_subgraph(layer_level=layer_level)

        nodes = set(graph.nodes)
        classifier_nodes = set(classifier.nodes)

        maintain_nodes = [x for x in nodes if x not in classifier_nodes]

        feature_extraction_subgraph = graph.subgraph(
            maintain_nodes
        )
        feature_extraction_subgraph.name = "feature extraction"
        
        return feature_extraction_subgraph
    
    def identify_classifier_subgraph(self, layer_level:bool=False) -> nx.DiGraph:
        """extracts the blocks that correspond to the final classifier of the original model

        Returns:
            nx.DiGraph: the classifier subgraph as its own nx.DiGraph
        """

        out_block = gt.get_first_output_node(self.block_graph)
        predecessor = list(self.block_graph.predecessors(out_block))[0]

        out_classifier_blocks = [out_block]

        ### check if we have an output activation function or if the model was trained with "from_logits"
        out_layer = gt.get_first_output_node(out_block.subgraph)
        if out_layer.layer_class in ["compute", "convolution"]:
            out_config = out_layer.keras.get_config()
            out_act = out_config["activation"]

            if out_act == "linear":
                log.warning("classifier subgraph appears to not contain a final activation function, trying to fix this...")
                ### select which function type would be appropriate
                if out_config["units"] > 1 and list(self.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
                    ### multiclass problem?
                    act_func = tf.keras.activations.softmax # "softmax"
                
                out_layer.keras.activation = act_func

        if not out_block.dominant_operation in ["compute"]:
            out_classifier_blocks.append(predecessor)
            predecessor = list(self.block_graph.predecessors(predecessor))[0]

        while not predecessor.dominant_operation in ["compute", "convolution"]:
            if predecessor.dominant_operation in ["training_layer"]:
                predecessor = list(
                    self.block_graph.predecessors(predecessor)
                )[0]
                continue

            out_classifier_blocks.append(predecessor)
            predecessor = list(self.block_graph.predecessors(predecessor))[0]

        block_classifier_subgraph = self.block_graph.subgraph(
            out_classifier_blocks
        )
        block_classifier_subgraph.name = "classifier"

        if layer_level:
            log.info("returning classifier subgraph in layer-level representation")
            layer_nodes = []
            for block in block_classifier_subgraph.nodes():
                layer_nodes += list(block.subgraph.nodes)

        layer_classifier_subgraph = self.network_graph.subgraph(
            layer_nodes
        )
        layer_classifier_subgraph.name = "classifier"

        return layer_classifier_subgraph

        return block_classifier_subgraph

    @classmethod
    def load(cls, folder_path: Union[str, pathlib.Path], analysis : BaseAnalysis = None) -> PartialAnalysis:
        """Experimential method to load the ArchitectureAnalysis object again from an information dump.\n
        **WARNING: pickle is used to create the peaxgraph files! Ensure that you only load data from trustworthy sources!**

        Args:
            folder_path (Union[str, pathlib.Path]): the folder to which the ModelAnalysis got dumped
            analysis (BaseAnalysis): the ModelAnalysis that contains this specific ArchitectureAnalysis

        Raises:
            FileNotFoundError: raises an error, if the necessary files could not be found

        Returns:
            ArchitectureAnalysis: the new ArchitectureAnalysis
        """
        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        json_path = folder_path / cls.__json_dump_name
        #block_graph_path = folder_path / cls.__block_dump_name
        layer_graph_path = folder_path / cls.__layer_dump_name

        #if not (json_path.exists() and block_graph_path.exists() and layer_graph_path.exists()):
        if not json_path.exists():
            raise FileNotFoundError("The required files for loading the ArchitectureAnalysis object are missing.")

        if not layer_graph_path.exists():
            log.warn("Layer-level Graph representation was not found, will need to be regenerated!")

        # load JSON information
        basic_information = {}
        with open(json_path, "r") as file:
            basic_information = json.load(file)

        arch_analysis = cls(analysis=analysis, deserialize=True)
        arch_analysis.is_feed_forward = basic_information["feed-forward"]
        arch_analysis.is_recurrent = basic_information["recurrent"]
        arch_analysis.is_branching = basic_information["branching"]
        arch_analysis.inputs = basic_information["inputs"]
        arch_analysis.outputs = basic_information["outputs"]
        arch_analysis._version = basic_information["version"]

        '''with open(block_graph_path, "rb") as file:
            block_graph = pkl.load(file)'''
        block_graph = arch.identify_blocks(model=arch_analysis.keras_model)

        arch_analysis.block_graph = block_graph

        '''with open(layer_graph_path, "rb") as file:
            network_graph = pkl.load(file)'''
        network_graph = gt.convert_to_graph(arch_analysis.keras_model)

        arch_analysis.network_graph = network_graph

        return arch_analysis

    def __str__(self) -> str:
        return f"ArchitectureAnalysis: FF={self.is_feed_forward}, RNN={self.is_recurrent}, BRANCH={self.is_branching}, inputs={self.inputs}, outputs={self.outputs}"

    def __repr__(self) -> str:
        return f"ArchitectureAnalysis(analysis={self.analysis})"

@dataclass
class ComputeAnalysis(PartialAnalysis):
    """This class contains all analysis information with regards to the compute operations"""

    __json_file_name = "compute.json"
    __est_dict_file_name = "estimator_functions.pkl"

    mac_estimators : Dict[object, callable]
    """the functions that are used to calculate the amount of MAC operations that need to be executed for the layer workloads"""

    mac_estimator_config : str
    """string that defines, how the mac_estimators are currently configured, either "default" or "custom" """

    layer_mac : Dict[str, int]
    """A dict that contains the MAC workload of each layer, the keys are the individual layer names"""

    total_mac : int
    """the total MAC operations of the network inference"""

    def __init__(
        self,
        analysis,
        estimation_functions: Dict[tf.keras.layers.Layer, callable] = None,
        deserialize:bool=False
    ) -> None:
        if analysis is not None:
            super().__init__(analysis)
        else:
            self.keras_model = None
            self.analysis = None
            self._version = "UNKNOWN"

        if not deserialize:
            if estimation_functions is None:
                self.mac_estimators = res._default_mac_estimators
                self.mac_estimator_config = "default"
            else:
                self.mac_estimators = res._default_mac_estimators
                self.mac_estimators.update(estimation_functions)
                self.mac_estimator_config = "custom"
                log.info("custom estimator function have been configured.")

            self.layer_mac = res.get_model_macs(
                model=self.keras_model, estimation_functions=self.mac_estimators
            )
            self.total_mac = res.get_output_macs(
                model=self.keras_model, estimation_functions=self.mac_estimators
            )
        else:
            self.mac_estimators = res._default_mac_estimators
            self.mac_estimator_config = "unknown"

            self.layer_mac = None
            self.total_mac = None

    def __str__(self) -> str:
        return f"ComputeAnalysis: total_mac={self.total_mac}, estimator_config={self.mac_estimator_config}"

    def __repr__(self) -> str:
        return f"ComputeAnalysis(analysis={self.analysis}, estimator_functions={self.mac_estimators})"

    def visualize(
        self,
        folder_path: Union[str, pathlib.Path] = None,
        bar: bool = True,
        pie: bool = True,
    ) -> None:
        """This function visualizes the required operations per layer for the analyzed model.
        The figures will be written as png files to the given folder path

        Args:
            folder_path (Union[str, pathlib.Path], optional): Where the image files are supposed to be stored. Defaults to None.
            bar (bool, optional): if a bar chart should be created. Defaults to True.
            pie (bool, optional): if a bar chart should be created. Defaults to True.

        Returns:
            None
        """
        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)

        folder_path.mkdir(exist_ok=True, parents=True)

        keys = list(self.layer_mac.keys())
        values = list(self.layer_mac.values())

        if pie:
            fig, ax = plt.subplots()

            ax.pie(values, autopct="%1.1f%%")
            ax.axis("equal")
            ax.legend(keys)
            ax.set_title(
                "Distribution of MAC Operations across Model Layers", pad=20
            )
            fig.tight_layout()

            file_path = folder_path / "mac_report_pie.svg"
            fig.savefig(file_path)

            file_path = folder_path / "mac_report_pie.png"
            fig.savefig(file_path)

        if bar:
            fig, ax = plt.subplots()

            ax.bar(keys, values)
            ax.set_title(
                "Distribution of MAC Operations across Model Layers", pad=20
            )
            ax.set_xlabel("Layer")
            ax.set_ylabel("MACs")
            plt.xticks(rotation=90)
            fig.tight_layout()

            file_path = folder_path / "mac_report_bar.svg"
            fig.savefig(file_path)

            file_path = folder_path / "mac_report_bar.png"
            fig.savefig(file_path)

        return None

    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> None:
        """
        Dumps the ComputeAnalysis object into this folder.
        This function creates two files:\n
         - a 'compute.json' file that contains the estimated MAC footprints for each layer, the total MAC estimation and a version string.
         - a 'estimator_functions.pkl' file that contains the custom functions that were used to estimate the layerwise MAC footprints.

        Raises:
            FileNotFoundError: raises an error, if the necessary files could not be found

        Args:
            folder_path (Union[str, pathlib.Path], optional): The folder to which the data is supposed to be dumped. Defaults to None. Uses current working directory if no folder_path is given.
        """

        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)

        folder_path.mkdir(exist_ok=True, parents=True)

        summary = {
            "estimator_config": self.mac_estimator_config,
            "layer_mac": [(key, int(x)) for key, x in self.layer_mac.items()],
            "total_mac": [(key, int(x)) for key, x in self.total_mac.items()],
            "version": self._version,
        }

        if self.mac_estimator_config != "default":
            with open(folder_path / self.__est_dict_file_name, "wb") as file:
                pkl.dump(self.mac_estimators, file)

        with open(folder_path / self.__json_file_name, "w") as file:
            json.dump(summary, file)

        return

    @classmethod
    def load(cls, folder_path: Union[str, pathlib.Path], analysis : BaseAnalysis = None) -> PartialAnalysis:
        """
        Experimential data loader for the ComputeAnalysis class.
        Creates a new instance from the data that has been dumped to disk.
        **WARNING: only load data from trustworthy sources as custom executable code
        can be introduced through the stored estimator functions!**

        Args:
            folder_path (Union[str, pathlib.Path]): the folder path were the dumped data is located
            analysis (_type_, optional): The ModelAnalysis object to which this ComputeAnalysis should be associated. Defaults to None.

        Returns:
            PartialAnalysis: A new ComputeAnalysis object
        """
        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        json_path = folder_path / cls.__json_file_name
        est_path = folder_path / cls.__est_dict_file_name

        if not json_path.exists():
            raise FileNotFoundError("The required JSON file for loading the ComputeAnalysis object is missing.")

        compute_analysis = cls(analysis=analysis, deserialize=True)

        with open(json_path, "r") as file:
            summary = json.load(file)

        compute_analysis.mac_estimator_config = summary["estimator_config"]
        compute_analysis.layer_mac = dict(summary["layer_mac"])
        compute_analysis.total_mac = dict(summary["total_mac"])
        compute_analysis._version = summary["version"]

        if summary["estimator_config"] != "default":

            if not est_path.exists():
                raise FileNotFoundError("The required estimation files for loading the ComputeAnalysis object is missing.")

            with open(est_path, "rb") as file:
                est_config = pkl.load(file)

            compute_analysis.mac_estimators.update(est_config)

        return compute_analysis

@dataclass
class MemoryAnalysis(PartialAnalysis):
    """Class that contains all information with regards to the memory allocation of the analyzed model."""
    __json_file_name = "memory.json"

    IFM_dimensions : Dict[str, Tuple[int]]
    """the intermediate feature map dimensions of the model architecture, identified by the layer that created them"""
    IFM_count : Dict[str, int]
    """the amount of elements of each IFM, identified by the layer that created them"""

    total_IFM : int
    """total amount of IFM elements within the network architecture"""

    def __init__(self, analysis, deserialize:bool=False) -> None:
        if analysis is not None:
            super().__init__(analysis)
        else:
            self.keras_model = None
            self.analysis = None
            self._version = "UNKNOWN"

        if not deserialize:
            self.IFM_dimensions = res.get_model_IFM_dims(model=self.keras_model)
            self.IFM_count = res.get_model_IFM_count(model=self.keras_model)
            self.total_IFM = int(np.sum(list(self.IFM_count.values())))
        else:
            self.IFM_dimensions = {}
            self.IFM_count = {}
            self.total_IFM = 0
        pass

    def estimate_total_IFM_size(self, dtype: str = "float") -> int:
        """Estimates of the total sum of all IFM's if they are using the given datatype.
        Sub-byte types are assumed to be densly packed.

        Args:
            dtype (str, optional): str that describes the assumed datatype (i.e. "float32", "float64", "int8"). Defaults to "float".

        Returns:
            int: total memory allocation in byte
        """
        if res._datatype_widths[dtype] < 8:
            log.info(
                f"{dtype} is {res._datatype_widths[dtype]} bit long, assumes dense packing for sub-byte types"
            )
        return np.ceil(self.total_IFM * res._datatype_widths[dtype] / 8)

    def estimate_IFM_size_for_layer(
        self, layer_name: str, dtype: str = "float"
    ) -> int:
        """Estimates the size of a layers IFM if they are using the given datatype.
        Sub-byte types are assumed to be densly packed.

        Args:
            layer_name (str): name of the layer, whose input will be analyzed
            dtype (str, optional): str that describes the assumed datatype (i.e. "float32", "float64", "int8"). Defaults to "float".

        Returns:
            int: layer's IFM memory allocation in byte
        """
        if res._datatype_widths[dtype] < 8:
            log.info(
                f"{dtype} is {res._datatype_widths[dtype]} bit long, assumes dense packing for sub-byte types"
            )
        return np.ceil(
            self.IFM_count[layer_name] * res._datatype_widths[dtype] / 8
        )

    def __str__(self) -> str:
        return f"MemoryAnalysis: total_elements={self.total_IFM}"

    def __repr__(self) -> str:
        return f"MemoryAnalysis(analysis={repr(self.analysis)})"

    def visualize(
        self,
        folder_path: Union[str, pathlib.Path] = None,
        bar: bool = True,
        pie: bool = True,
    ) -> None:
        """This function visualizes the required operations per layer for the analyzed model.
        The figures will be written as png files to the given folder path

        Args:
            folder_path (Union[str, pathlib.Path], optional): Where the image files are supposed to be stored. Defaults to None.
            bar (bool, optional): if a bar chart should be created. Defaults to True.
            pie (bool, optional): if a bar chart should be created. Defaults to True.

        Returns:
            None (NoneType)
        """
        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)

        folder_path.mkdir(exist_ok=True, parents=True)

        keys = list(self.IFM_count.keys())
        values = list(self.IFM_count.values())

        if pie:
            fig, ax = plt.subplots()

            ax.pie(values, autopct="%1.1f%%")
            ax.axis("equal")
            ax.legend(keys)
            ax.set_title(
                "Distribution of IFM sizes across Model Layers", pad=20
            )
            fig.tight_layout()

            file_path = folder_path / "ifm_report_pie.svg"
            fig.savefig(file_path)

            file_path = folder_path / "ifm_report_pie.png"
            fig.savefig(file_path)

        if bar:
            fig, ax = plt.subplots()

            ax.bar(keys, values)
            ax.set_title(
                "Distribution of IFM sizes across Model Layers", pad=20
            )
            ax.set_xlabel("Layer")
            ax.set_ylabel("# elems")
            plt.xticks(rotation=90)
            fig.tight_layout()

            file_path = folder_path / "ifm_report_bar.svg"
            fig.savefig(file_path)

            file_path = folder_path / "ifm_report_bar.png"
            fig.savefig(file_path)

        return None

    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> None:
        """
        Dumps the relevant information from the MemoryAnalysis object as files to the given folder.
        One file is created by this method:\n
        - 'memory.json', contains information about the IFMs and a version string that describes the version of the class


        Args:
            folder_path (Union[str, pathlib.Path], optional): the destination folder for the dumped data. Defaults to None. If folder_path is None, the current working directory is used.
        """

        summary = {
            "IFM_dimensions": self.IFM_dimensions,
            "IFM_count": [(key, int(x)) for key, x in self.IFM_count.items()] ,
            "total_IFM": self.total_IFM,
            "version": self._version,
        }

        with open(folder_path / self.__json_file_name, "w") as file:
            json.dump(summary, file)

        return

    @classmethod
    def load(cls, folder_path: Union[str, pathlib.Path], analysis : BaseAnalysis = None) -> PartialAnalysis:
        """
        creates a new MemoryAnalysis instance from previously dumped data.
        Requires the path of the folder containing the needed 'memory.json' file as input.

        Args:
            folder_path (Union[str, pathlib.Path]): The folder that contains the dump of the MemoryAnalysis object
            analysis (_type_): The ModelAnalysis object that contains this MemoryAnalysis

        Raises:
            FileNotFoundError: raises an error, if the required file could not be found

        Returns:
            PartialAnalysis: Returns a MemoryAnalysis object
        """
        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        json_path = folder_path / cls.__json_file_name

        if not json_path.exists():
            raise FileNotFoundError("The required JSON file for loading the MemoryAnalysis object is missing.")

        with open(json_path, "r") as file:
            summary = json.load(file)

        memory_analysis = cls(analysis=analysis, deserialize=True)

        memory_analysis._version = summary["version"]
        memory_analysis.total_IFM = int(summary["total_IFM"])
        memory_analysis.IFM_count = dict(summary["IFM_count"])
        memory_analysis.IFM_dimensions = dict(summary["IFM_dimensions"])

        return memory_analysis

@dataclass
class StorageAnalysis(PartialAnalysis):
    """class that contains all analysis data regarding the storage allocation of the Neural Network"""
    __json_name = "storage.json"

    params : int
    """the amount of weights in the model"""

    layer_params : Dict[str, int]
    """the amount of weights per layer in the model"""

    def __init__(self, analysis, deserialize:bool=False):
        if analysis is not None:
            super().__init__(analysis)
        else:
            self.keras_model = None
            self.analysis = None
            self._version = "unknown"

        if not deserialize:
            self.params = res.get_model_weight_count(model=self.keras_model)
            self.layer_params = dict()
            for layer in self.keras_model.layers:
                self.layer_params[layer.name] = res.get_layer_weight_count(
                    layer=layer
                )
        else:
            self.params = None
            self.layer_params = None

    def estimate_model_size(self, dtype: str = "float") -> int:
        """Estimates the model size in storage, based on the weights.
        The estimate is based on the passed datatype.

        Args:
            dtype (str, optional): datatype that has been used to encode the weights (i.e. "int8", "float16", "float32", etc.). Defaults to "float".

        Returns:
            int: Model weight size in byte
        """
        if res._datatype_widths[dtype] < 8:
            log.info(
                f"{dtype} is {res._datatype_widths[dtype]} bit long, assumes dense packing for sub-byte types"
            )
        return np.ceil(self.params * res._datatype_widths[dtype] / 8)

    def estimate_layer_size(
        self, layer_name: str, dtype: str = "float"
    ) -> int:
        """Estimates the required storage space for the given layer's weights.
        The estimate is based on the passed datatype.

        Args:
            layer_name (str): name of the layer that will be analyzed
            dtype (str, optional): datatype that has been used to encode the weights (i.e. "int8", "float16", "float32", etc.). Defaults to "float".

        Returns:
            int: Layer weight size in byte
        """
        if res._datatype_widths[dtype] < 8:
            log.info(
                f"{dtype} is {res._datatype_widths[dtype]} bit long, assumes dense packing for sub-byte types"
            )
        return np.ceil(
            self.layer_params[layer_name] * res._datatype_widths[dtype] / 8
        )

    def __str__(self) -> str:
        return f"StorageAnalysis for {self.keras_model.name} - part of {self.analysis}: weight_count={self.params}"

    def __repr__(self) -> str:
        return f"StorageAnalysis(analysis={self.analysis})"

    def visualize(
        self,
        folder_path: Union[str, pathlib.Path] = None,
        bar: bool = True,
        pie: bool = True,
    ) -> None:
        """Visualizes the storage allocation per layer as charts

        Args:
            folder_path (Union[str, pathlib.Path], optional): folder path where the created figures will be stored as png files. Defaults to None.
            bar (bool, optional): if a bar chart should be drawn. Defaults to True.
            pie (bool, optional): if a pie chart should be drawn. Defaults to True.

        Returns:
            None
        """
        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)

        folder_path.mkdir(exist_ok=True, parents=True)

        keys = list(self.layer_params.keys())
        values = list(self.layer_params.values())

        if pie:
            fig, ax = plt.subplots()

            ax.pie(values, autopct="%1.1f%%")
            ax.axis("equal")
            ax.legend(keys)
            ax.set_title(
                "Distribution of Layer Weights across Model Layers", pad=20
            )
            fig.tight_layout()

            file_path = folder_path / "weights_report_pie.svg"
            fig.savefig(file_path)

            file_path = folder_path / "weights_report_pie.png"
            fig.savefig(file_path)

        if bar:
            fig, ax = plt.subplots()

            ax.bar(keys, values)
            ax.set_title(
                "Distribution of Layer Weights across Model Layers", pad=20
            )
            ax.set_xlabel("Layer")
            ax.set_ylabel("# elems")
            plt.xticks(rotation=90)
            fig.tight_layout()

            file_path = folder_path / "weights_report_bar.svg"
            fig.savefig(file_path)

            file_path = folder_path / "weights_report_bar.png"
            fig.savefig(file_path)

        return None

    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> None:
        """
        Dumps the information collected by the StorageAnalysis into the given folder.
        One file will be created by the method:\n
        - 'storage.json' contains information about the amount of
        parameters per layer and the version string of the used StorageAnalysis class

        Args:
            folder_path (Union[str, pathlib.Path], optional): The folder in which the data is going to be dumped. Defaults to None. Uses the current working directory if no folder_path is given.
        """

        summary = {
            "params": self.params,
            "layer_params": self.layer_params,
            "version": self._version,
        }

        with open(folder_path / self.__json_name, "w") as file:
            json.dump(summary, file)

        return

    @classmethod
    def load(cls, folder_path: Union[str, pathlib.Path], analysis : BaseAnalysis) -> PartialAnalysis:
        """
        Creates a new StorageAnalysis instance from the previously dumped information of another StorageAnalysis object

        Args:
            folder_path (Union[str, pathlib.Path]): The folder to which the data got dumped
            analysis (BaseAnalysis): The ModelAnalysis object to which the StorageAnalysis will be associated

        Returns:
            PartialAnalysis: The new StorageAnalysis instance
        """

        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        json_path = folder_path / cls.__json_name

        if not json_path.exists():
            raise FileNotFoundError("The required JSON file for loading the StorageAnalysis object is missing.")

        with open(json_path, "r") as file:
            summary = json.load(file)

        storage_analysis = cls(analysis=analysis, deserialize=True)

        storage_analysis._version = summary["version"]
        storage_analysis.params = summary["params"]
        storage_analysis.layer_params = summary["layer_params"]

        return storage_analysis


@dataclass
class ModelAnalysis(BaseAnalysis):
    """unified analysis class that contains all the information about the model as objects of Architecture, Compute, Memory, Storage or TaskAnalysis"""

    __json_name = "summary.json"
    __keras_name = "original.keras"

    keras_model : tf.keras.models.Model
    """the model that is analyzed"""

    name : str
    """the name of the current project, currently always fixed to the model name"""

    architecture : ArchitectureAnalysis
    """Contains all information of the architecture step of the analysis"""

    compute : ComputeAnalysis
    """Contains all information of the compute step of the analysis"""

    memory : MemoryAnalysis
    """Contains all information of the memory allocation analyzation step of the analysis"""

    storage : StorageAnalysis
    """Contains all information of the storage allocation analysis"""

    tasks : Dict[str, prd.Task]
    """A dict that contains the estimated task for each identified output"""

    modalities : Dict[str, prd.Modality]
    """A dict that contains the estimated input data modality for each network input"""

    reporters : Dict[str, callable]
    """A dict with functions that will be called if the associated reports need to be evaluated"""

    reports : Dict[str, object]
    """A dict with already evaluated reports, they are associated with the same identifier to enable reuse"""

    rewriter : Dict[str, base.Rewriter]
    """ The rewriting flows that are currently registered with the ModelAnalysis."""

    reporter_history: Dict[str, dict]
    """used to keep track of internal states of the reports and reporters"""

    creation_time : datetime
    """The exact time when the object was created, used as unique identifier"""

    cache_dir : pathlib.Path
    """the path on which temporary or intermediate data will be stored"""

    optimization_queue : RewriterQueue
    """The rewriters that will be applied during the next optimization loop."""

    def __init__(
        self,
        model: tf.keras.Model,
        name : str = None,
        mac_estimators: Dict[tf.keras.layers.Layer, callable] = None,
        cache_dir: Union[str, pathlib.Path] = None,
        deserialize:bool=False
    ) -> None:
        self.keras_model: tf.keras.Model = model
        if name is not None:
            self.name = name
        else:
            self.name: str = model.name

        if not deserialize:
            self.architecture: ArchitectureAnalysis = ArchitectureAnalysis(
                analysis=self
            )
            self.compute: ComputeAnalysis = ComputeAnalysis(
                analysis=self, estimation_functions=mac_estimators
            )
            self.memory: MemoryAnalysis = MemoryAnalysis(analysis=self)
            self.storage: StorageAnalysis = StorageAnalysis(analysis=self)
            self.tasks: Dict[str, prd.Task] = prd.recognize_task(model=model)
            self.modalities: Dict[str, prd.Modality] = prd.recognize_modality(
                model=model
            )

        self.reporters: Dict[str, callable] = dict()
        self.reports: Dict[str, object] = dict()
        self.reporter_history: Dict[str, dict] = dict()
        self.optimization_queue = RewriterQueue()

        self.rewriters = dict()
        #self.rewriter_identifiers = set()

        if not deserialize:

            self.creation_time: datetime = datetime.now()

            if cache_dir is None:
                if os.environ.get('CONVERSION_CACHE') is not None:
                    cache_dir = pathlib.Path(os.environ.get('CONVERSION_CACHE'))
                cache_dir = pathlib.Path("./.cache")

            if isinstance(cache_dir, str):
                cache_dir = pathlib.Path(cache_dir)

            self.cache_dir = cache_dir

            cache_dir.mkdir(parents=True, exist_ok=True)

        else:
            self.creation_time = None
            self.cache_dir = None

    def __str__(self) -> str:
        time_str = self.creation_time.strftime("%Y-%m-%d %H:%M:%S:%f")
        return f"ModelAnalysis for {self.keras_model.name} ({list(self.tasks.values())}), created at: {time_str}"

    def __repr__(self) -> str:
        return f"ModelAnalysis(model={repr(self.keras_model)}, mac_estimator={repr(self.compute.mac_estimators)}, cache_dir={repr(self.cache_dir)})"

    def _plot_model(self, folder_path: Union[str, pathlib.Path]) -> None:
        """creates a plot of the model architecture, requires the newest keras and TF version to work

        Args:
            folder_path (Union[str, pathlib.Path]): folder in which the figure should be stored
        """
        from keras.utils import plot_model

        plot_model(
            self.keras_model,
            to_file=folder_path / "model.png",
            show_shapes=True,
        )

    def submit_rewriter(self, rewriter : base.Rewriter) -> base.Rewriter:
        identifier = rewriter.create_identifier()

        if identifier in self.rewriters.keys():
            log.warn("rewriter with same configuration has already been submitted")
            return self.rewriters[identifier]

        self.rewriters[identifier] = rewriter
        return rewriter

    def submit_reporter(self, unique_identifier: str, reporter_constructor: callable) -> None:
        """Adds a new reporter to the analysis

        Args:
            unique_identifier (string): a unique ID that will be used to find this specific reporter and its created Report again
            reporter_constructor (callable): closure / factory or constructor of the reporter object
        """

        # prevents duplicated work
        if unique_identifier not in self.reporters.keys():
            self.reporters[unique_identifier] = reporter_constructor
            self.reporter_history[unique_identifier] = {
                "submit": time.time(),
                "created": None,
            }
            return True
        return False

    def create_reports(self) -> None:
        """Creates all the reports that have been submitted.
        If the create_reports function is called multiple times, only the reports that have not previously created will be added.
        Already added reports will stay untouched!
        """
        for rep_id, reporter in self.reporters.items(): # will this create problems, if we create reports that submit new reporters?
            if self.reporter_history[rep_id]["created"] is None:
                self._create_report(rep_id, reporter)
            else:
                report = self.reporter_history[rep_id]["report"]
                log.info(
                    f"Report {report} has already been created, will not be recomputed"
                )

    def _create_report(self, unqiue_id : str, report_constructor : callable):
        """creates single report and adds it to the analysis objects

        Args:
            unqiue_id (str): a unique ID that will be used to find this specific reporter and its created Report again
            report_constructor (callable): closure / factory or constructor of the reporter object
        """
        report = report_constructor(self)
        self.reports[unqiue_id] = report
        self.reporter_history[unqiue_id]["created"] = time.time()
        self.reporter_history[unqiue_id]["report"] = report

        return report
        

    def access_report(self, report_id) -> object:
        """Access a specific model report based on its class

        Args:
            report_id (str): The unqiue ID of the report instance that should be returned

        Returns:
            object: ModelReport
        """

        # TODO: need to enable rewriters to access their specific report
        if report_id in self.reports.keys():
            return self.reports[report_id]
        else:
            log.info("report for {report_id} not found in {self}, checking in not-initialized reports!")
            if report_id in self.reporters.keys():
                log.info("found fitting reporter, starting initialization!")
                self._create_report(report_id, self.reporters[report_id])
                return self.reports[report_id]

            log.error("report for {report_id} not found in {self}, no fitting reporter was found!")
            return None

    def visualize(self, folder_path: Union[str, pathlib.Path]) -> None:
        """Visualization function that calls the visualization function of all subanalysis

        Args:
            folder_path (Union[str, pathlib.Path]): folder, where the created figures should be stored
        """
        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)

        folder_path = pathlib.Path(folder_path) / "figures/"
        folder_path.mkdir(exist_ok=True, parents=True)

        # plot keras model using Keras builtins
        self._plot_model(folder_path=folder_path)

        # plot subanalysis
        self.architecture.visualize(folder_path=folder_path)

        self.compute.visualize(folder_path=folder_path)
        self.memory.visualize(folder_path=folder_path)
        self.storage.visualize(folder_path=folder_path)

    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> None:
        """Function that dumps the collected analysis data in a machine-readable format

        Args:
            folder_path (Union[str, pathlib.Path], optional): Folder, where the HTML and its accompanying files should be stored. Defaults to None.
        """

        time_str = self.creation_time.strftime("%Y-%m-%d_%H-%M-%S-%f")
        suffix = f"{self.name}/{time_str}/"

        if folder_path is None:
            folder_path = pathlib.Path.cwd() / suffix

        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        #folder_path = pathlib.Path(folder_path) / suffix
        folder_path.mkdir(exist_ok=True, parents=True)

        # dump analysis
        self.architecture.dump(folder_path=folder_path)
        self.compute.dump(folder_path=folder_path)
        self.memory.dump(folder_path=folder_path)
        self.storage.dump(folder_path=folder_path)

        summary = {
            "name" : self.name,
            "tasks": [(key, str(x)) for key, x in self.tasks.items()],
            "created": time_str,
            "version": self._version,
            }

        # TODO: fix
        with open(folder_path / self.__json_name, "w") as file:
            json.dump(summary, file)

        # store model
        self.keras_model.save(folder_path / self.__keras_name)

        # dump reports
        for report in self.reports.values():
            report.dump(folder_path=folder_path)

        # dump rewriters (TODO)

        return None

    @classmethod
    def load(cls, folder_path: Union[str, pathlib.Path]) -> BaseAnalysis:

        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        keras_path = folder_path / cls.__keras_name
        json_path = folder_path / cls.__json_name

        if not keras_path.exists():
            raise FileNotFoundError("The original model that was the base for the ModelAnalysis could not be found")

        if not json_path.exists():
            raise FileNotFoundError("The ModelAnalysis summary could not be found in the given directory")

        #load data

        ## keras model
        keras_model = tf.keras.models.load_model(keras_path)

        ## summary data
        with open(json_path, "r") as file:
            summary = json.load(file)

        # create new ModelAnalysis object
        model_analysis = cls(model=keras_model, deserialize=True)

        if "name" in summary.keys():
            model_analysis.name = summary["name"]
        if "version" in summary.keys():
            model_analysis._version = summary["version"]

        model_analysis.creation_time = datetime.strptime(summary["created"], "%Y-%m-%d_%H-%M-%S-%f")

        #model_analysis.tasks = dict(summary["tasks"])

        # create PartialAnalysis objects, associate them with ModelAnalysis
        model_analysis.architecture = ArchitectureAnalysis.load(folder_path=folder_path, analysis=model_analysis)
        model_analysis.compute = ComputeAnalysis.load(folder_path=folder_path, analysis=model_analysis)
        model_analysis.memory = MemoryAnalysis.load(folder_path=folder_path, analysis=model_analysis)
        model_analysis.storage = StorageAnalysis.load(folder_path=folder_path, analysis=model_analysis)

        model_analysis.tasks = prd.recognize_task(model=keras_model)
        model_analysis.modalities = prd.recognize_modality(model=keras_model)

        # TODO: load reports
        log.warn("ModelAnalysis reports could not be restored while loading instance from dump")

        # TODO: load rewriter
        log.warn("ModelAnalysis rewriters could not be restored while loading instance from dump")

        # return newly created ModelAnalysis instance
        return model_analysis

    def summary(self, folder_path: Union[str, pathlib.Path] = None) -> None:
        """Function that creates an interactive HTML-based summary of the analysis

        Args:
            folder_path (Union[str, pathlib.Path], optional): Folder, where the HTML and its accompanying files should be stored. Defaults to None.
        """
        _template_path = pathlib.Path(os.path.dirname(__file__)) / 'templates'

        time_str = self.creation_time.strftime("%Y-%m-%d_%H-%M-%S-%f")
        suffix = f"{self.name.replace(':','_')}/{time_str}/"

        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)

        folder_path = pathlib.Path(folder_path) / suffix
        folder_path.mkdir(exist_ok=True, parents=True)

        self.visualize(folder_path=folder_path)

        # copy styling content to folder
        source_folder = _template_path / "styling" #pathlib.Path('templates/styling')
        shutil.copytree(source_folder, folder_path, dirs_exist_ok=True)

        self.create_reports()
        report_paths = dict()
        for report in self.reports.values():
            if report.hide:
                continue
            key, val = report.render_summary(folder_path=folder_path)
            report_paths[key] = str(val)

        rewriter_paths = dict()
        for rew_id, (rew_identifier, rewriter) in enumerate(self.rewriters.items()):
            key, val = rewriter.render_summary(folder_path=folder_path)
            rewriter_paths[f"{rew_id}: {key}"] = str(val)

        self.dump(folder_path=folder_path)

        with open(_template_path / "analysis.html", "r") as file:
            template = Template(file.read())

        block_node_data = [
            {"id": str(node), "label": str(node)}
            for node in self.architecture.block_graph.nodes()
        ]
        block_edge_data = [
            {"from": str(edge[0]), "to": str(edge[1])}
            for edge in self.architecture.block_graph.edges()
        ]
        block_graph_data = {"nodes": block_node_data, "edges": block_edge_data}

        layer_node_data = [
            {"id": str(node), "label": str(node)}
            for node in self.architecture.network_graph.nodes()
        ]
        layer_edge_data = [
            {"from": str(edge[0]), "to": str(edge[1])}
            for edge in self.architecture.network_graph.edges()
        ]
        layer_graph_data = {"nodes": layer_node_data, "edges": layer_edge_data}

        keras_version = "UNKNOWN"

        try:
            keras_version = keras.__version__
        except:
            log.info("could not determine Keras version")

        summary = {
            "name": self.name,
            "creation_date": self.creation_time,
            "task": self.tasks,
            "tf_version": tf.__version__,
            "keras_version": keras_version,
            "_version": self._version,
            "macs": self.compute.layer_mac,
            "total_mac": self.compute.total_mac,
            "ifms": self.memory.IFM_count,
            "total_ifms": self.memory.total_IFM,
            "weights": self.storage.layer_params,
            "total_weights": self.storage.params,
            "feed_forward": self.architecture.is_feed_forward,
            "recurrent": self.architecture.is_recurrent,
            "branching": self.architecture.is_branching,
            "inputs": self.architecture.inputs,
            "outputs": self.architecture.outputs,
            "block_graph": block_graph_data,
            "layer_graph": layer_graph_data,
            "reports": report_paths,
            "rewriters": rewriter_paths,
        }

        # Render the template with the summary data
        html = template.render(summary=summary)

        # Save the generated HTML to a file
        html_path = folder_path / "summary.html"

        with open(html_path, "w") as file:
            file.write(html)

        return None
