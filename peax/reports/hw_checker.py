import itertools
import pickle
from tensorflow import keras as keras
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import hashlib
import glob

import json
import os

import pandas as pd

from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Set
from typing_extensions import Self
import pathlib
from datetime import datetime
import copy
import networkx as nx
import logging as log
from jinja2 import Template

# from components import visualize_graph as vis
from ..components import architecture as arch
from ..components import resource as res
from ..components import graph_tools as gt
from ..hardware import processor as pr

from . import base

import peax.analysis as aNN

class HWReportSubmitter(base.ReportSubmitter):
    """
    Auxilliary class to simplify the submission of reports to the ModelAnalysis
    """
    def __init__(self, analysis: aNN.ModelAnalysis, lazy: bool = False):
        """
        the constructor for the submitter class

        Args:
            analysis (aNN.ModelAnalysis): the ModelAnalysis instance to which the reporter will be assigned
            lazy (bool, optional): The submission behavior. lazy will only create the report if it is required, otherwise it will be generated immediately. Defaults to False.
        """
        super().__init__(analysis, lazy)

    def with_config(self, processors: Set[pr.Processor], dtypes: Set[str] = ["int8", "float32"]) -> Union[base.Report, str]:
        """
        Creates a HWReport with the given config and associates it with the analysis.
        If lazy, the unqiue ID of the report will be returned and the report itself will not yet be generated.
        If lazy is False, the report will be returned.

        Args:
            processors (Set[pr.Processor]): The processing targets whose performance and support should be estimated and evaluated.
            dtypes (Set[str], optional): The compute datatypes that should be considered. Defaults to ["int8", "float32"].

        Returns:
            Union[base.Report, str]: Either the HWReport or the unique ID of the Report that can be created through the ModelAnalysis instance.
        """
        hw_reporter, hw_rep_id = HWReport.closure(processors=processors, dtypes=dtypes, create_id=True)

        self.analysis.submit_reporter(hw_rep_id, hw_reporter)
        if self.lazy:
            return hw_rep_id
        
        return self.analysis.access_report(hw_rep_id)


class HWReport(base.Report):
    """Report that analysis the cost of running the model on the given processors as well as if the individual layers can be executed by them."""

    __pkl_name = "report_hw_estimates_<hash>.pkl"

    dtypes: List[str]
    """Datatypes that will be considered during the evaluation, currently only in terms of memory utilization."""

    processors: Set[pr.Processor]
    """Descriptions of the processors that will be evaluated."""

    supported: Dict[pr.Processor, Dict[gt.Node, bool]]
    """Stores which layers can be executed by each processor."""

    latency: Dict[pr.Processor, Dict[gt.Node, float]]
    """Stores the estimated compute latency of the individual layer workloads for each target processor."""

    memory_util: Dict[pr.Processor, Dict[gt.Node, Dict[str, float]]]
    """The relative memory utilization of the individual output feature maps of the layers for the local memory of each processor."""

    macs: Dict[pr.Processor, Dict[gt.Node, int]]
    """The MACs required by each processor for each layer, allows us to account for different execution strategies that might be implemented by different hardware architectures."""

    def __init__(
        self,
        analysis: aNN.ModelAnalysis,
        processors: Set[pr.Processor],
        dtypes: List[str] = ["int8", "float32"],
    ) -> None:
        """
        The constructor for the HWReport, not recommended to be used

        Args:
            analysis (aNN.ModelAnalysis): the analysis to which the report will be submitted
            processors (Set[pr.Processor]): Descriptions of the processors that will be evaluated.
            dtypes (List[str], optional): Datatypes that will be considered during the evaluation, currently only in terms of memory utilization. Defaults to ["int8", "float32"].
        """
        super().__init__(analysis)

        self.dtypes = dtypes
        self.processors = processors
        (
            self.supported,
            self.latency,
            self.memory_util,
            self.macs,
        ) = self.check_support()

    def check_support(
        self,
    ) -> Tuple[
        Dict[pr.Processor, Dict[gt.Node, bool]],
        Dict[pr.Processor, Dict[gt.Node, float]],
        Dict[pr.Processor, Dict[gt.Node, Dict[str, float]]],
        Dict[pr.Processor, Dict[gt.Node, int]],
    ]:
        """Creates the cost estimates of the model for the given Processors

        Returns:
            Dict[pr.Processor, Dict[gt.Node, bool]] : Dict of bools, True if the given processor can execute the given layer configuration
            Dict[pr.Processor, Dict[gt.Node, float]]:Dict of floats that are the estimated latency estimate for the tuple of processor and layer configuration
            Dict[pr.Processor, Dict[gt.Node, Dict[str, float]]]]: Dict of floats that describe the share of memory allocated for the layer weights and IFM
            Dict[pr.Processor, Dict[gt.Node, int]] : Dict of the required MAC operations per processor and Layer
        """
        recommendations = dict()
        delays = dict()
        memory_utils = dict()
        macs = dict()
        for proc in self.processors:
            # print(f"\t{proc.name}")
            recommendations[proc] = dict()
            delays[proc] = dict()
            memory_utils[proc] = dict()
            macs[proc] = dict()

            for node in self._graph.nodes:
                recom, delay, memory, l_macs = proc.check(
                    node.keras, dtypes=self.dtypes
                )
                recommendations[proc][node] = recom
                delays[proc][node] = delay
                memory_utils[proc][node] = memory
                macs[proc][node] = l_macs

        return recommendations, delays, memory_utils, macs

    def check_block_support(
        self, block: gt.BlockNode
    ) -> Tuple[
        Dict[pr.Processor, bool],
        Dict[pr.Processor, float],
        Dict[pr.Processor, Dict[str, float]],
        Dict[pr.Processor, int],
    ]:
        """function to access performance data for node in block graph representations

        Args:
            block (gt.BlockNode): the given block

        Returns:
            Dict[pr.Processor, bool]: processors as keys, values describe key's ability to execute the layer as bool
            Dict[pr.Processor, float]: processors as keys, values describe key's latency
            Dict[pr.Processor, Dict[str, float]]]: processors as keys, values are dicts that describe the % of memory allocation depending on used dtype
            Dict[pr.Processor, int]: processors as keys, values describe the amount of MAC operations required per key to process the task
        """
        total_recom = {}
        total_delay = {}
        total_mem = {}
        total_macs = {}

        for proc in self.processors:
            recom = True
            delay = 0
            b_macs = 0
            mem_util = dict()
            for dtype in self.dtypes:
                mem_util[dtype] = 0

            for node in block.subgraph.nodes:
                if node.layer_class == "optimization_dummy":
                    self.supported[proc][node] = True
                    self.latency[proc][node] = 0
                    
                    self.memory_util[proc][node] = dict()
                    for dtype in self.dtypes:
                        self.memory_util[proc][node][dtype] = 0

                    self.macs[proc][node] = 0
                if not node in self.supported[proc]:
                    new_recom, new_delay, new_memory, macs = proc.check(
                        node.keras, dtypes=self.dtypes
                    )
                    self.supported[proc][node] = new_recom
                    self.latency[proc][node] = new_delay
                    self.memory_util[proc][node] = new_memory
                    self.macs[proc][node] = macs

                recom &= self.supported[proc][node]
                delay += self.latency[proc][node]
                b_macs += self.macs[proc][node]

                for dtype in self.dtypes:
                    mem_util[dtype] += self.memory_util[proc][node][dtype]

            total_recom[proc] = recom
            total_delay[proc] = delay
            total_mem[proc] = mem_util
            total_macs[proc] = b_macs

        return total_recom, total_delay, total_mem, total_macs

    def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> Tuple[str, str]:
        """Creates the HTML file for the summary overview.
        Returns the title and relative path of the report.

        Args:
            folder_path (Union[str, pathlib.Path], optional): folder, in which the file and auxiliary data will be stored. Defaults to None.
        """
        _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'

        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        file_name = f"report_hw-check_{self.access_id()}.html"

        with open(_template_path / "hw_check_report.html", "r") as file:
            template = Template(file.read())

        support_matrix = dict()

        for key in self.supported.keys():
            support_matrix[key] = (
                self.supported[key],
                self.latency[key],
                self.memory_util[key],
            )

        summary = {
            "report_type": "HW Check",
            "name": self.analysis.name,
            "creation_date": self.analysis.creation_time,
            "support_matrix": support_matrix,
            "processors": self.processors,
        }

        # Render the template with the summary data
        html = template.render(summary=summary)
        # Save the generated HTML to a file
        html_path = folder_path / file_name
        with open(html_path, "w") as file:
            file.write(html)

        return "HW Support Check and Performance Estimations", file_name
    
    def access_id(self) -> str:
        """
        Returns the unique identifer of the HWReport instance

        Returns:
            str: the identifier
        """

        return self.create_unique_id(
            processors=self.processors,
            dtypes=self.dtypes,
            name=self.analysis.name,
        )

    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> None:
        """
        Stores the contained information to disk.
        WIP!

        Args:
            folder_path (Union[str, pathlib.Path], optional): _description_. Defaults to None.
        """
        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        summary = dict()

        summary["dtypes"] = self.dtypes
        summary["processors"] = self.processors

        summary["latency"] = [(proc.name, [(layer.name, info) for layer, info in latency_info.items()]) for proc, latency_info in self.latency.items()]
        summary["memory_util"] = [(proc.name, [(layer.name, info) for layer, info in mem.items()]) for proc, mem in self.memory_util.items()] #self.memory_util
        summary["macs"] = [(proc.name, [(layer.name, info) for layer, info in macs.items()]) for proc, macs in self.macs.items()]
        summary["is_supported"] = [(proc.name, [(layer.name, info) for layer, info in sup.items()]) for proc, sup in self.supported.items()] #self.supported

        file_name = self.__pkl_name.replace("<hash>", self.access_id())
        with open(folder_path / file_name, "wb") as file:
            pickle.dump(summary, file)

    @classmethod
    def load(cls, folder_path: Union[str, pathlib.Path], analysis : aNN.ModelAnalysis) -> List[Self]:
        """
        loads a new HWReport instance from a previous dump.

        Args:
            folder_path (Union[str, pathlib.Path]): The folder being searched for HWReport dumps
            analysis (aNN.ModelAnalysis): the ModelAnalysis to which the reports will be assigned

        Returns:
            List[Self]: A list of HWReports
        """
        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        file_pattern = cls.__pkl_name.replace("<hash>", "*")
        files = glob.glob(str(folder_path) + "/" + file_pattern)

        if len(files) == 0:
            raise FileNotFoundError("no HWReport has been found in {folder_path}")

        reports = []
        for file_path in files:
            file_path = pathlib.Path(file_path)
            with open(file_path, "rb") as file:
                summary = pickle.load(file)

            new_report = HWReport(analysis=analysis,
                                  processors=[],
                                  dtypes=[])
            
            new_report.dtypes = summary["dtypes"]
            new_report.processors = summary["processors"]

            layer_nodes = list(analysis.architecture.network_graph.nodes)

            new_report.latency = {} #summary["latency"] # unpack latency info
            for info in summary["latency"]:
                proc_name, lat_data = info

                proc = [obj for obj in new_report.processors if obj.name == proc_name][0]
                new_report.latency[proc] = {}

                for details in lat_data:
                    layer_name, lat = details

                    layer = [obj for obj in layer_nodes if obj.name == layer_name][0]
                    new_report.latency[proc][layer] = lat


            new_report.memory_util = {} # summary["memory_util"] # unpack memory utilization info
            for info in summary["memory_util"]:
                proc_name, mem_data = info

                proc = [obj for obj in new_report.processors if obj.name == proc_name][0]
                new_report.memory_util[proc] = {}

                for details in mem_data:
                    layer_name, mem = details

                    layer = [obj for obj in layer_nodes if obj.name == layer_name][0]
                    new_report.memory_util[proc][layer] = mem

            new_report.supported = {} # summary["is_supported"] # unpack support info
            for info in summary["is_supported"]:
                proc_name, sup_data = info

                proc = [obj for obj in new_report.processors if obj.name == proc_name][0]
                new_report.supported[proc] = {}

                for details in sup_data:
                    layer_name, sup = details

                    layer = [obj for obj in layer_nodes if obj.name == layer_name][0]
                    new_report.supported[proc][layer] = sup

            new_report.macs = {} # summary["macs"] # unpack macs info
            for info in summary["macs"]:
                proc_name, macs_data = info

                proc = [obj for obj in new_report.processors if obj.name == proc_name][0]
                new_report.macs[proc] = {}

                for details in macs_data:
                    layer_name, macs = details

                    layer = [obj for obj in layer_nodes if obj.name == layer_name][0]
                    new_report.macs[proc][layer] = macs

            reports.append(new_report)

        return reports
    
    @classmethod
    def create_unique_id(cls, processors : List[pr.Processor], dtypes : List[str], name : str) -> str:
        """
        Function to create a unique ID for instances of the HWReport class.
        This can be used to create a unqiue ID for instances that have not yet been instantiated

        Args:
            processors (List[pr.Processor]): The processors being used for the report
            dtypes (List[str]): the considered datatypes
            name (str): the name of the ModelAnalysis

        Returns:
            str: a unique ID
        """

        proc_str = '-'.join([str(proc) for proc in processors])
        dt_str = '-'.join([dt for dt in dtypes])
        descr_str = f"HWReport-{proc_str}-{dt_str}-{name}"

        hashed_str = cls.create_reporter_id(descr_str)

        return hashed_str

    @classmethod
    def submit_to(cls, analysis:aNN.ModelAnalysis, lazy:bool=False) -> HWReportSubmitter:
        """
        Syntactic sugar to simplify submission of new HWReports to the ModelAnalysis
        i.e.: HWReport.submit_to(analysis).with_config(...)

        Args:
            analysis (aNN.ModelAnalysis): the analysis to which the reporter should be submitted
            lazy (bool, optional): The submission behavior:
                lazy will only create the report if it is required, otherwise it will be generated immediately.
                Defaults to False.

        Returns:
            HWReportSubmitter: The submitter auxiliary object that is used to provide the necessary syntax
        """
        return HWReportSubmitter(analysis=analysis, lazy=lazy)

    @classmethod
    def closure(
        cls,
        processors: Set[pr.Processor], dtypes: Set[str] = ["int8", "float32"],
        create_id:bool=True
    ):
        """Closure that should be passed to the ModelAnalysis object"""

        def builder(analysis: aNN.ModelAnalysis):
            return HWReport(
                analysis=analysis, processors=processors, dtypes=dtypes
            )
        
        if create_id:
            descr_str = f"HWReport:{[proc.name for proc in processors]}-{[dt for dt in dtypes]}"
            hashed_str = cls.create_reporter_id(descr_str)

            return builder, hashed_str
        
        return builder
