import pathlib
from typing import Union, Dict
import json
from typing import Set
from typing_extensions import Self
from abc import ABC, abstractmethod

import networkx as nx
from jinja2 import Template
import hashlib

import peax.analysis as aNN

class ReportSubmitter():

    def __init__(self, analysis: aNN.ModelAnalysis, lazy:bool=False):
        self.lazy = lazy
        self.analysis = analysis

    def with_config(self, **args):
        return None

class Report(ABC):
    analysis: aNN.ModelAnalysis
    hide: bool
    _graph: nx.DiGraph
    _block_graph: nx.DiGraph

    def __init__(self, analysis: aNN.ModelAnalysis) -> None:
        self.analysis = analysis
        self.hide = False
        self._graph = self.analysis.architecture.network_graph
        self._block_graph = self.analysis.architecture.block_graph
        pass

    @staticmethod
    def create_reporter_id(description_string : str) -> str:
        sha256_hash = hashlib.sha256()
        
        # Update the hash object with the input string
        sha256_hash.update(description_string.encode('utf-8'))
        hashed_representation = sha256_hash.hexdigest()

        return hashed_representation
    
    @classmethod
    def submit_to(analysis : aNN.ModelAnalysis, lazy:bool=False) -> ReportSubmitter:
        """feature to simplify the creation of reports by users.
        Creates a ReportSubmitter object that registers the reporter callable with the analysis.
        Basically just syntactic sugaring
        i.e. EarlyExitReport.submit_to(model_analysis).with_config(search_config="large")

        Args:
            analysis (aNN.ModelAnalysis): the ModelAnalysis to which the reporter should be submitted
            lazy (bool, optional): The submission behavior. lazy will only create the report if it is required, otherwise it will be generated immediately.
            Defaults to False.

        Returns:
            ReportSubmitter: returns a helper object which creates the reporter callable, submits it to the ModelAnalysis and returns the Report, if specified as non_lazy, otherwise just the reference to call its constructor will be returned.
        """
        return ReportSubmitter(analysis=analysis, lazy=lazy)

    @abstractmethod
    def render_summary(self, folder_path: Union[str, pathlib.Path] = None):
        """Renders a HTML-based summary of the report. Used for interfacing with the human user

        Args:
            folder_path (Union[str, pathlib.Path], optional): the path where the summary will be stored. Defaults to None.

        Returns:
            _type_: _description_
        """
        return "base report", ""

    @abstractmethod
    def dump(self, folder_path: Union[str, pathlib.Path] = None):
        """dumps the report two the specified folder, might use different formats
        JSON is preferred, but can also rely on alternative formats if required

        Args:
            folder_path (Union[str, pathlib.Path], optional): the path where the serialization shall be dumped. Defaults to None.

        Returns:
            _type_: _description_
        """
        return None
    
    @classmethod
    @abstractmethod
    def load(cls, folder_path : Union[str, pathlib.Path], analysis : aNN.ModelAnalysis) -> Union[Self, Set[Self]]:
        """
        Loads the dumped data to restore previously written reports from it.

        Args:
            folder_path (Union[str, pathlib.Path]): The folder that is searched for serialized reports of that type
            analysis (aNN.ModelAnalysis): The ModelAnalysis object to which the restored report will be assigned

        Returns:
            Union[Self, Set[Self]]: returns the report or a set of reports that were found within the given folder
        """

        return None
