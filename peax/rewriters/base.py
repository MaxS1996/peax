from abc import ABC, abstractmethod
from typing import Union
import pathlib

from typing import Union, List, Tuple

import tensorflow as tf


class Solution(ABC):
    """
    Wraps the solutions created by the Rewriters into helpful objects.
    """

    @abstractmethod
    def evaluate(self):
        """
        Evaluates the found solution. Way differs based on used optimization procedure.
        """
        pass

    @abstractmethod
    def finetune(self):
        """
        Optional finetuning step that can be applied to some solutions.
        """
        pass

    @abstractmethod
    def toDict(self):
        """
        Converts solution into a dict.
        """
        pass

    @abstractmethod
    def dump(self, path : Union[str, pathlib.Path]):
        """
        Writes the solution to disk.

        Args:
            path (Union[str, pathlib.Path]): the folder in which the solution is going to be stored.
        """
        pass

    @abstractmethod
    def to_analysis(self) -> List[object]:
        """
        Converts the solution into a new ModelAnalysis to proceed with the next analysis/optimization process.
        """
        pass

class Rewriter(ABC):
    """
    The rewriter class is intended to wrap a optimization procedure.
    It is supposed to contain references to all relevant reports and also additional information as well as
    the implementation of the necessary steps of the procedure.
    """
    
    def __init__(self, analysis) -> None:
        super().__init__()
        self.analysis = analysis
        analysis.submit_rewriter(self)

    @abstractmethod
    def create_identifier(self) -> str:
        pass
    
    '''@abstractmethod
    def compile_mapping() -> tf.keras.Model:
        pass'''
    
    @abstractmethod
    def dump(self, folder_path: Union[str, pathlib.Path] = None):
        pass

    @abstractmethod
    def render_summary(self, folder_path: Union[str, pathlib.Path] = None):
        pass

    @classmethod
    @abstractmethod
    def create_pass(cls, **params) -> Tuple[str, callable]:
        pass