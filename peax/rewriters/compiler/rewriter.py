from pathlib import Path
import pathlib
from typing import Tuple, List, Union, Dict
import hashlib

import numpy as np
import tensorflow as tf
from jinja2 import Template

import copy
import json
import os

from peax.components import graph_tools as gt

import peax.analysis as aNN
from .. import base

from peax.components import resource as res

class CompilerSolution(base.Solution):
  """
  Abstract Solutiomn class to establish a base class for the solutions created by CompilerInterfaces.
  """

  rewriter : base.Rewriter
  """The CompilerInterface that created this solution. """

  model : tf.keras.models.Model
  """The model that has been deployed for this solution through the CompilerInterface. """

  name : str
  """ The name of this solution. """

  config : Dict[str, object]
  """The compiler configuration that was used to create this solution."""

  def __init__(self, rewriter: base.Rewriter, model:tf.keras.models.Model, name:str, config:Dict[str, object]):
    """
    The constructor for the CompilerSolution base class.

    Args:
        rewriter (base.Rewriter): The rewriter that created the solution object.
        model (tf.keras.models.Model): A reference to the compiled model.
        name (str): A name for the created solution, can be used as unique identifer.
        config (Dict[str, object]): The compiler configuration that was used to create this solution.
    """
    super().__init__()
    
    self.model = model
    self.name = name
    self.config = config

    self.rewriter = rewriter

  def evaluate(self) -> Dict[str, object]:
    """
    This function can be implemented for CompilerSolutions to evaluate their performance, it is, however, not necessary to implement it.

    Raises:
        NotImplemented: Will raise an Exception, if the CompilerSolution object has not implemented an evaluation routine.

    Returns:
        Dict[str, object]: A dict of the evaluated metrics, the metric names will be used as keys in the dict.
    """
    raise NotImplemented("This function has not been implemented for the CompilerSolution base class")
    return
  
  def finetune(self) -> Dict[str, object]:
    """
    Finetuning of compiled solutions will not be able for most compiler toolchains, this function can however be used to support auto-tuning functionality as it is available in TVM.
    The current microTVM CompilerInterface does not support this as it relies on CMSIS-NN function kernels to execute the layer operations.

    The function, if implemented, should return the updated evaluation results after finetuning the solution.

    Raises:
        NotImplemented: Will raise an Exception, if the CompilerSolution object has not implemented a finetune routine.

    Returns:
        Dict[str, object]: A dict of the evaluated metrics, the metric names will be used as keys in the dict, and will most likely change during finetuning.
    """
    raise NotImplemented("This function has not been implemented for the CompilerSolution base class")
    return self.evaluate()
  
  def toDict(self, contain_objects:bool=False) -> Dict[str, object]:
    """
    Converts the Solution object into a dictionary representation that can be used for serialization or the creation of the summaries through Jinja templates.

    Args:
        contain_objects (bool, optional): If True, objects that cannot necessarily be serialized through JSON, will be added to the Dict. Defaults to False.

    Returns:
        Dict[str, object]: the parameters of the solution.
    """
    data = {
      "name" : self.name,
      "config" : self.config
    }

    if contain_objects:
      data["model"] = self.model
    return data
  
  def dump(self, path: Union[str,Path]) -> pathlib.Path:
    """
    Writes the solution to disk.

    Args:
        path (Union[str,Path]): The folder path, to which the solution should be written.

    Raises:
        NotImplemented: raises this exception, if the solution does not implement this functionality.

    Returns:
        pathlib.Path: The path of the file, to which the dump has been written.
    """
    raise NotImplemented("This function has not been implemented for the CompilerSolution base class")
    return
  
  def to_analysis(self) -> List[object]:
    """
    CompilerSolutions will not implement this function.

    Raises:
        NotImplemented: Will always raise a NontImplemented exception, as it is a part of the base.Solution class, from which the CompilerSolutions inhert.
    """
    raise NotImplemented("This function has not been implemented for the CompilerSolution base class")
    return []
  
class CompilerInterface(base.Rewriter):
  """
  Abstract Rewriter class to establish base for compiler interfaces.
  """

  def __init__(self, analysis : aNN.ModelAnalysis):
    """
    The constructor for the CompilerInterface

    Args:
        analysis (aNN.ModelAnalysis): The ModelAnalysis object, to which the rewriter will be submitted.
    """
    super().__init__(analysis)

  def evaluate(self):
    """
    This function can be used by the implemented CompilerInterfaces, to evaluate if and how the used compiler can process the given model.

    Raises:
        NotImplemented: raises an exception, if the CompilerInterface does not implement it.
    """
    raise NotImplemented("This function has not been implemented for the base CompilerInterface")
  
  def configure(self):
    """
    This function can be used to setup components of the CompilerInterface.

    Raises:
        NotImplemented: raises an exception, if the CompilerInterface does not implement it.
    """
    raise NotImplemented("This function has not been implemented for the base CompilerInterface")
  
  def compile(self, **compile_config) -> CompilerSolution:
    """
    This function is supposed to create the compiled models as CompilerSolution objects.
    Each CompilerInterface implementation should come with its own implementation of the CompilerSolution.

    Returns:
        CompilerSolution: The compiled solution created by the interfaced compiler.
    """
    raise NotImplemented("This function has not been implemented for the base CompilerInterface")
  
  def create_identifier(self) -> str:
    """
    The CompilerInterface objects should also provide a unique identifer, similar to other rewriters.

    Returns:
        str: The unique identifier.
    """
    return "CompilerInterface"
  
  def dump(self, folder_path : Union[str, pathlib.Path]) -> Dict[str, object]:
    """
    Dumps the CompilerInterface to disk and returns a dict representation of it.

    Args:
        folder_path (Union[str, pathlib.Path]): The folder to which the CompilerInterface object should be dumped.

    Returns:
        Dict[str, object]: A dict representation of the CompilerInterface object.
    """
    raise NotImplemented("This function has not been implemented for the base CompilerInterface")
  
  def render_summary(self, folder_path : Union[str, pathlib.Path]) -> Tuple[str, str]:
    """
    Creates the summary HTML file for the current CompilerInterface instance.

    Args:
        folder_path (Union[str, pathlib.Path]): The path of the folder, to which the summary should be written.

    Returns:
        Tuple[str, str]: Returns a title and the name of the created file.
    """
    raise NotImplemented("This function has not been implemented for the base CompilerInterface")
  
  def create_pass(self) -> Tuple[str, callable]:
    """
    A closure that wraps the necessary steps to create a Solution using the CompilerInterface.
    This function should return a closure function that takes the ModelAnalysis object as an input and this closure 
    should submit a new rewriter to the analysis and execute the necessary functions to create the solution.

    Returns:
        Tuple[str, callable]: Returns the unique ID of the rewriter as well as the closure to execute the pass.
    """
    raise NotImplemented("This function has not been implemented for the base CompilerInterface")