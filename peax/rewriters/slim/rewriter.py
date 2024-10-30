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

#from . import layers as slim_layers
from . import weight_handling as weight

from ...reports import accuracy as acc
from ...reports import dataset as d

from peax.components import resource as res

from tensorflow.keras import backend as K
from peax.utils.keras_graph import exchange_layer

class SlimmableSolution(base.Solution):
  """
  A slimmed version of the neural network, implemented as a dict of individual models.
  Be careful during training as the shared weights need to be copied between split instances and are not automatically shared!
  """

  model : tf.keras.models.Model
  """The slimmed down model."""
  name : str
  """The name of the configuration, optional"""
  rewriter : base.Rewriter
  """The rewriter that created that solution"""

  train_dataset : d.DatasetReport
  """The training dataset used to create this model"""
  validation_dataset : d.DatasetReport
  """The validation dataset used to create the model"""
  test_data : d.DatasetReport
  """The test dataset used for this solution"""

  _finetune_steps : List
  """History of the subsequent finetuning steps that have been applied since the solution has been created."""
  _eval_results : Dict[str, Dict[str, object]]
  """caching solution to reuse evaluation results."""
   
  def __init__(self, model:tf.keras.models.Model, rewriter:base.Rewriter, name:str=None) -> None:
    super().__init__()

    self.model = tf.keras.models.clone_model(model)
    self.model.set_weights(model.get_weights())
    self.model.compile(**rewriter.compile_config)

    self.rewriter = rewriter

    self.name = name
    if name is None:
      self.name = self.model.name

    self.train_dataset = rewriter.train_dataset
    self.validation_dataset = rewriter.validation_dataset
    self.test_dataset = rewriter.test_dataset

    self._finetune_steps = []
    self._eval_results = {}

  def evaluate(self, dataset : d.DatasetReport = None) -> Dict[str, object]:
    """
    Evaluates the solution with the given dataset, if the dataset is None, the test set will be used instead.

    Args:
        dataset (d.DatasetReport, optional): The dataset that will be used for the evaluation. Defaults to None.

    Returns:
        Dict[str, object]: the evaluation results.
    """
    if dataset is None:
      dataset = self.test_dataset

    if dataset.name in self._eval_results.keys():
      return self._eval_results[dataset.name]

    self.model.compile(**self.rewriter.compile_config)
    eval_result = self.model.evaluate(dataset.data.batch(128).prefetch(tf.data.AUTOTUNE), return_dict=True)
    exec_cost = 0
    for layer in self.model.layers:
      exec_cost += res.get_layer_macs(layer)

    eval_result["mac_footprint"] = exec_cost

    self._eval_results[dataset.name] = eval_result

    return eval_result

  def finetune(self, epochs:int=1, train_config:Dict[str, object]=None) -> Dict[str, object]:
    """
    Finetunes the solution for the given epochs with the given training configuration.

    Args:
        epochs (int, optional): The number of epochs that the finetuning will run. Defaults to 1.
        train_config (Dict[str, object], optional): Other training configurations. Defaults to None.

    Returns:
        Dict[str, object]: The training history of the finetuning process
    """
    if train_config is None:
      train_config = copy.deepcopy(self.rewriter.train_config)
    train_config["epochs"] = epochs

    train_batch = self.rewriter.acc_report.bs_estimator.training
    inf_batch = self.rewriter.acc_report.bs_estimator.inference

    train_data = self.train_dataset.data.shuffle(buffer_size=3_000).batch(train_batch).prefetch(tf.data.AUTOTUNE)
    val_data = self.validation_dataset.data.batch(inf_batch).prefetch(tf.data.AUTOTUNE)

    self._finetune_steps.append(train_config)

    return self.model.fit(train_data, validation_data=val_data, **train_config)
  
  def toDict(self, contain_objects:bool=False) -> Dict[str, object]:
    """
    Converts the solution into a dict.
    Depending on the use-case, the model can be included or removed from the dict.

    Args:
        contain_objects (bool, optional): If True, complex objects will be part of the dict, will prevent it from being dumped as JSON. Defaults to False.

    Returns:
        Dict[str, object]: the Dict representation of the solution.
    """
    data = {
      "name" : self.name,
      "finetune_history" : self._finetune_steps
    }

    if len(self._eval_results) == 0:
      self.evaluate()
    data["evaluation"] = self._eval_results

    if contain_objects:
      data["model"] = self.model
    return data
  
  def dump(self, path: Union[str,Path]):
    """
    Dumps the solution as collection of JSON and binary files.

    Args:
        path (Union[str,Path]): the path where the solution should be dumped.

    Returns:
        None
    """
    import json

    if isinstance(path, str):
      path = pathlib.Path(path)

    path.mkdir(parents=True, exist_ok=True)

    data = self.toDict()

    # Define a custom encoder function to convert NumPy data types
    def numpy_encoder(obj):
      if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
      elif isinstance(obj, np.float32):
        return float(obj)
      elif isinstance(obj, np.ndarray):
        return obj.tolist()
      else:
        return obj

    with open(path / f"{self.name}_info.json", "w") as file:
      json.dump(data, file, default=numpy_encoder)

    self.model.save(path / f"model.h5")
    self.model.save(path / f"model.tf")

    return None
  
  def to_analysis(self, with_data:bool=False) -> List[object]:
    """
    Converts the solution to a list of new ModelAnalysis instances.
    One instance is created per solution for SlimmableSolutions.

    Returns:
        List[object]: The created ModelAnalysis objects.
    """

    cache_dir = self.rewriter.analysis.cache_dir
    new_analysis = aNN.ModelAnalysis(
      model=self.model,
      name=f"{self.model.name}:Slim-{self.name}",
      cache_dir=cache_dir
    )

    if not with_data:
      return [new_analysis]

    dataset_reports = [
      self.rewriter.train_dataset,
      self.rewriter.validation_dataset,
      self.rewriter.test_dataset,
    ]

    for dataset in dataset_reports:
      if dataset is None:
        continue

      d.DatasetReport.submit_to(new_analysis, lazy=False).with_config(
        name=self.rewriter.train_dataset.name,
        modality=self.rewriter.train_dataset.modality,
        data_preprocessor=self.rewriter.train_dataset.data_preprocessor,
        label_preprocessor=self.rewriter.train_dataset.label_preprocessor
      ).from_source(tf_dataset=self.rewriter.train_dataset.data.batch(64).prefetch(tf.data.AUTOTUNE))

    return [new_analysis]

class SlimmableRewriter(base.Rewriter):
  """
  A rewriter that attempts to convert the submitted model into a multiple different slim version of the model.
  """

  train_dataset : d.DatasetReport
  """The train dataset report used to train the classifier options during the search."""

  validation_dataset : d.DatasetReport
  """The validation dataset report used to evaluate the classifier options during the search."""

  test_dataset : d.DatasetReport
  """The test dataset used to evaluate the found solutions."""

  acc_report : acc.AccuracyReport
  """Uses the accuracy report to get information about the performance of the final classifer."""

  split_style : str
  """How the filters of the model are supposed to be split/grouped"""

  split_steps : int
  """The number of slimmable configurations that shall be created"""

  compile_config : Dict[str, object]
  """The arguments that will be passed to the model.compile function call"""

  train_config : Dict[str, object]
  """The arguments that will be passed to the model.fit function call"""

  exclusive_heads : bool
  """If False, the weights within the classifier head will be shared across configurations.
  If True, each configuration will get its own exclusive set of weights, which can be benefitial for static deployments."""

  solutions : Dict[float, SlimmableSolution]
  """Solutions created by the Rewriter."""

  def __init__(self,
      analysis : aNN.ModelAnalysis,
      train_dataset: d.DatasetReport,
      validation_dataset: d.DatasetReport=None,
      test_dataset: d.DatasetReport=None,
      split_style:str="constant",
      split_steps:int=4,
      compile_config:Dict[str, object]=None,
      train_config:Dict[str, object]=None,
      exclusive_heads:bool=False,
    ) -> None:

    self.solutions = {}
      
    assert split_style in ["constant", "quadratic"], "Input string is not allowed"
    self.split_style = split_style
    self.split_steps = split_steps
    self.exclusive_heads = bool(exclusive_heads)

    self.train_config = train_config
    self.compile_config = compile_config

    assert train_dataset is not None, "Training data can not be None"
    self.train_dataset = train_dataset
    
    if validation_dataset is None:
      log.warn("validation dataset is None, will use train data for these steps!")
      self.validation_dataset = train_dataset
    else:
      self.validation_dataset = validation_dataset

    assert test_dataset is not None, "Training data can not be None"
    self.test_dataset = test_dataset

    self.acc_report : acc.AccuracyReport = acc.AccuracyReport.submit_to(
      analysis=analysis, lazy=False).with_config(datasets={self.train_dataset, self.validation_dataset, self.test_dataset}
    )

    self.steps = self._calculate_steps()

    super().__init__(analysis)

    self.split_config = self._analyze_architecture()

  def _calculate_steps(self) -> List[float]:
    """
    Calculates the step sizes for the given search config.

    Raises:
        NotImplementedError: So far, only 'constant' is implemented as slimming style

    Returns:
        List[float]: the configured steps
    """
    if self.split_style == "constant":
      step_size = 1 / self.split_steps
      steps = [(n+1) * step_size for n in range(self.split_steps)] #[step_size] * self.split_steps
    else:
      raise NotImplementedError("currently only 'constant' has been implemented as split_style, creating equally sized splits.")
    
    return steps


  def _analyze_architecture(self)-> Dict[str, List[int]]:
    """
    This function will identify the layers of the model that can be converted and which filter configs would fit the split configuration

    Returns:
        Dict[str, List[int]]: A Dictionary, the keys are the layer names, the values are lists with the appropriate filter lists
    """

    # only consider layers in the feature extraction subgraph of the model, the classifier heads should not be touched
    fe_subgraph = self.analysis.architecture.identify_feature_extraction_subgraph(layer_level=True)

    configs = {}

    for node in fe_subgraph.nodes:
      if node.layer_class in ["convolution", "compute"]:
        inter_param = None

      if node.layer_class == "convolution" and node.layer_type != "DepthwiseConv2D":
        inter_param = node.keras.filters
      elif node.layer_class == "compute" and node.layer_type == "Dense":
        inter_param = node.keras.units
      else:
        continue
      
      configs[node.name] = [int(inter_param * step) for step in self.steps]

    return configs
  
  def evaluate(self)-> Dict[str, tf.keras.models.Model]:
    """
    Creates the slimmed model configurations, which are not yet trained.

    Returns:
        Dict[str, tf.keras.models.Model]: A dict of the created model architectures.
    """
    self.split_config = self._analyze_architecture()

    # need to convert model into Slimmable version
    #TODO: replace found layers with slimmable variant

    slim_models = {}

    for step_id, step in enumerate(self.steps):
      model = tf.keras.models.clone_model(self.analysis.keras_model)
      model.set_weights(self.analysis.keras_model.get_weights())
      
      slimmed_layers = {}
      for layer_name, filters_list in self.split_config.items():
        print(layer_name, filters_list)

        orig_layer = model.get_layer(layer_name)
        orig_config = orig_layer.get_config()
        
        orig_type = type(orig_layer)

        if isinstance(orig_layer, (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D, tf.keras.layers.Conv3D)):
          orig_config["filters"] = filters_list[step_id]

          
        elif isinstance(orig_layer, tf.keras.layers.Dense):
          orig_config["units"] = filters_list[step_id]

        slim_version = orig_type.from_config(orig_config)
        slimmed_layers[layer_name] = slim_version
        #exchange layers
        model, layer_ids = exchange_layer(orig_layer, [slim_version], model, share_layers=False)


      # copy weights from original model to slimmed versions
      for new_layer, old_layer in zip(model.layers, self.analysis.keras_model.layers):
        weight.copy_weights(old_layer=old_layer, new_layer=new_layer)
      
      model._name = f"{self.analysis.keras_model.name}_{step}"
      slim_models[step] = (model, slimmed_layers)

    self.slim_models = slim_models

    return slim_models
  
  def configure(self) -> Dict[float, SlimmableSolution]:
    """
    Configures the created model architectures by training them.
    Performs transfer learning by copying the weights from the smaller to the larger configurations.
    Returns a dict of SlimmableSolutions that represent the different created slimmed configurations.

    Returns:
        Dict[float, SlimmableSolution]: A dict of the created solutions.
    """

    if self.exclusive_heads:
      classifier_head = self.analysis.architecture.identify_classifier_subgraph(layer_level=True)
      head_layer_names = [node.keras.name for node in classifier_head.nodes]

    if self.slim_models == None:
      self.slim_models = self.evaluate()

    configs = sorted(list(self.slim_models.keys()))

    train_batch = self.acc_report.bs_estimator.training
    inf_batch = self.acc_report.bs_estimator.inference
    train_data = self.train_dataset.data.shuffle(buffer_size=3_000).batch(train_batch).prefetch(tf.data.AUTOTUNE)
    val_data = self.validation_dataset.data.batch(inf_batch).prefetch(tf.data.AUTOTUNE)

    prev_model = None
    solutions = {}
    for width_config in configs:
      model, slim_layers = self.slim_models[width_config]

      #TODO: copy weights from previous split into larger one
      if not prev_model is None:
        for old_layer, new_layer in zip(prev_model.layers, model.layers):
          if self.exclusive_heads and old_layer.name in head_layer_names:
            continue
          weight.copy_weights(old_layer=old_layer, new_layer=new_layer)

      #TODO: train model
      model.compile(**self.compile_config)
      model.fit(train_data, validation_data=val_data, **self.train_config)

      solutions[width_config] = SlimmableSolution(model, self, name=width_config)

      prev_model = model

    self.solutions.update(solutions)
    return solutions

  def create_identifier(self) -> str:
    """
    Creates a unique ID for the rewriter, based on its parameters.
    This will be used to reuse already processed rewrites for the model.

    Returns:
        str: the unique ID, which is a hash of the rewriter parameters.
    """
    # what makes this rewriter unique?
    # datasets, split_style, split_steps, exclusive_heads compile and train config
    config_str = f"{self.analysis.keras_model.name}: {self.train_dataset.access_id()}-{self.validation_dataset.access_id()}-{self.test_dataset.access_id()}"
    config_str = f"{config_str}: {self.split_style}-{self.split_steps}-{self.exclusive_heads}: {self.train_config}-{self.compile_config}"
    descriptor = f"SlimRewriter:{config_str}"
    sha256_hash = hashlib.sha256()
        
    # Update the hash object with the input string
    sha256_hash.update(descriptor.encode('utf-8'))
    hashed_representation = sha256_hash.hexdigest()

    return hashed_representation
  
  @classmethod
  def create_pass(cls,
                  train_dataset:tf.data.Dataset,
                  validation_dataset:tf.data.Dataset,
                  test_dataset:tf.data.Dataset,
                  split_style:str="constant",
                  split_steps:int=4,
                  compile_config:Dict[str, object]=None,
                  train_config:Dict[str, object]=None,
                  exclusive_heads:bool=False
                ) -> Tuple[str, callable]:
    """
    Creates a pass that can be added to the optimization queue of a model analysis object.
    This pass contains all steps necessary to produce a solution of this rewrite from the ModelAnalysis of the submitted model.

    Returns a name for the pass as well as the function that needs to be called to apply the rewrite to the analysis

    Args:
        train_dataset (tf.data.Dataset): The training data
        validation_dataset (tf.data.Dataset): The validation data
        test_dataset (tf.data.Dataset): The test data
        split_style (str, optional): How the different splits should be created. Defaults to "constant".
        split_steps (int, optional): How many slimmed models should be created. Defaults to 4.
        compile_config (Dict[str, object], optional): Compilation parameters for the models. Defaults to None.
        train_config (Dict[str, object], optional): training parameters for the models. Defaults to None.
        exclusive_heads (bool, optional): False, if weights should be copied between the layers of the classifier heads. Defaults to False.

    Returns:
        Tuple[str, callable]: The pass name as well as its callable implementation.
    """

    str_id:str = f"SlimmableRewrite_{split_style}, {split_steps}, {compile_config}, {train_config}, {exclusive_heads}"


    def rewrite(analysis:aNN.ModelAnalysis) -> List[SlimmableSolution]:

      # create dataset reports
      train_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="train", modality=None).from_source(tf_dataset=train_dataset)

      valid_data_report = None
      if validation_dataset is not None:
          valid_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="valid", modality=None).from_source(tf_dataset=validation_dataset)

      test_data_report = None
      if test_dataset is not None:
          test_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="test", modality=None).from_source(tf_dataset=test_dataset)

      rewriter = SlimmableRewriter(
        analysis=analysis,
        train_dataset=train_data_report,
        validation_dataset=valid_data_report,
        test_dataset=test_data_report,
        split_style=split_style,
        split_steps=split_steps,
        compile_config=compile_config,
        train_config=train_config,
        exclusive_heads=exclusive_heads
      )

      sols = rewriter.configure()
      return sols

    return str_id, rewrite
  
  def dump(self, folder_path: Union[str, pathlib.Path] = None) -> Dict[str, object]:
    """
    Dumps the raw data of the rewriter into the given folder

    Args:
        folder_path (Union[str, pathlib.Path], optional): The folder in which the dump will be stored. Uses current working directory if None. Defaults to None.

    Returns:
        Dict[str, object]: dict representation of the rewriter.
    """

    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    solutions_path = folder_path / f"rewriter_slimming_{self.create_identifier()}"
    solutions_path.mkdir(parents=True, exist_ok=True)
    
    sol_paths = {}
    for name, sol in self.solutions.items():
      path: Path = solutions_path / f"sol_{sol.name}"
      sol.dump(path)
      sol_paths[name] = str(path)
    
    compile_config = {}
    for key, val in self.compile_config.items():
      if isinstance(val, list):
        compile_config[key] = [str(x) for x in val]
      else:
        compile_config[key] = str(val)

    data = {
      "solution_paths" : sol_paths,
      "split_style" : self.split_style,
      "split_steps" : self.split_steps,
      "compile_config":compile_config,
      "train_config": self.train_config,
      "exclusive_heads": self.exclusive_heads
    }

    with open(folder_path / f"rewriter_slimming_{self.create_identifier()}.json", "w") as file:
      json.dump(data, file)

    return data
  
  def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> Tuple[str, str]:
    """
    Creates the HTML file for the summary overview

    Args:
      folder_path (Union[str, pathlib.Path], optional): folder, in which the file and auxiliary data will be stored. Defaults to None.
    
    Returns:
      Tuple[str, str]: The name of the rewriter and the path to its html summary.
    """
    
    _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / '..' / 'templates'

    if folder_path is None:
      folder_path = pathlib.Path.cwd()

    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    with open(_template_path / "slim_rewriter.html", "r") as file:
      template = Template(file.read())
        
    summary = self.dump(folder_path=folder_path)

    summary["report_type"] = "SlimmingRewriter"
    summary["text"] = f"The rewriter created {len(self.solutions)} slimmed variants of the base model."

    summary["solutions"] = self.solutions

    html = template.render(summary=summary)
    html_filename = f"rewriter_slim_models_{self.create_identifier()}.html"
    html_path = folder_path / html_filename
    with open(html_path, "w") as file:
      file.write(html)
    
    return (
            "Slimmed Models",
            html_filename,
        )
