import hashlib
import logging as log
import pathlib
import json
from typing import Set, Tuple, Union, Dict, List
import numpy as np
import tensorflow as tf

import peax.analysis as aNN
from . import base

from peax.components import graph_tools as gt

from ..reports import early_exit as ee
from ..reports import accuracy as acc
from ..reports import dataset as d

class RightSizingSolution(base.Solution):
  """
  The solution object for the RightSizing Rewriter.
  Wraps all important components of the solution.
  """
  
  location : gt.BlockNode
  """Block Node after which the new classifier is attached."""

  cost_value : float
  """The relative cost for the inference of the new model compared to the original version."""

  description : dict
  """Evaluation results from search."""

  model : tf.keras.models.Model
  """The new model as keras model."""

  finetune_epochs: int
  """Number of applied fine-tuning epochs."""

  def __init__(self, location : gt.BlockNode, score : float, description : dict, new_classifier: tf.keras.models.Model, rewriter : base.Rewriter) -> None:
    super().__init__()
    self.rewriter = rewriter
    self.location = location
    self.cost_value = score
    self.description = description
    self.new_classifier = new_classifier
    self.model = self.compile()
    self.finetune_epochs = 0

  def compile(self) -> tf.keras.models.Model:
    """
    Creates the new model from the description and a copy of the original model.

    Returns:
        tf.keras.models.Model: The shrunken model.
    """

    # get a copy of the backbone model
    orig_model = tf.keras.models.clone_model(self.rewriter.analysis.keras_model)
    orig_model.build(self.rewriter.analysis.keras_model.input_shape)
    orig_model.set_weights(self.rewriter.analysis.keras_model.get_weights())
    input_tensors = orig_model.inputs

    # get the section that will be attached and where to attach it
    attach_layer_name = gt.get_first_output_node(self.location.subgraph).keras.name
    attach_layer = orig_model.get_layer(attach_layer_name)
    x = attach_layer.output

    attach_subgraph = tf.keras.models.clone_model(self.new_classifier)
    attach_subgraph.build(x)
    attach_subgraph.set_weights(self.new_classifier.get_weights())

    # glue it together
    source_weights : Dict[str, np.array] = dict()
    for layer in attach_subgraph.layers[1::]:
      config = layer.get_config()
      config["name"] = f"{attach_layer.name}_{config['name']}"
      source_weights[config["name"]] = layer.get_weights()

      x = type(layer).from_config(config)(x)

    # Create the new Keras model
    new_model = tf.keras.Model(inputs=input_tensors, outputs=[x])

    for layer_name, weights in source_weights.items():
        new_model.get_layer(layer_name).set_weights(weights)

    return new_model
  
  def finetune(self, epochs:int=1):
    """
    Optional finetuning / training step for the model within this solution

    Args:
        epochs (int, optional): The number of epochs for the training. Defaults to 1.

    Returns:
        Dict[str, object]: The evaluation of the model after the tuning step
    """
    batch_size = self.rewriter.acc_report.batch_size
    train_dataset = self.rewriter.train_dataset.data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_config = self.rewriter.ee_report._determine_ee_training_config(self.location, batch_size=batch_size)

    self.model.compile(**train_config)
    self.model.fit(train_dataset, batch_size=batch_size, epochs=epochs)
    self.finetune_epochs += epochs

    return self.evaluate()
  
  def evaluate(self, x=None, y=None, **kwargs) -> Dict[str, object]:
    """
    Evaluates the performance of the found solution.
    Dedicated samples and labels can be submitted as lists of numpy arrays, but this is optional.
    If no x and y values are submitted, the solution will try to access the test or validation report that should be submitted to the ModelAnalysis.

    Args:
        x (_type_, optional): The samples that should be used for the evaluation. Defaults to None.
        y (_type_, optional): The labels that should be used for the evaluation. Defaults to None.

    Returns:
        Dict[str, object]: The evaluation results, contains information about accuracy.
    """
    result = []
    batch_size = self.rewriter.acc_report.batch_size
    if x is None and y is None:
      data = self.rewriter.test_dataset.data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
      result = self.model.evaluate(data, **kwargs, return_dict=True)
    else:
      result = self.model.evaluate(x, y, **kwargs, return_dict=True)

    self.description["accuracy"] = result["categorical_accuracy"] #TODO: extend for non-classification tasks
    self.description["relative"]["accuracy"] = result["categorical_accuracy"] - (self.rewriter.acc_report.results[self.rewriter.test_dataset]["top1_accuracy"]/100)
    
    return result
  
  def dump(self, path : Union[str, pathlib.Path]):
    """
    Writes solution to disk.

    Args:
        path (Union[str, pathlib.Path]): The folder in which the data should be stored.
    """
    import json

    if isinstance(path, str):
      path = pathlib.Path(path)

    self.model.save(path / f"{self.location.name}_model.tf")
    self.model.save(path / f"{self.location.name}_model.h5")
    
    data = self.toDict()
    data["location"] = self.location.name

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

    with open(path / f"{self.location.name}_info.json", "w") as file:
      json.dump(data, file, default=numpy_encoder)

    return
  
  def to_analysis(self) -> List[aNN.ModelAnalysis]:
    """
    Converts the solution object into a new ModelAnalysis to continue

    Returns:
        List[aNN.ModelAnalysis]: a list of length one, that contains a new model analysis object
    """

    model = self.model
    name = self.rewriter.analysis.name + ":Right-Sized"
    mac_estimators = self.rewriter.analysis.compute.mac_estimators
    cache_dir = self.rewriter.analysis.cache_dir


    analysis = aNN.ModelAnalysis(model=model,
                                 name=name,
                                 mac_estimators=mac_estimators,
                                 cache_dir=cache_dir
                                )
    
    return [analysis]

  def toDict(self, contain_objects:bool=False) -> Dict[str, object]:
    """
    Converts the solution into a dict.

    Args:
        contain_objects (bool, optional): If True, the model and other objects that cannot be stored as JSON will be included. Defaults to False.

    Returns:
        Dict[str, object]: The dict representation of the solution object.
    """
    components = dict()
    components["location"] = self.location
    components["score"] = self.cost_value
    components["description"] = self.description
    components["finetune"] = self.finetune_epochs

    if contain_objects:
      components["model"] = self.model

    return components

class RightSizingRewriter(base.Rewriter):
  """
  The rewriter to apply right sizing.
  Right Sizing introduces a new classifier between hidden layers of the original model and prunes all layers after this attachement point.
  This effectively reduces the model depth. Such an approach can often achieve significant efficiency gains with acceptable reductions in accuracy.
  Especially ResNet architectures can benefit from this approach and allow for deploying one model to a wide range of different devices easily.
  """

  train_dataset : d.DatasetReport
  """The train dataset report used to train the classifier options during the search."""

  validation_dataset : d.DatasetReport
  """The validation dataset report used to evaluate the classifier options during the search."""

  test_dataset : d.DatasetReport
  """The test dataset used to evaluate the found solutions."""

  ee_report : ee.EarlyExitReport
  """This rewriter uses the Early Exiting infrastructure to find and configure possible early exits."""

  acc_report : acc.AccuracyReport
  """Uses the accuracy report to get information about the performance of the final classifer."""

  ee_performance : Dict[gt.BlockNode, Dict[str, object]]
  """Performance of the evaluated classifier options."""

  solutions : Dict[gt.BlockNode, RightSizingSolution]
  """Found an extracted solutions."""

  def __init__(self,
              analysis: aNN.ModelAnalysis,
              train_dataset: d.DatasetReport,
              validation_dataset: d.DatasetReport=None,
              test_dataset: d.DatasetReport=None,
              classifier_size:str="large"
              ) -> None:

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
      log.warn("validation dataset is None")
      self.validation_dataset = train_dataset
    else:
      self.validation_dataset = validation_dataset
      self.validation_dataset.shuffle()

    self.ee_report : ee.EarlyExitReport = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config=classifier_size)
    self.acc_report : acc.AccuracyReport = acc.AccuracyReport.submit_to(analysis=analysis, lazy=False).with_config(datasets={self.train_dataset, self.validation_dataset, self.test_dataset})

    self.ee_performance = {}
    self.solutions = dict()

    super().__init__(analysis)
    pass

  def evaluate(self) -> RightSizingSolution:
    """
    Executes the search for solutions.
    """
    self.ee_performance = {}
    exit_submodels = {}
    for recom in self.ee_report.recommendations:
      #ee_model = self.ee_report.to_keras((recom, self.ee_report.exit_configs[recom]))
      #train_config = self.ee_report._determine_ee_training_config(ee_model)

      exit_submodel = self.ee_report.to_keras((recom, self.ee_report.exit_configs[recom]))

      self.ee_performance[recom] = {}
      self.ee_performance[recom]["accuracy"] =self.ee_report.evaluate_precision((recom, exit_submodel), self.train_dataset, self.test_dataset, batch_size=256)[0]
      self.ee_performance[recom]["macs"] = self.ee_report.subgraph_costs[recom] + self.ee_report.exit_costs[recom]
      self.ee_performance[recom]["relative"] = {}
      self.ee_performance[recom]["relative"]["accuracy"] = self.ee_performance[recom]["accuracy"] - (self.acc_report.results[self.test_dataset]["top1_accuracy"]/100)
      self.ee_performance[recom]["relative"]["macs"] = self.ee_performance[recom]["macs"] / list(self.analysis.compute.total_mac.values())[0]
      
      
      exit_submodels[recom] = exit_submodel

    ## create Pareto Front from the found solutions
    def is_optimal(point, points):
      """
      Check if a given point is Pareto optimal based on its coordinates (relative_accuracy, relative_macs)
      """
      is_dominated = False
      for other_point in points:
          if (
              other_point["relative"]["accuracy"] > point["relative"]["accuracy"]
              and other_point["relative"]["macs"] < point["relative"]["macs"]
          ):
              is_dominated = True
              break
      return not is_dominated
    
    optimal_points = {
      key: value
      for key, value in self.ee_performance.items()
      if is_optimal(value, self.ee_performance.values())
    }

    ## find best accuracy per mac cost point on pareto front
    best_reward = -np.Infinity
    best_point = None
    for point, description in optimal_points.items():
      reward = description["relative"]["accuracy"] / description["relative"]["macs"]
      print(point, reward)

      if reward > best_reward:
        best_reward = reward
        best_point = point
        best_description = description
        submodel = self.ee_report.to_keras((best_point, self.ee_report.exit_configs[best_point])) #ee_report.to_keras((pos, layers))

    if best_point is not None:
      self.solutions[best_point] = RightSizingSolution(best_point, best_reward, best_description, exit_submodels[best_point], self) #(best_reward, description)

    return self.solutions[best_point]

  def create_identifier(self) -> str:
    """
    Creates the unique identifer for this report.

    Returns:
        str: the unique ID.
    """
    descriptor = f"RightSizing:{self.analysis.keras_model.name}"
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
        folder_path (Union[str, pathlib.Path], optional): The folder in which the data shall be dumped.
        Uses current working directoy if None. Defaults to None.

    Returns:
        Dict[str, object]: A dict representation of the rewriter.
    """
    
    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)
    
    solutions_path = folder_path / f"rewriter_right_sizer_{self.create_identifier()}/"
    solutions_path.mkdir(parents=True, exist_ok=True)

    solutions = {location.name: (values.finetune_epochs, values.toDict()["description"]) for location, values in self.solutions.items()}
    
    summary = {
      "report_type": "Right-Sizing",
      "name": self.analysis.name,
      "creation_date": str(self.analysis.creation_time),
      "solutions_path" : str(solutions_path),
      "solutions" : solutions,
      "options" : {location.name: values for location, values in self.ee_performance.items()}, #self.ee_performance,
    }

    for location, sol in self.solutions.items():
      log.debug(f"dumping solution for {location.name}")
      sol.dump(solutions_path)

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

    with open(folder_path / f"rewriter_right_sizer_{self.create_identifier()}.json", "w") as file:
      json.dump(summary, file, default=numpy_encoder)
    
    return summary
  
  def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> Tuple[str, str]:
    """
    Creates a human-readable HTML-based summary of the rewriter and the found solutions.

    Args:
        folder_path (Union[str, pathlib.Path], optional): The folder in which the data shall be dumped.
        Uses current working directoy if None. Defaults to None.

    Returns:
        Tuple[str, str]: The title being shown for the report in the ModelAnalysis summary and the name of the html file.
    """
    import os
    from jinja2 import Template
    
    _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'

    if folder_path is None:
      folder_path = pathlib.Path.cwd()

    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    with open(_template_path / "right_sizing_rewriter.html", "r") as file:
      template = Template(file.read())

    summary = self.dump(folder_path=folder_path)

    summary["possible_x"] = (x["macs"] for x in self.ee_performance.values())
    summary["possible_y"] = (x["accuracy"] for x in self.ee_performance.values())
    summary["possible_labels"] = (x for x in self.ee_performance.keys())

    summary["solutions_x"] = (x[1]["macs"] for x in self.solutions.values())
    summary["solutions_y"] = (x[1]["accuracy"] for x in self.solutions.values())
    summary["solutions_labels"] = (x for x in self.solutions.keys())

    # Render the template with the summary data
    html = template.render(summary=summary)
    # Save the generated HTML to a file
    html_filename = f"rewriter_right_sizer_{self.create_identifier()}.html"
    html_path = folder_path / html_filename
    with open(html_path, "w") as file:
        file.write(html)

    return (
        "Right Sizer",
        html_filename,
    )
  
  @classmethod
  def create_pass(cls, train_data:tf.data.Dataset, valid_data:tf.data.Dataset, test_data:tf.data.Dataset, classifier_size:str="small") -> Tuple[str, callable]:
    """
    Creates a pass that can be added to the optimization queue of a model analysis object.
        This pass contains all steps necessary to produce a solution of this rewrite from the ModelAnalysis of the submitted model.

        Returns a name for the pass as well as the function that needs to be called to apply the rewrite to the analysis.

    Args:
        train_data (tf.data.Dataset): The training data report wrapper.
        valid_data (tf.data.Dataset): The validation data report wrapper.
        test_data (tf.data.Dataset): The test data report wrapper.
        classifier_size (str, optional): The design guideline for the added classifiers, currently limited to "small" and "large". Defaults to "small".

    Returns:
        Tuple[str, callable]: The pass name as well as its callable implementation.
    """
    
    str_id:str = f"RightSizingRewrite_{classifier_size}"

    def rewrite(analysis:aNN.ModelAnalysis) -> RightSizingSolution:

      # create dataset reports
      train_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="train", modality=None).from_source(tf_dataset=train_data)

      valid_data_report = None
      if valid_data is not None:
          valid_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="valid", modality=None).from_source(tf_dataset=valid_data)

      test_data_report = None
      if test_data is not None:
          test_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="test", modality=None).from_source(tf_dataset=test_data)

      rs_rewriter = RightSizingRewriter(
          analysis=analysis,
          train_dataset=train_data_report,
          validation_dataset=valid_data_report,
          test_dataset=test_data_report,
          classifier_size=classifier_size,
        )

      return rs_rewriter.evaluate()
    
    return str_id, rewrite