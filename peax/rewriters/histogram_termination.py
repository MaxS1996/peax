
import hashlib
import os
from pathlib import Path
from re import search
from typing import List, Union, Tuple, Dict
import pathlib
from xml.etree.ElementInclude import include
import numpy as np
import logging as log
import json
import pickle as pkl

#from scipy.stats import norm
import tensorflow as tf
from sklearn.metrics import accuracy_score as acc_score
from jinja2 import Template

from .base import Rewriter, Solution
from . import temporal_early_exit as tee

from peax.components import graph_tools as gt
from peax.components import predictive as prd

from peax.reports import dataset, histograms as hist
from peax.reports import dataset as data
from peax.reports import temporal as temp
from peax.reports import accuracy as acc

import peax.analysis as aNN
import peax.utils.optimization as optimization
import peax.utils.keras_graph as keras_graph
import peax.utils.normal_distribution as norm

def _eval_histo_term(samples, labels, raw_predictions, ifms, threshold, scene_based:bool = False) -> Tuple[float, float]:
  """
  Evals the performance of a histogram-based difference detection.
  To minimize the cost, the model predictions are passed as inputs to this function to enable reuse
  and minimize the cost of multiple executions.

  Args:
      samples (_type_): The input samples.
      labels (_type_): The ground truth labels.
      raw_predictions (_type_): The predictions of the final classifier in the original model.
      ifm_deltas (_type_): the deltas between the IFMs created by the histogram exit.
      threshold (_type_): The threshold parameter.

  Raises:
      AssertionError: If the required labels, samples and predictions are not of the same length.

  Returns:
      Tuple[float, float]: the accuracy and early termination rate
  """
  if len(samples) != len(raw_predictions):
    raise AssertionError("sample and prediction data not of the the same length!")

  if len(samples) != len(labels):
    raise AssertionError("sample and labels data not of the same length!")
  
  curr_pred = raw_predictions[0]

  activity = [1]
  preds = [curr_pred]

  pred_ifm = ifms[0]

  for pred, ifm in zip(raw_predictions[1::], ifms[1::]):
    delta = np.abs(np.linalg.norm(pred_ifm - ifm))

    if not scene_based:
      pred_ifm = ifm
    
    if delta >= threshold:
      is_active = 1
      curr_pred = pred
      pred_ifm = ifm # only relevant for scene based decisions
    else:
      is_active = 0

    activity.append(is_active)
    preds.append(curr_pred)

  if isinstance(labels, np.ndarray) and len(labels.shape) > 1 and labels.shape[-1] != 1:
    labels = np.argmax(labels, axis=-1)

  preds = np.array(preds)

  if labels.shape != preds.shape:
    if len(labels.shape) < len(preds.shape):
      preds = np.argmax(preds, axis=-1)

  acc = acc_score(labels, preds)
  savings = 1 - (sum(activity) / len(activity))

  return acc, savings

class HistogramExitSolution(Solution):
  """
  The solution class to wrap the output of the search step for the HistogramExitRewriter pass.
  """

  model : tf.keras.models.Model
  """ The model with the inserted avg pooling branch. """

  location : gt.BlockNode
  """ The location of the added branch. """

  threshold : float
  """ The threshold for the histogram delta to be identified as different enough to execute a full inference. """

  scene_based : bool
  """ False to use the direct predecessor as reference value, otherwise the last active value is used as reference."""

  def __init__(self, location: gt.BlockNode, threshold:float, scene_based:bool, rel_cost:float, accuracy:float, rewriter:Rewriter) -> None:
    self.rewriter = rewriter
    self.threshold = threshold
    self.scene_based = scene_based

    self.rel_cost = rel_cost
    self.accuracy = accuracy

    base_model : tf.keras.models.Model = self.rewriter.analysis.keras_model
    branch : gt.BlockNode = self.rewriter.histogram_report.branches[location]
    attach_layer = gt.get_first_output_node(location.subgraph).keras

    branched_model = keras_graph.attach_branch(base_model, attach_layer, branch, reorder=True)

    self.location = location
    self.model = branched_model

  def evaluate(self, sequence: temp.CalibrationSequence, as_dict:bool=True, scene_based:bool=False) -> Union[Dict[str, float], Tuple[float, float, float]]:
    """
    _summary_

    Args:
        sequence (temp.CalibrationSequence): An example sequence of correlated sensor data.
        as_dict (bool, optional): If the metrics should be returned as dict, with descriptive string names as keys. Defaults to True.
        scene_based (bool, optional): If the direct predecessor should be considered or a reference scene start. Defaults to False.

    Returns:
        Union[Dict[str, float], Tuple[float, float, float]]: The accuracy, early termination rate and mac inference cost on the provided test sequence.
    """

    samples = sequence.x
    labels = sequence.y
    node = self.location

    preds = self.model.predict(samples) #creates list of numpy arrays, first array is the output of the histogram branch, second of the classifier
    ifms = preds[0]
    final_preds = preds[-1]

    '''ifm_deltas = [np.linalg.norm(ifms[0])]

    for i in range(1, ifms.shape[0]):
      distance = np.linalg.norm(ifms[i] - ifms[i-1])
      ifm_deltas.append(distance)'''

    '''if list(self.rewriter.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
      if isinstance(preds, np.ndarray) and len(preds.shape) > 1 and preds.shape[-1] != 1:
        preds = np.argmax(preds, axis=-1)
    elif list(self.rewriter.analysis.tasks.values())[0] == prd.Task.BINARY_CLASSIFICATION:
      preds= [1 if val > 0.5 else 0 for val in preds]'''
    
    if list(self.rewriter.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
      if isinstance(preds, np.ndarray) and len(preds.shape) > 1 and preds.shape[-1] != 1:
        preds = np.argmax(preds, axis=-1)
    elif list(self.rewriter.analysis.tasks.values())[0] == prd.Task.BINARY_CLASSIFICATION:
      final_preds = np.array([1 if val > 0.5 else 0 for val in final_preds])

    #acc, early_rate = _eval_histo_term(samples, labels, final_preds, ifm_deltas, self.threshold)
    acc, early_rate = _eval_histo_term(samples, labels, final_preds, ifms, self.threshold, scene_based=scene_based)
    mac_cost = early_rate * self.rewriter.histogram_report.mac_costs[node] + (1-early_rate) * (1.0 + self.rewriter.histogram_report.branch_mac_costs[node]  + self.rewriter.histogram_report.dist_mac_costs[node])

    vals = (
      acc,
      early_rate,
      mac_cost
    )

    if as_dict:
      vals = {
        "accuracy" : acc,
        "early_termination" : early_rate,
        "mac_footprint" : mac_cost,
      }

    return vals
  
  def finetune(self):
    """
    Finetuning is not supported/required for HistogramExitSolutions.

    Raises:
        NotImplementedError: will always return an NotImplementedError
    """
    log.warn("HistogramExitSolution cannot be finetuned.")
    raise NotImplementedError
  
  def toDict(self, include_model:bool=True, make_serializable:bool=False) -> Dict[str, object]:
    """
    Converts the solution object into a dict.

    Returns:
        Dict[str, object]: the members of the serialized object.
    """
    data_dict = {
      "threshold" : self.threshold,
      "location" : self.location,
      "scene_based" : self.scene_based,
      "accuracy" : self.accuracy,
      "relative_cost" : self.rel_cost
    }

    if include_model:
      data_dict["model"] = self.model

    if make_serializable:
      data_dict["location"] = self.location.name

    return data_dict
  
  def split(self) -> List[tf.keras.models.Model]:
    """
    Splits the model into submodels.
    First submodel contains all layers up to and including the histogram branch, second submodel contains remaining layer up to original output.

    Returns:
        List[tf.keras.models.Model]: The list of submodels as Keras models.
    """

    return keras_graph.split_eenn_model(self.model, [self.location])
  
  def dump(self, path : Union[str, pathlib.Path], model_format:str="tf") -> Dict[str, object]:
    """
    Writes the solution object to disk.

    Args:
        path (Union[str, pathlib.Path]): the folder path to which the solution shall be written.
        model_format (str, optional): The model format to serialize the wrapped model. "h5", "tf", "keras" are supported. Defaults to "tf".

    Raises:
        NotImplementedError: Raises an error if an unsupported model format is passed to the function

    Returns:
        Dict[str, object]: Returns a serialized version of the solution object.
    """

    if not model_format in ["tf", "h5", "keras"]:
      raise NotImplementedError(f"{model_format} is not supported.")

    data_dict = self.toDict()
    del data_dict["model"]

    model_path = path / f"{self.model.name}.{model_format}"
    dump_path = path / f"sol-{self.model.name}.json"
    tf.keras.models.save_model(self.model, model_path)

    data_dict["model_path"] = str(model_path)
    data_dict["location"] = self.location.name

    with open(dump_path, "w") as file:
      json.dump(data_dict, file)

    return data_dict
  
  def to_analysis(self) -> List[aNN.ModelAnalysis]:
    """
    Converts the solution into a List of new ModelAnalysis objects.

    Returns:
        List[aNN.ModelAnalysis]: The ModelAnalysis objects that represent the subgraphs created for this solution.
    """
    # two solutions
    submodels = self.split()

    anal_list = []
    old_analysis :aNN.ModelAnalysis = self.rewriter.analysis
    for sub_id, submodel in enumerate(submodels):
      print(sub_id)
      new_analysis = aNN.ModelAnalysis(model=submodel, name=f"{old_analysis.name}:HistoTERM-{sub_id}", mac_estimators=old_analysis.compute.mac_estimators, cache_dir=old_analysis.cache_dir)
      anal_list.append(new_analysis)

    return anal_list

class HistogramExitRewriter(Rewriter):
  """
  A rewrite pass that inserts a average pooling branch into the model and uses its output to compare the current inference to the previous iteration.
  """

  __pickle_name_template = "rewriter_histogram_termination-<hash>.pkl"
  __solution_path_template = "solution_histogram_termination-<hash>-<mode>/"
  __summary_name_template = "rewriter_histogram_termination-<hash>.html"
  __summary_template_file = _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates' / "histogram_rewriter.html"

  analysis: aNN.ModelAnalysis
  """ the full model analysis that is the base for this optimization run """

  histogram_report : hist.HistogramReport
  """ The histogram report that is used to configure the adaptive inference termination"""

  node_score : Dict[gt.BlockNode, Tuple[bool, Tuple[float, float], Tuple[float, float]]]
  """
  The scoring of each node on the provided calibration sequences.
  The keys are the BlockNodes, the values are a Tuple containing the viability of using this location and the parameters of the normal distributions of deltas for scenes and transitions.
  """

  thres_boundaries : Dict[gt.BlockNode, Tuple[float, float]]
  """ The recommended boundaries for the threshold parameter search space."""

  search_space : Dict[Tuple[float, float], Tuple[gt.BlockNode, float, bool]]
  """
  A dict that represents the points within the currently constructed search space.
  The keys are the coordinates within the search space (cost, accuracy), the values are the configurations (node location, threshold value).
  """ 

  solutions : Dict[str, Tuple[Tuple[float, float], HistogramExitSolution]]
  """
  Solutions that already have been created by this rewriter.
  Values contain the location of the solution within the search space
  as first element and the solution object as second.
  """

  def __init__(self,
               analysis: aNN.ModelAnalysis,
               train_dataset : data.DatasetReport,
               validation_dataset : data.DatasetReport,
               calibration_sequences : List[temp.CalibrationSequence],
               histogram_size:str="global",
               cost_limit:float=0.8

              ) -> None:
    """
    Constructor function. Should not be used directly, instead use the submit syntax.

    Args:
        analysis (aNN.ModelAnalysis): The ModelAnalysis with which the Rewriter is associated.
        train_dataset (data.DatasetReport): A training dataset.
        validation_dataset (data.DatasetReport): An optional validation dataset. Both datasets are used for the histogram report.
        calibration_sequences (List[temp.CalibrationSequence]): Representitive sequences of data as it would be encountered in production.
        histogram_size (str, optional): The configuration for the histogram size. Defaults to "global".
        cost_limit (float, optional): The limitation of the relative MAC cost for evaluated locations within the network graph. Defaults to 0.8.
    """

    self.train_dataset = train_dataset

    if validation_dataset	is not None:
      self.validation_dataset = validation_dataset
      datasets = [train_dataset, validation_dataset]
    else:
      self.validation_dataset = train_dataset
      datasets = [train_dataset]
    self.calibration_sequences = calibration_sequences

    self.hist_size = histogram_size

    self.histogram_report : hist.HistogramReport = hist.HistogramReport.submit_to(
      analysis=analysis,
      lazy=False
    ).with_config(
      train_dataset=train_dataset,
      validation_dataset=validation_dataset,
      calibration_sequences=calibration_sequences,
      max_cost=cost_limit,
      resolution=histogram_size
      )
    
    #self.accuracy_report : acc.AccuracyReport = acc.AccuracyReport.submit_to(analysis=analysis, lazy=False).with_config(datasets=datasets)

    self.node_score = {}
    self.thres_boundaries = {}
    self.search_space = {}
    self.solutions = {}

    self.__search_space_res = 150

    super().__init__(analysis)
    
    return
  
  def evaluate(self) -> Dict[gt.BlockNode, Tuple[bool, Tuple[float, float], Tuple[float, float]]]:
    """
    Creates the initial information.
    Each extracted location is evaluated in terms of its viability as histogram-based difference detection location.
    In addition, the normal distributions for the scene and transition deltas are fitted.

    Returns:
        Dict[gt.BlockNode, Tuple[bool, Tuple[float, float], Tuple[float, float]]]: The base information generated by this initial step.
    """

    #create data for the provided sequences
    self.histogram_report.evaluate_sequences(overwrite=True)

    # fit normal distributions for scenes and transitions

    node_score = {}
    for node, seq_data in self.histogram_report.sequence_data.items():
      log.debug(f"evaluating {node.name}")

      scene_deltas = []
      trans_deltas = []
      is_usable = False
      for seq_idx, seq in enumerate(seq_data):
        log.debug(f"evaluating {node.name} with seq {seq_idx}")
        scene_deltas += seq["scenes"]
        trans_deltas += seq["transitions"]

      scene_mu, scene_sigma = norm.fit(scene_deltas)
      trans_mu, trans_sigma = norm.fit(trans_deltas)

      # check distance
      if trans_mu - 0.75*trans_sigma > scene_mu:
        is_usable = True

      # check overlap
      overlap = norm.calculate_overlap(scene_mu, scene_sigma, trans_mu, trans_sigma)

      if overlap > 2/3:
        is_usable = False

      node_score[node] = [is_usable, (scene_mu, scene_sigma), (trans_mu, trans_sigma)]

      log.debug(node_score[node])

    self.node_score = node_score
    return node_score
  
  def configure(self) -> Dict[gt.BlockNode, Tuple[float, float]]:
    """
    Configures the threshold search space boundaries for each viable locations.
    Each location is annotated with a minimum and maximum boundary.

    Returns:
        Dict[gt.BlockNode, Tuple[float, float]]: keys are the BlockNodes of the model, the keys are a tuple containing the minimum and maximum boundaries for the threshold parameter.
    """
    if len(self.node_score) == 0:
      self.evaluate()

    # try to identify threshold boundaries for each location
    thres_boundaries = {}
    for node, seq_data in self.node_score.items():
      is_usable = seq_data[0]

      # skip unusable locations
      if not is_usable:
        continue

      scene_mu, scene_sigma = seq_data[1]
      trans_mu, trans_sigma = seq_data[2]

      # find threshold boundaries
      
      ## min boundary -> scene_mu + Y?
      if self.hist_size == "global":
        min_boundary = scene_mu
        if scene_mu + scene_sigma < trans_mu - trans_sigma:
          min_boundary = scene_mu + scene_sigma

        ## max boundary -> trans_mu - X?
        max_boundary = trans_mu
        if trans_mu - 0.9 * trans_sigma > min_boundary:
          max_boundary = trans_mu - 0.75 * trans_sigma
      else:
        min_boundary = max(trans_mu - (3*trans_sigma), 0)
        max_boundary = trans_mu - trans_sigma

      thres_boundaries[node] = (min_boundary, max_boundary)
    
    self.thres_boundaries = thres_boundaries
    return thres_boundaries
  
  def build(self, search_resolution:int=10, overwrite:bool=False) -> Dict[Tuple[float, float], Tuple[gt.BlockNode, float]]:
    """
    Creates the search space.
    Associates points in the search space (x: relative inference cost, y: accuracy) with configurations (location BlockNode, threshold parameter).

    Args:
        search_resolution (int, optional): the amount of threshold steps being evaluated for each node. Defaults to 10.
        overwrite (bool, optional): If the previous search space should be overwritten, if a search space of the same resolution has already been created. Defaults to False.

    Returns:
        Dict[Tuple[float, float], Tuple[gt.BlockNode, float]]: keys are the (rel. cost, accuracy) points in the search space, values are the associated configurations (node, threshold).
    """

    if not overwrite and len(self.search_space) != 0:
      if self.__search_space_res == search_resolution:
        return self.search_space
      
    if len(self.thres_boundaries) == 0:
      self.configure()

    #orig_acc = self.accuracy_report.results["top1_accuracy"]
    orig_accs = []
    raw_preds = []
    for seq in self.histogram_report.calibration_sequences:
      preds = self.analysis.keras_model.predict(seq.x)

      if len(preds.shape) >= 2 and preds.shape[-1] > 1:
        preds = np.argmax(preds, axis=-1)

      # handle binary classification
      if list(self.analysis.tasks.values())[-1] == prd.Task.BINARY_CLASSIFICATION:
        preds= [1 if val > 0.5 else 0 for val in preds] # convert to clear labels

      ground_truth = seq.y
      if len(seq.y.shape) >= 2 and seq.y.shape[-1] > 1:
        ground_truth = np.argmax(seq.y, axis=-1)
      orig_acc = acc_score(preds, ground_truth)
      raw_preds.append(preds)
      orig_accs.append(orig_acc)

    hists = {}
    for node in self.histogram_report.locations:
      for seq_id, seq in enumerate(self.histogram_report.calibration_sequences):
        hists[(node, seq_id)] = self.histogram_report.branch_models[node].predict(seq.x) #self.histogram_report.get_sequence_delta(node, seq_id)

    points = {}
    for node, (min_bound, max_bound) in self.thres_boundaries.items():
      #print(node, min_bound, max_bound)

      steps = np.linspace(min_bound, max_bound, num=search_resolution)

      for step in steps:
        node_accs = []
        node_cost = []

        node_scene_accs = []
        node_scene_cost = []
        for seq_id, seq in enumerate(self.histogram_report.calibration_sequences):
          samples = seq.x
          labels = seq.y

          preds = raw_preds[seq_id]
          
          ifms = hists[(node, seq_id)]

          for scene_based in [True, False]:
            acc, early_rate = _eval_histo_term(samples, labels, preds, ifms, step, scene_based=scene_based)
            mac_cost = early_rate * self.histogram_report.mac_costs[node] + (1-early_rate) * (1.0 + self.histogram_report.branch_mac_costs[node]  + self.histogram_report.dist_mac_costs[node])  # fixed this

            if not scene_based:
              node_accs.append(acc)
              node_cost.append(mac_cost)
            else:
              node_scene_accs.append(acc)
              node_scene_cost.append(mac_cost)

        avg_acc = np.mean(node_accs)
        avg_cost = np.mean(node_cost)

        points[(avg_cost, avg_acc)] = (node, step, False)

        avg_scene_acc = np.mean(node_scene_accs)
        avg_scene_cost = np.mean(node_scene_cost)

        points[(avg_scene_cost, avg_scene_acc)] = (node, step, True)

    self.search_space = points
    self.__search_space_res = search_resolution
    return points
  
  def search(self, mode:str="pareto", search_resolution:int=150) -> Tuple[Tuple[float, float], HistogramExitSolution]:
    """
    Searches for the solution with the given search mode.

    Args:
        mode (str, optional): _description_. Defaults to "pareto".
        search_resolution (int, optional): _description_. Defaults to 150.

    Raises:
        AttributeError: _description_

    Returns:
        Tuple[Tuple[float, float], HistogramExitSolution]: _description_
    """

    scatter = self.search_space
    
    if len(self.search_space) == 0:
      if self.__search_space_res != search_resolution:
        scatter = self.build(search_resolution=search_resolution)
        
    pareto = optimization.pareto_front(scatter.keys())

    optimum = None
    if mode == "pareto":
      optimum = optimization.find_pareto_optimum(pareto)
    elif mode == "min_cost":
      optimum = min(pareto, key=lambda x: x[0])
    elif mode == "max_acc":
      optimum = max(pareto, key=lambda x: x[-1])
    elif mode == "bend":
      optimum = optimization.find_pareto_dominant(pareto)
    else:
      raise AttributeError(f"mode {mode} not supported")
    
    node, threshold, scene_based = scatter[optimum]

    log.info(f"found possible solution using{mode} at:{optimum}, with config {node.name, threshold, scene_based}")

    # return solution object here
    #branch = self.histogram_report.branches[node]
    sol = HistogramExitSolution(
        location=node,
        threshold=threshold,
        scene_based=scene_based,
        #branch=branch,
        rel_cost=optimum[0],
        accuracy=optimum[1],
        rewriter=self
      )
    
    self.solutions[mode] = (optimum, sol)
    return optimum, sol
  
  def create_identifier(self) -> str:
    """
    Creates unique identifier for the rewriter object that depends on its parameters.
    This can be used to compare rewriters.

    Returns:
        str: A string that is unique to a rewriter with the given configuration.
    """

    return self.create_unique_identifier(
      self.train_dataset,
      self.validation_dataset,
      self.calibration_sequences,
      self.hist_size,
      self.histogram_report.cost_limit
    )
  
  def dump(self, folder_path: Union[str, pathlib.Path] = None) -> Dict[str, object]:
    """
    Writes the rewriter obejct to disk.

    Args:
        folder_path (Union[str, pathlib.Path], optional): The folder in which the files will be dumped. defaults to current working directory if None. Defaults to None.

    Returns:
        Dict[str, object]: A dict representation of the rewriter.
    """

    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    # TODO: take care of generated solutions
    sol_paths = {}
    solutions = {}
    for search_mode, data in self.solutions.items():
      coords, sol = data
      sol_path = folder_path / f"solution_histogram_termination-{self.create_identifier()}-{search_mode}"
      sol.dump(sol_path)
      sol_paths[search_mode] = self.__solution_path_template.replace("<hash>", self.create_identifier()).replace("<mode>", search_mode)
      solutions[search_mode] = sol.toDict(make_serializable=True, include_model=False)

    summary = {
      "report_type": "Histogram-based Adaptive Inference",
      "name" : self.analysis.name,
      "creation_date": str(self.analysis.creation_time),
      "histogram_resolution" : self.hist_size,
      "cost_limit" : self.histogram_report.cost_limit,
      "search_space" : [(coords) + (location.name, threshold, scene_based) for coords, (location, threshold, scene_based) in self.search_space.items()],
      "solution_paths" : sol_paths,
      "solutions" : solutions,
      "histogram_report" : self.histogram_report.access_id(),
    }

    file_name = self.__pickle_name_template.replace("<hash>", self.create_identifier())
    with open(folder_path / file_name, "w") as file:
      json.dump(summary, file)

    return summary
  
  def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> Tuple[str, str]:
    """
    Creates an HTML-based summary of the rewriter for the user.

    Args:
        folder_path (Union[str, pathlib.Path], optional): The folder in which the files will be dumped. defaults to current working directory if None. Defaults to None.

    Returns:
        Tuple[str, str]: The title and filename of the created report.
    """

    if folder_path is None:
      folder_path = pathlib.Path.cwd()

    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    with open(self.__summary_template_file, "r") as file:
      template = Template(file.read())

    summary = self.dump(folder_path=folder_path)

    hist_report = self.histogram_report
    summary["histogram_report"] = str(hist_report._html_name_template.replace("<hash>", hist_report.access_id()))
    #summary["search_space"] = {coords : (location.name, threshold) for coords, (location, threshold) in self.search_space.items()}

    #render summary
    html = template.render(summary=summary)

    #store summary html file to disk
    html_filename = self.__summary_name_template.replace("<hash>", self.create_identifier())
    html_path = folder_path / html_filename
    with open(html_path, "w") as file:
      file.write(html)
    return "Histogram-based adaptive Inference", html_filename
  
  @classmethod
  def create_unique_identifier(cls, train_dataset:dataset.DatasetReport, validation_dataset:dataset.DatasetReport, calibration_sequence : List[temp.CalibrationSequence], histo_config:str, cost_limit:float) -> str:
    """
    Class method to generate the identifier for not yet created rewriter objects.

    Args:
        train_dataset (dataset.DatasetReport): The training dataset.
        validation_dataset (dataset.DatasetReport): The validation dataset.
        calibration_sequence (List[temp.CalibrationSequence]): The calibration sequences.
        histo_config (str): The histogram resolution.
        cost_limit (float): The cost limit for evaluated locations.

    Returns:
        str: a unique identifier for a HistogramExitRewriter with the given config.
    """
    
    train_data_name = train_dataset.name
    valid_data_name = validation_dataset.name

    calib_length = len(calibration_sequence)
    
    unique_str = f"HistogramTermRewrite::datasets:{train_data_name},{valid_data_name}|calib:{calib_length}|histogram_resolution:{histo_config}|cost_limit:{cost_limit}"

    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the input string
    sha256_hash.update(unique_str.encode('utf-8'))
    hashed_representation = sha256_hash.hexdigest()

    return hashed_representation
  
  @classmethod
  def create_pass(cls, train_data:tf.data.Dataset, valid_data:tf.data.Dataset, calibration_sequence : List[temp.CalibrationSequence], histo_config:str, cost_limit:float, search_res:int=100, search_mode:str="pareto") -> Tuple[str, callable]:
    """
    Creates a rewrite pass for the ModelAnalysis optimization queue.
    Creates the rewriter object, constructs the search space and selects the solution based on the configured search mode.

    Args:
        train_data (tf.data.Dataset): The training dataset.
        valid_data (tf.data.Dataset): The validation dataset.
        calibration_sequence (List[temp.CalibrationSequence]): Calibration sequences that are similar in their behavior to the production environments.
        histo_config (str): resolution of the created histograms.
        cost_limit (float): relative maximum cost for the evaluation of the histogram creation locations.
        search_res (int, optional): The number of steps between the minimum and maximum threshold for each evaluated location. Defaults to 100.
        search_mode (str, optional): The search mode, Supported are: "pareto", "bend", "min_cost" and "max_acc". Defaults to "pareto".

    Returns:
        Tuple[str, callable]: The name of the pass and its callable function to execute the rewrite pass.
    """
    str_id:str = f"HistogramExitRewrite_{histo_config}_{cost_limit}"

    def rewrite(analysis:aNN.ModelAnalysis) -> HistogramExitSolution:
      """ The application of the rewrite pass to the given ModelAnalysis."""

      # create dataset reports
      train_data_report = dataset.DatasetReport.submit_to(analysis, lazy=False).with_config(name="train", modality=None).from_source(tf_dataset=train_data)

      valid_data_report = None
      if valid_data is not None:
        valid_data_report = dataset.DatasetReport.submit_to(analysis, lazy=False).with_config(name="valid", modality=None).from_source(tf_dataset=valid_data)

      rewriter = HistogramExitRewriter(
        analysis=analysis,
        train_dataset=train_data_report,
        validation_dataset=valid_data_report,
        calibration_sequences=calibration_sequence,
        histogram_size=histo_config,
        cost_limit=cost_limit
        )
      
      rewriter.evaluate()
      rewriter.configure()
      rewriter.build(search_resolution=search_res)
      return rewriter.search(mode=search_mode, search_resolution=search_res)[-1]

    return str_id, rewrite