import hashlib
import os
from typing import Tuple, Dict, List, Union
from typing_extensions import Self
import pathlib
import math
import logging as log
import pickle as pkl
from jinja2 import Template
import json

import tensorflow as tf
import numpy as np
#from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .base import Rewriter, Solution
from . import heterogeneous_platforms as hp

from ..reports import temporal as temp
from ..reports import dataset as data

from peax.components import graph_tools as gt
from peax.components import predictive as prd
from peax.components import resource as res

import peax.analysis as aNN
import peax.utils.optimization as optimization
import peax.utils.keras_graph as keras_graph
import peax.utils.normal_distribution as norm

def eval_dd_mode(y_true : np.array, preds, threshold, patience : bool = False, task:prd.Task=prd.Task.CLASSIFICATION) -> Dict[str, Union[float, np.ndarray]]:
  """
  Evaluates the performance of the Difference Detection or Temporal Patience (patience=True).
  For performance reasons you have to provide the raw predictions made by your classifiers to enable better reuse and faster search times.


  Args:
      y_true (np.array): the ground truth as classification vectors
      preds (_type_): the raw predictions of your model/classifiers, first dimension are the classifiers, second the samples, third the classes
      threshold (_type_): the decision threshold for your Early Exit
      patience (bool, optional): True, if you want to enable Temporal Patience and consider the class label for the change detection. Defaults to False.

  Returns:
      Dict[str, Union[float, np.ndarray]]: Dict with the results of the evaluation
  """
  if isinstance(preds, list):
    preds = np.array(preds)

  if task == prd.Task.CLASSIFICATION:
    if len(y_true.shape) > 1 and y_true.shape[-1] != 1:
      true_labels = np.argmax(y_true, axis=-1)
    else:
      log.warn("y_true was given as labels, untested!")
      true_labels = y_true

    ee_labels = np.argmax(preds[0:-1], axis=-1) # has additional dimension to address the different EEs
    fe_labels = np.argmax(preds[-1], axis=-1)
  elif task == prd.Task.BINARY_CLASSIFICATION:
    true_labels = y_true

    # need to fix this
    ee_labels = (preds[0:-1] > 0.0).astype(int) # early exit does not utilize sigmoid function, which moves the threshold to zero
    fe_labels = (preds[-1] > 0.5).astype(int) # final exit still has its sigmoid

    ee_labels = np.squeeze(ee_labels, axis=-1)
    fe_labels = np.squeeze(fe_labels, axis=-1)

  ref_label = fe_labels[0]
  raw_distances = []
  scene_distances = []
  transition_distances = []

  if isinstance(threshold, list):
    threshold.append(0.0)

  def get_curr_idx(curr_ee_labels, curr_fe_label) -> int:
    labels = list(curr_ee_labels)#
    labels.append(curr_fe_label)
    curr = np.where(labels == ref_label)[0][0]
    if curr == len(labels) - 1:
      curr = -1

    return curr
  
  curr_idx = 0
  if isinstance(threshold, float):
    curr_thres = threshold
  
  term_ids = []
  if patience:
    curr_idx = get_curr_idx(ee_labels[:,0], ref_label) #np.where(ee_labels[:,0] == ref_label)[0][0]
    if isinstance(threshold, float):
      curr_thres = threshold
    elif isinstance(threshold, list):
      curr_thres = threshold[curr_idx]
  
  ref_pos = preds[curr_idx][0]

  thres_preds = [ref_label]
  active = [1]

  prev_label = ee_labels[curr_idx][0]
  for i in range(1, fe_labels.shape[0]):

    curr_label = fe_labels[i]
    
    #only used for patience
    if curr_idx == -1:
      curr_ee_label = fe_labels[i]
    else:
      curr_ee_label = ee_labels[curr_idx][i]
    
    curr_pos = preds[curr_idx][i]
    distance = np.linalg.norm(ref_pos - curr_pos)
    raw_distances.append(distance)

    if patience:
      ref_label = curr_ee_label # ee_labels[curr_idx][i]

    activity = 0
    if distance >= curr_thres or (patience and curr_ee_label != prev_label):
      activity = 1
      ref_label = curr_label
      prev_label = curr_ee_label

      if patience:
        curr_idx = get_curr_idx(ee_labels[:,i], ref_label) #np.where(ee_labels[:,i] == ref_label)[0][0]
        if isinstance(threshold, float):
          curr_thres = threshold
        elif isinstance(threshold, list):
          curr_thres = threshold[curr_idx]

      

    thres_preds.append(ref_label)
    if activity == 0:
      term_ids.append(curr_idx) # can be 0,1,2, ... for EE, -1 for FE and needs value for majority vote
      scene_distances.append(distance)
    else:
      term_ids.append(-1)#(preds.shape[0]+1) #why are we going with this?
      transition_distances.append(distance)
    active.append(activity)

  evaluation_result = {}

  evaluation_result["accuracy"] = accuracy_score(true_labels, thres_preds)
  evaluation_result["precision"] = precision_score(true_labels, thres_preds, average="micro")
  evaluation_result["recall"] = recall_score(true_labels, thres_preds, average="micro")
  evaluation_result["f1"] = f1_score(true_labels, thres_preds, average="micro")
  
  active_rate = np.count_nonzero(active) / len(active)
  evaluation_result["early_termination"] = 1 - active_rate
  evaluation_result["distances"] = raw_distances
  evaluation_result["scene_delta"] = scene_distances
  evaluation_result["transition_delta"] = transition_distances
  evaluation_result["used_classifier"] = term_ids

  evaluation_result["confusion_matrix"] = confusion_matrix(true_labels, thres_preds)
  
  return evaluation_result

class TemporalExitSolution(Solution):
  """ Represents a solution that has been found be the TemporalExitRewriter.
  Contains the found model architecture, its individual parts and configurations and performance information. """

  recoms : List[gt.BlockNode]
  """ the attachment points of the added classifier branches """

  branchs : List[tf.keras.models.Model]
  """ the description of the attached branches as keras models """

  thresholds : List[float]
  """ the exit-wise thresholds of the early exit classifers that are required to terminate """

  modes : List[str]
  """ the exit-wise decision modes, either "DD" for Difference Detection and "TP" for Temporal Patience """

  model : tf.keras.Model
  """ The Early Exit Model that was created by the Rewrite operation. """

  rewriter : Rewriter
  """ the rewriter that created this solution """

  accuracy : float
  """ the accuracy that has been achieved on the calibration sequences provided by the rewriter """

  efficiency : float
  """ the efficiency that has been achieved on the calibration sequences provided by the rewriter relative to the MACs of the original model """
  
  search_config : str
  """ the search configuration that was used to create this result """

  task : prd.Task
  """ The task that the solution needs to perform """


  def __init__(self, recoms : List[gt.BlockNode], branchs : List[tf.keras.models.Model], thresholds : List[float], modes : List[str], search_config : str, accuracy:float, efficiency:float, rewriter : Rewriter) -> None:
    
    if not isinstance(recoms, list):
      recoms = [recoms]

    if not isinstance(branchs, list):
      branchs = [branchs]

    if not isinstance(thresholds, list):
      thresholds = [thresholds]

    if not isinstance(modes, list):
      modes = [modes]
    
    self.recoms = recoms
    self.branchs = branchs
    self.thresholds = thresholds
    self.modes = modes

    self.search_config = search_config
    self.accuracy = accuracy
    self.efficiency = efficiency
    self.rewriter = rewriter

    if self.rewriter is not None:
      self.task = list(rewriter.analysis.tasks.values())[0]

    self.branch_costs = {}
    for loc, branch_model in zip(recoms, branchs):
      self.branch_costs[loc] = 0
      for layer in branch_model.layers:
        self.branch_costs[loc] += res.get_layer_macs(layer=layer)

    self.subgraph_costs = {}
    start_point = gt.get_first_input_node(self.rewriter.analysis.architecture.block_graph)
    for loc in recoms:
      end_point = loc
      shortest_path = list(nx.shortest_path(
            self.rewriter.analysis.architecture.block_graph, start_point, end_point
        ))
      self.subgraph_costs[loc] = 0
      for block in shortest_path:
        self.subgraph_costs[loc] +=  block.macs

    self.ee_costs = {}
    for loc in recoms:
      self.ee_costs[loc] = self.branch_costs[loc] + self.subgraph_costs[loc]

    self.combined_costs = {}
    for loc in recoms:
      self.combined_costs[loc] = self.branch_costs[loc] + list(self.rewriter.analysis.compute.total_mac.values())[0]

    #build model
    backbone = self.rewriter.analysis.keras_model
    attach_layers = [gt.get_output_nodes(loc.subgraph)[0].keras.name for loc in self.recoms]

    self.model = keras_graph.attach_branches(original_model=backbone, attachment_layers=attach_layers, branches=branchs, reorder=True)

    return

    #evaluate, finetune

  def evaluate(self, data : Union[Tuple[List, List], Tuple[np.ndarray, List], tf.data.Dataset]) -> Dict[str, Union[float, np.ndarray]]:
    """Evaluates the performance of the found solution on the given data

    Args:
        data (Union[Tuple[List, List], Tuple[np.ndarray, List], tf.data.Dataset]): the input data, needs to contain samples and corresponding labels

    Returns:
        Dict[str, Union[float, np.ndarray]]: the evaluation result
    """

    patience = self.modes[0] == "TP" # need to fix this

    if len(data) == 2:
      x_inp = data[0]
      y_inp = data[-1]

    raw_preds = self.model.predict(x_inp)

    eval_result = eval_dd_mode(y_inp, raw_preds, self.thresholds[0], patience, task=self.task)
    eval_result["mean_macs"] = eval_result["early_termination"] * list(self.ee_costs.values())[0] + (1-eval_result["early_termination"])*list(self.combined_costs.values())[0]
    eval_result["relative_macs"] = eval_result["mean_macs"] / list(self.rewriter.analysis.compute.total_mac.values())[0]
    if self.task == prd.Task.CLASSIFICATION:
      early_out = np.argmax(raw_preds[0], axis=-1)
      final_out = np.argmax(raw_preds[-1], axis=-1)
      y_inp = np.argmax(y_inp, axis=-1)
    elif self.task == prd.Task.BINARY_CLASSIFICATION:
      early_out = (raw_preds[0] > 0.0).astype(int)
      final_out = (raw_preds[-1] > 0.5).astype(int)
    

    early_result = {}
    early_result["accuracy"] = accuracy_score(y_inp, early_out)
    early_result["precision"] = precision_score(y_inp, early_out, average="micro")
    early_result["recall"] = recall_score(y_inp, early_out, average="micro")
    early_result["f1"] = f1_score(y_inp, early_out, average="micro")
    early_result["confusion_matrix"] = confusion_matrix(y_inp, early_out)
    eval_result["early_exit"] = early_result

    final_result = {}
    final_result["accuracy"] = accuracy_score(y_inp, final_out)
    final_result["precision"] = precision_score(y_inp, final_out, average="micro")
    final_result["recall"] = recall_score(y_inp, final_out, average="micro")
    final_result["f1"] = f1_score(y_inp, final_out, average="micro")
    final_result["confusion_matrix"] = confusion_matrix(y_inp, final_out)
    eval_result["final_exit"] = final_result

    return eval_result

    #raise NotImplementedError

  def finetune(self):
    """ Finetuning of the early exit model and the accompanying decision mechanism, currently not implemented

    Raises:
        NotImplementedError: Function is currently not implemented
    """

    raise NotImplementedError

  def dump(self, path : Union[str, pathlib.Path]) -> Dict[str, object]:
    """Writes the solution to disk

    Args:
        path (Union[str, pathlib.Path]): the target folder

    Returns:
        Dict[str, object]: The solution in Dict form
    """
    if isinstance(path, str):
      path = pathlib.Path(path)


    name = f"temporal_solution-{self.search_config}"

    path = path / name
    path.mkdir(parents=True, exist_ok=True)

    info = self.toDict()
    info["rewriter"] = self.rewriter.create_identifier()
    info["locations"] = [recom.name for recom in self.recoms]
    
    info["branch_configs"] = []
    for branch in self.branchs:
      branch_path = path / f"{branch.name}.keras"
      tf.keras.models.save_model(branch, branch_path)
      info["branch_configs"].append(str(branch_path))

    tf.keras.models.save_model(self.model, path / "full.keras")

    with open(path / f"{name}.pkl", "wb") as file:
      pkl.dump(info, file)

    return info
  
  def split(self) -> List[tf.keras.Model]:
    return keras_graph.split_model(self.model, self.recoms)
    '''eenn_model = tf.keras.models.clone_model(self.model)
    eenn_model.build(self.model.input_shape)
    eenn_model.set_weights(self.model.get_weights())

    attach_layers = [gt.get_output_nodes(loc.subgraph)[0].keras.name for loc in self.recoms]

    inp_layers = eenn_model.inputs

    connection_layers = []
    for name in attach_layers:
        connection_layers.append(eenn_model.get_layer(name).output)

    exit_names = eenn_model.output_names
    exit_layers = []
    for name in exit_names:
        exit_layers.append(eenn_model.get_layer(name).output)

    subgraph_models = []
    for idx, (connect, out_classifier) in enumerate(zip(connection_layers, exit_layers)):
      #TODO: need to check if order of connects and out_classifiers fits
      print(connect, out_classifier)
      out_layers = [connect, out_classifier]

      sub_model = tf.keras.Model(inputs=inp_layers, outputs=out_layers, name=f"{self.model.name}-{idx}")
      new_submodel = tf.keras.models.clone_model(sub_model)
      new_submodel.build(sub_model.input_shape)
      new_submodel.set_weights(sub_model.get_weights())

      subgraph_models.append(new_submodel)

      connection_layer = eenn_model.get_layer(connect.node.layer.name)
      inp_layers = [tf.keras.layers.Input(shape=connection_layer.output_shape, tensor=connection_layer.output)]

    #last submodel
    final_submodel = tf.keras.Model(inputs=inp_layers, outputs=[exit_layers[-1]], name=f"{self.model.name}-{idx+1}")

    new_submodel = tf.keras.models.clone_model(final_submodel)
    new_submodel.build(final_submodel.input_shape)
    new_submodel.set_weights(final_submodel.get_weights())
    subgraph_models.append(new_submodel)

    return subgraph_models'''

    
  def toDict(self) -> Dict[str, object]:
    info = {
      "locations" : self.recoms,
      "branch_configs" : self.branchs,
      "thresholds" : self.thresholds,
      "modes" : self.modes,
      "rewriter" : self.rewriter,
      "search_config" : self.search_config,
      "accuracy" : self.accuracy,
      "efficiency" : self.efficiency,
    }

    return info
  
  def to_analysis(self):
    submodels = self.split()

    analysis_list = []

    for submodel in submodels:
      new_analysis = aNN.ModelAnalysis(
        model=submodel,
        name=f"{self.rewriter.analysis.name}_TEERewrite:{submodel.name}",
        mac_estimators=self.rewriter.analysis.compute.mac_estimators,
        cache_dir=self.rewriter.analysis.cache_dir
      )

      analysis_list.append(new_analysis)

    return analysis_list


class TemporalExitRewriter(Rewriter):
  """ 
  The Rewriter component to turn the orginal model into a Temporal Early Exit Neural Network.
  Utilizes information from an TemporalExitDecisionReport that will be created if it does not already exists.
  Uses the distribution of detected sequence changes to limit the search space to sensible values.
  Currently only able to create Difference Detection Early Exit Neural Networks with a single Early Exit.
  WIP: Temporal Patience Early Exit Neural Networks.
  """

  analysis: aNN.ModelAnalysis
  """ the full model analysis that is the base for this optimization run """
  
  temporal_report : temp.TemporalExitDecisionReport
  """All information that has been extracted with regards to the behavior of the early exit branches over time."""

  extracted_info : Dict
  """ the information that has been extracted from the temporal report """

  thresholds : Dict
  """ the threshold configurations that have been found """

  trial_evaluation : Dict
  """ the evaluation results for different threshold configurations of each branch location """

  solutions : Dict[str, TemporalExitSolution]
  """ already found solutions """

  def __init__(self,
               analysis: aNN.ModelAnalysis,
               
               train_dataset : data.DatasetReport,
               validation_dataset : data.DatasetReport,
               #test_dataset : data.DatasetReport,

               calibration_sequences : List[temp.CalibrationSequence],

               search_config : str = "small") -> None:

    self.temporal_report = temp.TemporalExitDecisionReport.submit_to(
        analysis=analysis,
        lazy=False
      ).with_config(
          calibration_sequences=calibration_sequences,
          train_dataset=train_dataset,
          validation_dataset=validation_dataset,
          #test_dataset = test_dataset,
          search_config=search_config
        )
    
    self.extracted_info = {}
    self.trial_evaluation = {}
    self.thresholds = {}
    self.solutions = {}

    super().__init__(analysis)

  def evaluate(self) -> Dict[gt.BlockNode, Dict[str, object]]:
    """Collects raw evaluation data that will be used for the search.

    Returns:
        Dict[gt.BlockNode, Dict[str, object]]: Evaluation data for the possible Early Exit attachment points.
    """
    if len(self.temporal_report.ee_performance) == 0:
      self.temporal_report.evaluate()

    eval_result = self.temporal_report.ee_performance

    extracted_info = {}
    for recom, info in eval_result.items():
      # eval if DD or TP are more suitable
      score_dd = 0
      score_tp = 0
      score_recom = 0
      ## low accuracy: better suited for DD, otherwise TP
      if info["relative"]["accuracy"] < -0.1:
        score_dd += 1
      else:
        score_tp += 1

      if abs(info["relative"]["accuracy"]) < 0.05 or info["relative"]["accuracy"] > 0:
        score_tp += 5

      seq_perfs = []
      for seq in info["sequences"]:
        seq_perf = {}
        ## fp values are created if the label of the classifier changes incorrectly, relevant for TP performance
        fp_rate = len(seq["fp"]) / len(seq["raw_predictions"])
        tp_rate = len(seq["tp"]) / len(seq["raw_predictions"])
        ## fn values are created, if label change does not occure when it should, also relevant for TP performance
        fn_rate = len(seq["fn"]) / len(seq["raw_predictions"])
        tn_rate = len(seq["tn"]) / len(seq["raw_predictions"])
        
        # construct histograms and fit distribtuions for for tp, tn, fp, fn
        histograms = {}
        distr_params = {}
        for val in ["tp", "tn", "fp", "fn"]:
          histograms[val] = np.histogram(seq[val], density=True) # counts, bin_edges

          # fit normal distributions to them
          fit_mu, fit_sigma = norm.fit(seq[val])
          distr_params[val] = [fit_mu, fit_sigma]

        overlaps = {}
        tn_sigma = distr_params["tn"][-1]
        tn_mu = distr_params["tn"][0]
        for val, (mu, sigma) in distr_params.items():
          #if val == "tn":
          #  continue
        
          overlaps[val] = norm.calculate_overlap(tn_mu, tn_sigma, mu, sigma) #self._normal_overlap(tn_mu, tn_sigma, mu, sigma)
        
        # would temporal decision work for this exit on this sequence?
        ## compare the distributions for tp, tn, fp, fn; check for signifant differences
        # updated with overlap based metrics
        if overlaps["tp"] < 0.8 and overlaps["fn"] < 0.8:
          score_recom += 1
        else:
          score_recom -= 1

        if fp_rate > tp_rate or (overlaps["tp"] >= 0.5 and overlaps["fp"] >= 0.5):
          score_dd += 1
        else:
          score_tp += 1

        if fn_rate > tn_rate or (overlaps["fn"] >= 0.5):
          score_dd += 1
        else:
          score_tp += 1

        seq_perf["hists"] = histograms
        seq_perf["distributions"] = distr_params
        seq_perf["overlap"] = overlaps
        seq_perfs.append(seq_perf)
      
      mechanism = "DD"
      if score_tp >= score_dd:
        mechanism = "TP"

      extracted_info[recom] = {
        "mechanism" : mechanism,
        "scores_mech" : [score_dd, score_tp],
        "recommended" : score_recom > 0,
        "scores_recom" : score_recom,
        "seq_perfs" : seq_perfs,
      }

    self.extracted_info = extracted_info
    return extracted_info

  def configure(self, search_resolution:int=50, thres_min:int=None, thres_max:int=None) -> Dict[gt.BlockNode, Dict[float, Dict[str, object]]]:
    """
    Creates more detailed evaluation data for the combination of 
    Early Exit configuration and its sequence change detection thresholds.

    Args:
        search_resolution (int, optional): Number of threshold configurations that will be evaluated per EE. Defaults to 50.

    Returns:
        Dict[gt.BlockNode, Dict[float, Dict[str, object]]]: Detailed threshold evaluation results
    """
    if self.extracted_info is None:
      log.info("temporal report has not yet been evaluated. Evaluation started...")
      self.evaluate()

    eval_result = self.temporal_report.ee_performance
    sequences = self.temporal_report.calibration_sequences
    self.trial_evaluation = {}

    


    final_predictions = []
    for seq in sequences:
      #generating final classifier output predictions
      if isinstance(seq, temp.TFCalibrationSequence):
        x = seq.get_x()
      else:
        x = seq.x
      final_predictions.append(self.analysis.keras_model.predict(x))

    thres_ranges = {}
    log.debug("got information, need to find thresholds")
    for recom, info in self.extracted_info.items():

      # do not evaluate options that are not recommended due to poor performance on the calibration data
      if info["recommended"] == False:
        continue

      branch_macs = self.temporal_report.ee_report.exit_costs[recom]

      early_macs = self.temporal_report.ee_performance[recom]["macs"]
      full_macs = branch_macs + self.temporal_report.ee_report.model_macs
      orig_macs = self.temporal_report.ee_report.model_macs

      self.trial_evaluation[recom] = {}
      #find threshold
      deltas = {}
      total_length = np.sum(len(sequence["raw_predictions"]) for sequence in eval_result[recom]["sequences"])
      for val in ["tp", "tn", "fp", "fn"]:
        all_seq_lists = [sequence[val] for sequence in eval_result[recom]["sequences"]]
        flattened_list = [item for sublist in all_seq_lists for item in sublist]
        deltas[val] = flattened_list

      total_length = sum(len(deltas[val]) for val in deltas.keys())
      #thres_min = min(min(deltas[val]) for val in deltas.keys()) #np.min([val["distributions"]["tn"][0]-val["distributions"]["tn"][-1] for val in info["seq_perfs"]])
      #thres_max = max(max(deltas[val]) for val in deltas.keys()) #np.max([val["distributions"]["fp"][0]-val["distributions"]["fp"][-1] for val in info["seq_perfs"]])
      
      #if thres_min is None:
      thres_min = np.mean([val["distributions"]["tn"][0]-2*val["distributions"]["tn"][-1] for val in info["seq_perfs"]])
      #if thres_max is None:
      thres_max = np.mean([val["distributions"]["tp"][0]-1.5*val["distributions"]["tp"][-1] for val in info["seq_perfs"]])

      if thres_min < 0:
        thres_min = 0 #np.mean([val["distributions"]["tn"][0] for val in info["seq_perfs"]]) #0
        thres_max = np.mean([val["distributions"]["tp"][0]+1*val["distributions"]["tp"][-1] for val in info["seq_perfs"]]) #np.mean([val["distributions"]["tp"][0]-1.5*val["distributions"]["tp"][-1] for val in info["seq_perfs"]])

      if thres_max < thres_min:
        thres_min = np.mean([val["distributions"]["tn"][0] for val in info["seq_perfs"]])
        thres_max = np.mean([val["distributions"]["tp"][0] for val in info["seq_perfs"]])

      # TODO: check if min is larger than max
      thres_range = (thres_min, thres_max)
      thres_ranges[recom] = thres_range

      trial_results = {}
      
      for trial in np.linspace(thres_min, thres_max, num=search_resolution):
        tp_miss = np.count_nonzero(deltas["tp"] <= trial)
        fn_miss = np.count_nonzero(deltas["fn"] <= trial)

        tn_detect = np.count_nonzero(deltas["tn"] <= trial)
        fp_detect = np.count_nonzero(deltas["fp"] <= trial)

        trial_results[trial] = {
          "tp_miss_total" : tp_miss,
          "fn_miss_total" : fn_miss,
          "tn_detect_total" : tn_detect,
          "fp_detect_total" : fp_detect,
          "tp_miss_share" : tp_miss/len(deltas["tp"]),
          "fn_miss_share" : fn_miss/len(deltas["fn"]),
          "tn_detect_share" : tn_detect/len(deltas["tn"]),
          "fp_detect_share" : fp_detect/len(deltas["fp"]),
          "early_termination_rate" : (tp_miss + fn_miss + tn_detect + fp_detect)/total_length,
          "detection_rate" : (tn_detect + fp_detect)/total_length
        }

        #eval performance on sequence:
        trial_results[trial]["evaluation"] = []
        for idx, seq_eval in enumerate(eval_result[recom]["sequences"]):
          seq = sequences[idx]

          y_true = seq.y
          early_preds = seq_eval["raw_predictions"]
          final_preds = final_predictions[idx]

          preds = np.stack([early_preds, final_preds])
          task = list(self.analysis.tasks.values())[0]

          # eval termination algo here dd/tp_eval(y_true, preds, trial)
          if info["mechanism"] == "DD":
            log.debug(f"evaling {recom} with {trial} in DD mode")
            trial_eval = eval_dd_mode(y_true, preds, trial, task=task)
          else:
            log.debug(f"evaling {recom} with {trial} in TP mode")
            trial_eval = eval_dd_mode(y_true, preds, trial, patience=True, task=task)
          
          trial_eval["cost"] = ((trial_eval["early_termination"] * early_macs + (1-trial_eval["early_termination"]) * full_macs))/orig_macs
          trial_results[trial]["evaluation"].append(trial_eval)

        self.trial_evaluation[recom] = trial_results

        total_accuracy = 0
        total_cost = 0
        total_len = 0
        for idx, eval in enumerate(trial_results[trial]["evaluation"]):
          total_len += len(sequences[idx].y)

          total_accuracy += eval["accuracy"] * len(sequences[idx].y)
          total_cost += eval["cost"] *  len(sequences[idx].y)

        trial_results[trial]["total_cost"] = total_cost / total_len
        trial_results[trial]["total_accuracy"] = total_accuracy / total_len
    
    return self.trial_evaluation
  
  def search(self, mode:str="pareto") -> TemporalExitSolution:
    """ Finds a solution within the search space.

    Args:
        mode (str, optional): The search mode that will be used. Can be "pareto", "min_cost", "max_acc", "bend". Defaults to "pareto".

    Raises:
        AttributeError: Raises this error, if mode is not one of ["pareto", "min_cost", "max_acc", "bend"]

    Returns:
        TemporalExitSolution: The found solution, not yet fine-tuned.
    """

    scatter = self.create_search_space()
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
    
    recom, threshold = scatter[optimum]
    mecha = self.extracted_info[recom]["mechanism"]
    branch = self.temporal_report._remove_activation(self.temporal_report.exit_submodels[recom])

    sol = TemporalExitSolution(
      recoms=recom,
      branchs=branch,
      thresholds=threshold,
      modes=mecha,
      search_config=mode,
      accuracy=optimum[-1],
      efficiency=optimum[0],
      rewriter=self
      )

    self.solutions[mode] = sol

    return sol


  def create_search_space(self) -> Dict[Tuple[float, float], Tuple[gt.BlockNode, float]]:
    """
    Maps the solutions into a 2D space. X = cost relative to orginal model, Y = accuracy
    TODO: extend to support multiple Early Exits in the architectures

    Returns:
        Dict[Tuple[float, float], Tuple[gt.BlockNode, float]]: the search space, keys are the positions, values are the selected EE and its threshold configuration.
    """
    if len(self.trial_evaluation) == 0:
      self.configure()

    # map out search space to find Pareto front
    scatter = {}
    for recom, eval in self.trial_evaluation.items():
      for trial, info in eval.items():
        acc = info["total_accuracy"]
        cost = info["total_cost"]

        val = (recom, trial)
        scatter[cost, acc] = val

    return scatter

  def create_identifier(self) -> str:
    descriptor = f"TemporalExitRewriter:{self.analysis.keras_model.name}, {self.temporal_report}"
    sha256_hash = hashlib.sha256()
        
    # Update the hash object with the input string
    sha256_hash.update(descriptor.encode('utf-8'))
    hashed_representation = sha256_hash.hexdigest()

    return hashed_representation
  
  def dump(self, folder_path: Union[str, pathlib.Path] = None):
    info ={
      "report_type": "Temporal Exits",
      "name": self.analysis.name,
      "creation_date": str(self.analysis.creation_time),
      "temporal_report" : self.temporal_report.dump(folder_path=folder_path),
      "extracted" : {key.name: value for key, value in self.extracted_info.items()},
      "trial_runs" : {key.name: value for key, value in self.trial_evaluation.items()},
      "solutions" : {key: value.dump(path=folder_path) for key, value in self.solutions.items()},
    }

    with open(folder_path / f"optimizer_temporal_exits_{self.create_identifier()}.json", "wb") as file:
      pkl.dump(info, file)

    return info
  
  def render_summary(self, folder_path: Union[str, pathlib.Path] = None):

    _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'

    if folder_path is None:
        folder_path = pathlib.Path.cwd()

    if isinstance(folder_path, str):
        folder_path = pathlib.Path(folder_path)

    with open(_template_path / "temporal_exiting_rewriter.html", "r") as file:
        template = Template(file.read())

    summary = self.dump(folder_path=folder_path)
    html = template.render(summary=summary)
  
    html_filename = f"optimization_temporal_early_exiting_{self.create_identifier()}.html"
    html_path = folder_path / html_filename
    with open(html_path, "w") as file:
      file.write(html)

    return (
        "Temporal Exiting",
        html_filename,
    )
  
  @classmethod
  def create_pass(cls, train_ds:tf.data.Dataset, valid_ds:tf.data.Dataset, calibration_sequences : List[temp.CalibrationSequence], search_config : str = "small", search_mode:str="pareto") -> Tuple[str, callable]:

    str_id = f"TemporalEarlyExitRewrite_{search_config}_{search_mode}_{calibration_sequences}"

    def rewrite(analysis : aNN.ModelAnalysis) -> TemporalExitSolution:

      train_data_report = data.DatasetReport.submit_to(analysis, lazy=False).with_config(name="train", modality=None).from_source(tf_dataset=train_ds)

      valid_data_report = None
      if valid_ds is not None:
        valid_data_report = data.DatasetReport.submit_to(analysis, lazy=False).with_config(name="valid", modality=None).from_source(tf_dataset=valid_ds)

      temp_rewriter = TemporalExitRewriter(
        analysis=analysis,
        train_dataset=train_data_report,
        validation_dataset=valid_data_report,
        calibration_sequences=calibration_sequences,
        search_config=search_config
        )

      temp_rewriter.evaluate()

      temp_rewriter.configure()

      temp_rewriter.create_search_space()

      sol = temp_rewriter.search(mode=search_mode)

      return sol

    return str_id, rewrite