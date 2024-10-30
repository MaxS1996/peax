import copy
import glob
import numpy as np
from typing import Dict, List, Tuple, Union, Set
from numpy.core.multiarray import array as array
from typing_extensions import Self
import pathlib
import os
import logging as log
import json
import pickle

from sklearn.metrics import confusion_matrix

from jinja2 import Template
import tensorflow as tf

from peax.reports import dataset

from peax.components import predictive as prd

from . import base
from . import early_exit as ee
from . import accuracy as acc

import peax.analysis as aNN

class CalibrationSequence:
  """This class is a wrapper for the calibration sequence data required to run the TemporalExit Evaluation
  """

  x : np.array
  """The input samples of the sequence"""

  y : List[np.array]
  """The labels for the samples of the sequence"""

  def __init__(self, x : np.array, y : List[np.array]) -> None:
    """
    The constructor for the calibration sequence class.

    Args:
        x (np.array): the samples of the sequence
        y (List[np.array]): The labels of the sequence
    """
    assert len(x) == len(y), "x and y must have the same length"

    self.x = x
    self.y = y
    pass

  @staticmethod
  def _check_correlation(sequence: np.array, threshold: float = 1.0) -> bool:
    """
    Tries to figure out, if the samples in the sequence are correlated

    Args:
        sequence (np.array): the samples of the calibration sequence
        threshold (float, optional): the similarity threshold. Defaults to 1.0.

    Returns:
        bool: True, if correlated, False if unsure
    """
    '''sequence = sequence / sequence.max()

    # Iterate through the sequence of samples
    for i in range(len(sequence) - 1):
      # Calculate the Manhattan distance between subsequent samples
      dist = np.sum(np.abs(sequence[i] - sequence[i+1]))
      # If the distance is greater than the threshold, they're not similar
      if dist < threshold:
        return True
    # If the loop completes, all samples were similar
    return False'''
    #sequence = sequence / sequence.max()
    distances = np.sum(np.abs(np.diff(sequence)), axis=1)
    return np.any(distances < threshold)
  
  def is_correlated(self) -> bool:
    """
    checks, if the samples of the sequence are correlated by calculating the Manhattan distance of subsequent samples and comparing it to a threshold.

    Returns:
        bool: True, if correlated, False if not or inconclusive
    """
    return self._check_correlation(self.x, 0.2)
  
  def contains_transitions(self) -> bool:
    """
    checks, if the necessary transitions are contained in the sequence

    Returns:
        bool: True, if contains transitions
    """

    y = np.argmax(self.y, axis=-1) if self.y.ndim > 1 else self.y
    return np.unique(y).size > 1
  
  def transitions(self) -> List[int]:
    """
    Returns indicies of the contained transitions

    Returns:
        List[int]: A list of the indicies on which a change in label between subsequent samples has been detected.
    """    
    '''indices = list()

    if len(self.y.shape) > 1:
      y = np.argmax(self.y, axis=-1)
    else:
      y = self.y
    for i in range(len(y) - 1):
      if y[i] != y[i+1]:
        indices.append(i)

    return indices'''
  
    y = np.argmax(self.y, axis=-1) if self.y.ndim > 1 else self.y
    #return (np.where(np.diff(y) != 0)[0] - 1).tolist()
    return (np.where(np.diff(y) != 0)[0]).tolist()
  
  def scenes(self) -> List[Tuple[int, int]]:
    """
    Returns the start and end indicies as tuples for all scenes that have been detected in the sequence.

    Returns:
        List[Tuple[int, int]]: List of the detected scene indicies. A scene is represented by a tuple of start and end index
    """
    trans = self.transitions()
    #add indices for start and end of sequence
    trans = [0] + trans + [len(trans)-1]

    scenes = []
    for i in range(len(trans)-1):
      scenes.append((trans[i], trans[i+1]))

    return scenes
  
  def calculate_deltas(self) -> List[float]:
    """
    Calculates the distance between subsequent samples.

    Returns:
        List[float]: all distances of samples to their predecessors
    """
    seq = self.x
    seq = seq / seq.max()

    deltas = []
    # Iterate through the sequence of samples
    for i in range(len(self.x) - 1):
      # Calculate the Manhattan distance between subsequent samples
      dist = np.sum(np.abs(self.x[i] - self.x[i+1]))
      deltas.append(dist)
    return deltas
  
  def to_dict(self) -> Dict[str, object]:
    """
    Converts the calibration sequence object into a dict

    Returns:
        Dict[str, object]: all important members of the object as a dict
    """

    '''if len(self.y) > 0 and isinstance(self.y[0], np.ndarray):
      y = [a.tolist() for a in self.y]
    else:
      y = self.y'''

    data = {
      #"x" : self.x.tolist(),
      #"y" : y,
      "is_correlated" : self.is_correlated(),
      "contains_transitions" : self.contains_transitions(),
    }

    return data
  
  def __len__(self):
    return len(self.y)
  
class TFCalibrationSequence(CalibrationSequence):
  """
  Experimential class to enable PEAX to handle calibration sequences that would not fit into (GPU) memory.
  This will be enabled by using tf.data.Datasets to store the data, instead of the standard numpy-based implementation.
  """

  dataset : tf.data.Dataset
  batch_size : int
  __length : int

  def __init__(self, dataset: tf.data.Dataset, batch_size:int=1) -> None:
    self.dataset = dataset
    self.batch_size = batch_size

    self.__length = None

  @staticmethod
  def _check_correlation(dataset: tf.data.Dataset, threshold: float = 1.0) -> bool:
    # Use TensorFlow's tensor operations instead of NumPy's array operations
    dataset = dataset.batch(2)
    for elem in dataset:
      distances = tf.abs(elem[0][0] - elem[0][1])
      if bool(tf.reduce_any(distances < threshold).numpy()):
        return True
    return False

  def get_x(self, batch_size:int=None) -> tf.data.Dataset:
    if batch_size is None:
      batch_size = self.batch_size
    return self.dataset.map(lambda x, y: x).batch(batch_size).prefetch(tf.data.AUTOTUNE)

  @property
  def x(self) -> tf.data.Dataset:
    self.get_x(batch_size=1)

  def get_y(self, batch_size:int=None) -> tf.data.Dataset:
    if batch_size is None:
      batch_size = self.batch_size
    return self.dataset.map(lambda x, y: y).batch(batch_size).prefetch(tf.data.AUTOTUNE)

  @property
  def y(self) -> tf.data.Dataset:
    self.get_y(batch_size=1)
  
  def is_correlated(self) -> bool:
    return self._check_correlation(self.dataset, 0.2)
  
  def contains_transitions(self) -> bool:
    # Use TensorFlow's tensor operations instead of NumPy's array operations
    #labels = self.dataset.map(lambda x, y: y)
    prev_label = None
    for idx, (sample, label) in enumerate(self.dataset):
        if prev_label is not None and tf.reduce_all(tf.not_equal(label, prev_label)).numpy():
            log.debug(f"found transition at idx {idx}")
            return True
        prev_label = label

    return False

  def transitions(self):
    # Use TensorFlow's tensor operations instead of NumPy's array operations
    log.warn("the TFCalibrationSequence.transitions function behaves differently than the numpy-based implementation, it returns the samples and labels instead of the indicies!")
    paired_data = tf.data.Dataset.zip((self.dataset, self.dataset.skip(1))).prefetch(tf.data.AUTOTUNE)
    for (x1, y1), (x2, y2) in paired_data:
      if tf.reduce_all(tf.not_equal(y1, y2)).numpy():
        yield (x1, x2), (y1, y2)
    
  def transitions_list(self):
    return list(self.transitions())
  
  def scenes(self, prune_start:int=0, prune_end:int=0, min_scene_length:int=1, max_scene_length:int=15_000):
    log.warn("the TFCalibrationSequence.scenes function behaves differently than the numpy-based implementation, it returns the samples and labels instead of the indicies!")
   
    current_scene = []
    current_label = None
    for sample, label in self.dataset:
        if current_label is None:
          current_scene = [sample.numpy()]
          current_label = label
          continue
        if tf.reduce_all(tf.not_equal(label, current_label)) or len(current_scene) > max_scene_length:
            if current_scene:
                if len(current_scene) >= min_scene_length + prune_start + prune_end:
                  if prune_end == 0:
                    return_scene = current_scene[prune_start:]
                  else:
                    return_scene = current_scene[prune_start:-prune_end]
                  yield return_scene, np.stack([current_label.numpy()] * (len(return_scene)))
            current_scene = [sample.numpy()]
            current_label = label
        else:
            current_scene.append(sample.numpy())
    if current_scene:
        if len(current_scene) >= min_scene_length + prune_start + prune_end:
          if prune_end == 0:
            return_scene = current_scene[prune_start:]
          else:
            return_scene = current_scene[prune_start:-prune_end]
          yield return_scene, np.stack([current_label.numpy()] * (len(return_scene)))
          #yield current_scene[prune_start:-prune_end], [current_label.numpy()] * (len(current_scene) - prune_start - prune_end)
  
  def calculate_deltas(self) -> tf.data.Dataset:
    # Use TensorFlow's tensor operations instead of NumPy's array operations
    seq = self.dataset.map(lambda x, y: x)
    seq = seq / tf.reduce_max(seq)
    deltas = tf.abs(seq[:-1] - seq[1:])
    return deltas
  
  def to_dict(self) -> Dict[str, object]:
    # Convert the dataset to a dictionary
    data = {
        #"x": self.dataset.map(lambda x, y: x).numpy().tolist(),
        #"y": self.dataset.map(lambda x, y: y).numpy().tolist(),
        "is_correlated": self.is_correlated(),
        "contains_transitions": self.contains_transitions(),
    }
    return data

  def __len__(self):
    log.warn("TFCalibrationSequence objects cannot determine their length for performance reasons!")

    if self.__length is None:
      count = 0
      for element in self.dataset:
        count += 1
      self.__length = count
    
    return self.__length

class TemporalExitDecisionReportSubmitter(base.ReportSubmitter):
  """class that is used for syntactic sugaring.
    Allows developers to create an TemporalExitDecisionReport with:
    'TemporalExitDecisionReport.submit_to(analysis).with_config(calibration_sequences:List[CalibrationSequence], search_config="large")'
  """
  def __init__(self, analysis: aNN.ModelAnalysis, lazy: bool = False):
    """
    Creates the Submitter object, not recommeded to use its constructor directly, instead go through
    TemporalExitDecisionReport.submit_to(analysis, ...).with_config(...) to submit a new report to the ModelAnalysis

    Args:
        analysis (aNN.ModelAnalysis): the ModelAnalysis to which the report(er) will be assigned
        lazy (bool, optional): The submission behavior. lazy will only create the report if it is required, otherwise it will be generated immediately.
            Defaults to False.
    """
    super().__init__(analysis, lazy)

  def with_config(self, calibration_sequences:List[CalibrationSequence], train_dataset: dataset.DatasetReport,
              validation_dataset: dataset.DatasetReport=None,search_config:str="large") -> Union[str, base.Report]:
    """
    Creates submission to the ModelAnalysis object.
    If configures as lazy, the unique ID of the report will be returned, but it will not yet be created.
    Otherwise the TemporalExitReport instance will be created and returned.

    Args:
        calibration_sequences (List[CalibrationSequence]): the calibration sequences that will be used during the evaluation
        train_dataset (dataset.DatasetReport): the training dataset used during the evaluation
        validation_dataset (dataset.DatasetReport, optional): the validation dataset used during the evaluation. Defaults to None.
        search_config (str, optional): search config for the construction of the early exit branches. Defaults to "large".

    Returns:
        Union[str, base.Report]: _description_
    """
    td_reporter, td_uid = TemporalExitDecisionReport.closure(create_id=True, calibration_sequences=calibration_sequences, train_dataset=train_dataset, validation_dataset=validation_dataset, search_config=search_config)

    self.analysis.submit_reporter(td_uid, td_reporter)

    if self.lazy:
      return td_uid
    else:
      #self.analysis.create_report(ee_uid, ee_reporter)
      return self.analysis.access_report(td_uid)

class TemporalExitDecisionReport(base.Report):
  """
  Report to evaluate suitability of using Early Exits with a temporal decision mechanism to improve the efficiency of the neural network during inference.
  CAN ONLY BE USED IF THEIR IS CORRELATION/TRANSINFORMATION BETWEEN SUBSEQUENT SAMPLES OF THE INPUT.
  """

  __pkl_name = "report_temporal_decision_<hash>.pkl"
  """The standard file name used for storing the report"""

  ee_report : ee.EarlyExitReport
  """This report creates/utilizes an early exit report to create and train the early exit for evaluation."""

  train_dataset : dataset.DatasetReport
  """Data to train the early exit branches."""

  validation_dataset : dataset.DatasetReport
  """Data to evaluate the early exit branches after their training."""

  calibration_sequences : List[CalibrationSequence]
  """The calibration data used to decide, if and how temporal decision mechanism can be used."""

  is_recommended : bool
  """True, if temporal decision early exit neural networks are a suitable option for the scenario."""

  search_config : str
  """The construction setting for the early exit branches."""

  #prefer_diff_detect : bool
  #""""""
  #prefer_temp_pat : bool

  def __init__(self, analysis: aNN.ModelAnalysis, calibration_sequences : List[CalibrationSequence], train_dataset: dataset.DatasetReport,
              validation_dataset: dataset.DatasetReport=None,search_config : str = "large", deserialize:bool=False) -> None:
    """
    Creates the TemporalEarlyExitReport.
    Not recommended to be used directly, use the submitter syntax instead!

    Args:
        analysis (aNN.ModelAnalysis): The ModelAnalysis to which the report will be submitted.
        calibration_sequences (List[CalibrationSequence]): the calibration data for the temporal decision.
        train_dataset (dataset.DatasetReport): Used for the early exits. Can be identical to the original training data of the model.
        validation_dataset (dataset.DatasetReport, optional): Used for the early exits. Can be identical to the original validation data of the model. Defaults to None.
        search_config (str, optional): configuration of the early exit branches. Defaults to "large".

    Raises:
        ValueError: _description_
    """
    super().__init__(analysis)

    self.search_config = search_config
    self.is_recommended = False

    if not deserialize:
      self.ee_report : ee.EarlyExitReport = ee.EarlyExitReport.submit_to(analysis=analysis).with_config(search_config=search_config)
    else:
      self.ee_report = None
    self.calibration_sequences = calibration_sequences

    if train_dataset is None:
      raise ValueError("training dataset cannot be None")
    else:
      self.train_dataset = train_dataset
      self.train_dataset.shuffle()

    if validation_dataset is None:
      log.warn("validation dataset is None")
      self.validation_dataset = train_dataset
    else:
      self.validation_dataset = validation_dataset
      self.validation_dataset.shuffle()

    if not deserialize:
      self.acc_report = acc.AccuracyReport.submit_to(analysis=analysis).with_config(datasets=[self.train_dataset, self.validation_dataset])
    else:
      self.acc_report = None
  
    self.ee_performance = {}
    self.exit_submodels = {}

  def _correlated_sequences(self) -> List[bool]:
    """
    Identifies sequences that can be used for the calibration.

    Returns:
        List[bool]: The calibration sequences that contain enough correlation.
    """
    corr = list()

    for seq in self.calibration_sequences:
      corr.append(seq.is_correlated())

    return corr

  def correlation_present(self) -> bool:
    """checks for the provided sequences, if they are temporally correlated

    Returns:
        bool: True, if correlation was detected in these sequences, False otherwise
    """
    #check if there is a temporal component in the calibration sequences
    for seq in self.calibration_sequences:
      if seq.is_correlated():
        return True

    return False
  
  def _remove_activation(self, model : tf.keras.models.Model) -> tf.keras.models.Model:
    """
    Removes the final activation function from a model.
    Used to improve the expressiveness of early exit classifiers for the propagation of input changes.

    Args:
        model (tf.keras.models.Model): The keras model from which the final activation function will be stripped.

    Raises:
        NotImplementedError: Currently unable to handle dedicated activation layers,

    Returns:
        tf.keras.models.Model: A copy of the model without the final activation layer.
    """
    if isinstance(model.layers[-1], tf.keras.layers.Activation):
      print("handle activation layer")
      raise NotImplementedError

    if hasattr(model.layers[-1], 'activation'):
      model_config = model.get_config()
      model_config["layers"][-1]["config"]["activation"] = "linear"

      new_model = tf.keras.models.Model.from_config(model_config)
      new_model.set_weights(model.get_weights())
      compile_config = self.ee_report._determine_ee_training_config(new_model, batch_size=256)
      new_model.compile(**compile_config)

      return new_model
    else:
      log.info("The last layer does not have an activation function. No changes were made.")
      return model

  def _eval_seq_numpy(self, seq : CalibrationSequence, partial_model : tf.keras.models.Model, exit_submodel:tf.keras.models.Model)-> Dict[str, object]:

    tp = []
    tn = []
    fp = []
    fn = []

    #creates inputs for early exit branch model
    inter_x = partial_model.predict(x=seq.x)

    # converts ouput into labels, if currently in one-hot encoding
    if list(self.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
      true_labels = np.argmax(seq.y, axis=-1)
    else:
      true_labels = seq.y

    # evaluates distribution of labels in current sequence
    label_statistics = np.unique(true_labels, return_counts=True)

    # evaluate exit submodel performance
    seq_eval = exit_submodel.evaluate(inter_x, seq.y)

    # create raw predictions and turn them into labels as well
    seq_pred = exit_submodel.predict(inter_x)
    if list(self.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
      pred_labels = np.argmax(seq_pred, axis=-1)
    elif list(self.analysis.tasks.values())[0] == prd.Task.BINARY_CLASSIFICATION:
      pred_labels = (seq_pred > 0.0).astype(int) # set threshold to 0.0 instead of 0.5 due to removed sigmoid activation function

    confusion_mtx = confusion_matrix(true_labels, pred_labels)

    prev_trans_id = 0
    for idx, start in enumerate(seq.transitions()):
      start_idx = start
      end_idx = start + 2
      if prev_trans_id < start-4:
        start_idx = start-4

      if start + 4 < len(seq.y) and idx+1 < len(seq.transitions()):
        if start + 4 < seq.transitions()[idx+1]:
          end_idx = start + 4

      prev_trans_id = start

      trans_x = inter_x[start_idx: end_idx]
      trans_y = seq.y[start_idx: end_idx]
      trans_pred = seq_pred[start_idx: end_idx]

      trans_pred_labels = pred_labels[start_idx: end_idx]
      trans_true_labels = true_labels[start_idx: end_idx]

      # check for label change

      # check for distance
      deltas = np.linalg.norm(trans_pred - trans_pred[0], axis=1)#[1::]
      #for i, val in enumerate(deltas):

      # store deltas for detected changes - TP
      pred_trans_deltas = [deltas[i] for i in range(1, len(trans_pred_labels)) if trans_pred_labels[i] != trans_pred_labels[0]]
      tp += pred_trans_deltas

      # store deltas for undetected changes - FN
      un_trans_deltas = [deltas[i] for i in range(1, len(trans_true_labels)) if trans_true_labels[i] != trans_true_labels[0]]
      fn += un_trans_deltas
      #print(pred_trans_deltas, un_trans_deltas)


    for start, end in seq.scenes():
      if end-start < 2:
        continue
      scene_x = inter_x[start: end]
      scene_y = seq.y[start: end]
      scene_true_labels = true_labels[start:end]

      x_pred = seq_pred[start: end]
      scene_pred_labels = pred_labels[start:end]

      deltas = np.linalg.norm(x_pred - x_pred[0], axis=1)

      # TN
      pred_scene_deltas = [deltas[i] for i in range(1, len(scene_pred_labels)) if scene_pred_labels[i] == scene_true_labels[0]]
      tn += pred_scene_deltas

      # FP
      un_scene_deltas = [deltas[i] for i in range(1, len(scene_pred_labels)) if scene_pred_labels[i] != scene_true_labels[0]]
      fp += un_scene_deltas

    eval_results = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "label_distribution" : label_statistics,
        "confusion_matrix" : confusion_mtx,
        "evaluation": seq_eval,
        "raw_predictions" : seq_pred,
        "length" : len(seq),
      }

    return eval_results
  
  def _eval_seq_tfdata(self, seq : TFCalibrationSequence, partial_model:tf.keras.models.Model, exit_submodel:tf.keras.models.Model)-> Dict[str, object]:
    
    tp = []
    tn = []
    fp = []
    fn = []

    # create intermediate results
    x = seq.get_x(batch_size=None) # will use preset batch size
    y = seq.get_y(batch_size=None)
    inter_x = partial_model.predict(x=x)

    # convert from one-hot, if necessary
    if list(self.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
      true_labels = y.map(lambda y: tf.argmax(y, axis=-1), num_parallel_calls=tf.data.AUTOTUNE)
    else:
      true_labels = y
    
    true_labels = true_labels.prefetch(tf.data.AUTOTUNE)
    true_labels = np.array([y.numpy() for y in true_labels.unbatch()])

    # x, inter_x and true_labels should all have the same batch_size at this point

    # calculate label distribution
    label_statistics = np.unique(true_labels, return_counts=True)

    seq_eval = exit_submodel.evaluate(inter_x, np.array([out.numpy() for out in y.unbatch()]))
    seq_pred = exit_submodel.predict(inter_x)

    if list(self.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
      pred_labels = np.argmax(seq_pred, axis=-1)
    elif list(self.analysis.tasks.values())[0] == prd.Task.BINARY_CLASSIFICATION:
      # set threshold to 0.0 instead of 0.5 due to removed sigmoid activation function
      pred_labels = np.array((seq_pred > 0.0).astype(np.int32))

    confusion_mtx = confusion_matrix(true_labels, pred_labels) # will this work with tf.datasets?

    # transitions behave differently on TFCalibrationSequences return (x1, x2), (y1, y2)
    prev_trans_id = 0
    for idx, (trans_x, trans_y) in enumerate(seq.transitions()):
      trans_x = np.array([x.numpy() for x in trans_x])
      trans_y = np.array([x.numpy() for x in trans_y])
      trans_pred = exit_submodel.predict(partial_model.predict(trans_x))
      trans_pred_labels = np.argmax(trans_pred, axis=-1)
      trans_true_labels = np.argmax(trans_y, axis=-1)

      deltas = np.linalg.norm(trans_pred - trans_pred[0], axis=1)

      # store deltas for detected changes - TP
      pred_trans_deltas = [deltas[i] for i in range(1, len(trans_pred_labels)) if trans_pred_labels[i] != trans_pred_labels[0]]
      tp += pred_trans_deltas

      # store deltas for undetected changes - FN
      un_trans_deltas = [deltas[i] for i in range(1, len(trans_true_labels)) if trans_true_labels[i] != trans_true_labels[0]]
      fn += un_trans_deltas

    for scene_x, scene_y in seq.scenes(min_scene_length=2):

      scene_x = np.stack(scene_x)
      
      scene_true_labels = np.argmax(scene_y, axis=-1)
      y_pred = exit_submodel.predict(partial_model.predict(x=scene_x))
      scene_pred_labels = np.argmax(y_pred, axis=-1)

      deltas = np.linalg.norm(y_pred - y_pred[0], axis=1)

      # TN
      pred_scene_deltas = [deltas[i] for i in range(1, len(scene_pred_labels)) if scene_pred_labels[i] == scene_true_labels[0]]
      tn += pred_scene_deltas

      # FP
      un_scene_deltas = [deltas[i] for i in range(1, len(scene_pred_labels)) if scene_pred_labels[i] != scene_true_labels[0]]
      fp += un_scene_deltas
    
    eval_results = {
      "tp": tp,
      "tn": tn,
      "fp": fp,
      "fn": fn,
      "label_distribution" : label_statistics,
      "confusion_matrix" : confusion_mtx,
      "evaluation": seq_eval,
      "raw_predictions" : seq_pred,
      "length" : len(seq),
    }

    return eval_results



  def evaluate(self) -> dict:
    """
    Performs the costly evaluation.

    Returns:
        Dict: The found information about the early exit branches.
    """
    tf.config.run_functions_eagerly(True)

    for recom in self.ee_report.recommendations:
      print(recom)
      # create EE

      if recom not in self.exit_submodels.keys():
        exit_submodel = self.ee_report.to_keras((recom, self.ee_report.exit_configs[recom]))
        if exit_submodel.optimizer is None:
          log.debug(f"{exit_submodel} has not been compiled yet")
          compile_config = self.ee_report._determine_ee_training_config(exit_submodel, batch_size=256)
          exit_submodel.compile(**compile_config)
      else:
        exit_submodel = self.exit_submodels[recom]

      # train / evaluate normal accuracy/performance
      if recom not in self.ee_performance.keys():
        self.ee_performance[recom] = {}
        self.ee_performance[recom]["accuracy"] =self.ee_report.evaluate_precision((recom, exit_submodel), self.train_dataset, self.validation_dataset, batch_size=256)[0]
        self.ee_performance[recom]["macs"] = self.ee_report.subgraph_costs[recom] + self.ee_report.exit_costs[recom]
        self.ee_performance[recom]["relative"] = {}
        self.ee_performance[recom]["relative"]["accuracy"] = self.ee_performance[recom]["accuracy"] - (self.acc_report.results[self.validation_dataset]["top1_accuracy"]/100)
        self.ee_performance[recom]["relative"]["macs"] = self.ee_performance[recom]["macs"] / list(self.analysis.compute.total_mac.values())[0]
        self.exit_submodels[recom] = exit_submodel

      # remove final activation from exit_submodel
      if list(self.analysis.tasks.values())[0] in [prd.Task.CLASSIFICATION, prd.Task.BINARY_CLASSIFICATION]:
        log.warn("EXPERIMENTIAL: final activation function will also be removed for binary classification tasks")
        exit_submodel = self._remove_activation(exit_submodel)

      # evaluate delta within scenes and on transitions
      partial_model = self.ee_report.get_exit_subgraph(recom)

      self.ee_performance[recom]["sequences"] = []
      for seq in self.calibration_sequences:

        if not isinstance(seq, TFCalibrationSequence):
          self.ee_performance[recom]["sequences"].append(self._eval_seq_numpy(seq, partial_model, exit_submodel))
          continue
          
        if isinstance(seq, TFCalibrationSequence):
          self.ee_performance[recom]["sequences"].append(self._eval_seq_tfdata(seq, partial_model, exit_submodel))
          continue

        '''tp = []
        tn = []
        fp = []
        fn = []

        if isinstance(seq, TFCalibrationSequence):
          x = seq.get_x(batch_size=None)
        else:
          x = seq.x

        inter_x = partial_model.predict(x=x)

        # no difference up to this point between TFCalibrationSequence and CalibrationSequence

        if isinstance(seq, TFCalibrationSequence):
          if list(self.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
            true_labels = seq.get_y(batch_size=1).map(lambda y: tf.argmax(y, axis=-1), num_parallel_calls=tf.data.AUTOTUNE)
          else:
            true_labels = seq.get_y(batch_size=1)
          
          true_labels = true_labels.prefetch(tf.data.AUTOTUNE)
        else:
          if list(self.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
            true_labels = np.argmax(seq.y, axis=-1)
          else:
            true_labels = seq.y
        
        if isinstance(seq, TFCalibrationSequence):
          print("TODO: Label statistics for TFCalibrationSequence")
          true_labels = np.concatenate([y.numpy() for y in true_labels])

        label_statistics = np.unique(true_labels, return_counts=True)

        if isinstance(seq, TFCalibrationSequence):
          y = seq.get_y(batch_size=None)
        else:
          y = seq.y

        seq_eval = exit_submodel.evaluate(inter_x, y)
        # TODO: eval prediction performance to decide terminate/difference detection/temporal patience

        seq_pred = exit_submodel.predict(inter_x)
        if list(self.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
          pred_labels = np.argmax(seq_pred, axis=-1)
        elif list(self.analysis.tasks.values())[0] == prd.Task.BINARY_CLASSIFICATION:
          pred_labels = (seq_pred > 0.0).astype(int) # set threshold to 0.0 instead of 0.5 due to removed sigmoid activation function

        confusion_mtx = confusion_matrix(true_labels, pred_labels)

        prev_trans_id = 0
        for idx, start in enumerate(seq.transitions()):
          start_idx = start
          end_idx = start + 2
          if prev_trans_id < start-4:
            start_idx = start-4

          if start + 4 < len(seq.y) and idx+1 < len(seq.transitions()):
            if start + 4 < seq.transitions()[idx+1]:
              end_idx = start + 4

          prev_trans_id = start

          trans_x = inter_x[start_idx: end_idx]
          trans_y = seq.y[start_idx: end_idx]
          trans_pred = seq_pred[start_idx: end_idx]

          trans_pred_labels = pred_labels[start_idx: end_idx]
          trans_true_labels = true_labels[start_idx: end_idx]

          # check for label change

          # check for distance
          deltas = np.linalg.norm(trans_pred - trans_pred[0], axis=1)#[1::]
          #for i, val in enumerate(deltas):

          # store deltas for detected changes - TP
          pred_trans_deltas = [deltas[i] for i in range(1, len(trans_pred_labels)) if trans_pred_labels[i] != trans_pred_labels[0]]
          tp += pred_trans_deltas

          # store deltas for undetected changes - FN
          un_trans_deltas = [deltas[i] for i in range(1, len(trans_true_labels)) if trans_true_labels[i] != trans_true_labels[0]]
          fn += un_trans_deltas
          #print(pred_trans_deltas, un_trans_deltas)


        for start, end in seq.scenes():
          if end-start < 2:
            continue
          scene_x = inter_x[start: end]
          scene_y = seq.y[start: end]
          scene_true_labels = true_labels[start:end]

          x_pred = seq_pred[start: end]
          scene_pred_labels = pred_labels[start:end]

          deltas = np.linalg.norm(x_pred - x_pred[0], axis=1)

          # TN
          pred_scene_deltas = [deltas[i] for i in range(1, len(scene_pred_labels)) if scene_pred_labels[i] == scene_true_labels[0]]
          tn += pred_scene_deltas

          # FP
          un_scene_deltas = [deltas[i] for i in range(1, len(scene_pred_labels)) if scene_pred_labels[i] != scene_true_labels[0]]
          fp += un_scene_deltas

          #print("what now?")


        self.ee_performance[recom]["sequences"].append(
          {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "label_distribution" : label_statistics,
            "confusion_matrix" : confusion_mtx,
            "evaluation": seq_eval,
            "raw_predictions" : seq_pred,
            "length" : len(seq),
          }
        )

      continue'''

    return self.ee_performance
  
  def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> Tuple[str, str]:
    """
    Renders a HTML-based summary of the report. Used for interfacing with the human user

    Args:
      folder_path (Union[str, pathlib.Path], optional): the path where the summary will be stored. Defaults to None.

    Returns:
        Tuple[str, str]: title of the summary and its filename to enable other summaries to link to it.
    """
    _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'
    if folder_path is None:
      folder_path = pathlib.Path.cwd()

    file_name = f"report_temporal_decision_{hash(self)}.html"

    with open(_template_path / "temporal_decision_report.html", "r") as file:
      template = Template(file.read())

    summary = self.dump(folder_path=folder_path)
    summary["calibration_sequences"] = self.calibration_sequences
    summary["evaluation_results"] = {}

    for recom, perf in self.ee_performance.items():
      summary["evaluation_results"][recom] = {}
      summary["evaluation_results"][recom]["sequences"] = []
      summary["evaluation_results"][recom]["accuracy"] = perf["accuracy"]
      summary["evaluation_results"][recom]["macs"] = perf["macs"]
      summary["evaluation_results"][recom]["relative"] = perf["relative"]

      for seq in perf["sequences"]:
        new_seq = {}
        for val in ["tp", "tn", "fp", "fn"]:
          data = {
            "min" : np.min(seq[val]),
            "max" : np.max(seq[val]),
            "mean": np.mean(seq[val]),
            "median": np.median(seq[val]),
          }

          new_seq[val] = data
        summary["evaluation_results"][recom]["sequences"].append(new_seq)
        

    # Render the template with the summary data
    try:
      html = template.render(summary=summary)
      # Save the generated HTML to a file
      with open(folder_path / file_name, "w") as file:
        file.write(html)

    except Exception as e:
      log.error(f"error {e} occured while trying to print summary of report")

    return "Temporal Decision Recommendations", file_name

  def access_id(self) -> str:
    """
    Creates a unique ID that identifies the report instance based on its input parameters.

    Returns:
        str: the unique identifier
    """
    return self.create_unique_id(search_config=self.search_config, calib_sequences=self.calibration_sequences)

  def dump(self, folder_path: Union[str, pathlib.Path] = None) -> None:
    """dumps the report two the specified folder, might use different formats
    JSON is preferred, but can also rely on alternative formats if required

    Args:
      folder_path (Union[str, pathlib.Path], optional): the path where the serialization shall be dumped. Defaults to None.

    Returns:
      None
    """
    summary = {
      "report_type": "Temporal Decisions",
      "name": self.analysis.name,
      "creation_date": str(self.analysis.creation_time),
      "search_config": self.search_config,
      "is_recommended" : self.is_recommended,
      "calibration_sequences" : [seq.to_dict() for seq in self.calibration_sequences],
      "train_dataset" : self.train_dataset.access_id(),
      "validation_dataset": self.validation_dataset.access_id(),
      "accuracy_report" : self.acc_report.access_id(),
      "early_exit_report" : self.ee_report.access_id(),
    }

    filename = self.__pkl_name.replace("<hash>", self.access_id())
    with open(folder_path / filename, "wb") as file:
      pickle.dump(summary, file)

    return summary
  
  @classmethod
  def load(cls, folder_path : Union[str, pathlib.Path], analysis : aNN.ModelAnalysis) -> Set[Self]:

    if not isinstance(folder_path, pathlib.Path):
      folder_path = pathlib.Path(folder_path)

    file_pattern = cls.__pkl_name.replace("<hash>", "*")
    files = glob.glob(str(folder_path) + "/" + file_pattern)

    if len(files) == 0:
      raise FileNotFoundError("no TemporalEarlyExitReport has been found in {folder_path}")
    
    dataset_reports = dataset.DatasetReport.load(folder_path=folder_path, analysis=analysis)
    acc_reports = acc.AccuracyReport.load(folder_path=folder_path, analysis=analysis)
    ee_reports = ee.EarlyExitReport.load(folder_path=folder_path, analysis=analysis)
    
    reports = []
    for file in files:
      file_path = pathlib.Path(file_path)
      with open(file_path, "rb") as file:
        summary = pickle.load(file)

      train_dataset_report = [d_set for d_set in dataset_reports if d_set.access_id() == summary["train_dataset"]][0]
      valid_dataset_report = [d_set for d_set in dataset_reports if d_set.access_id() == summary["validation_dataset"]][0]
      
      acc_report = [report for report in acc_reports if report.access_id() == summary["accuracy_report"]][0]
      ee_report = [report for report in ee_reports if report.access_id() == summary["early_exit_report"]][0]

      calib_sequences = []
      for seq in summary["calibration_sequences"]:
        new_seq = CalibrationSequence(x=np.array(seq["x"]), y=seq["y"])
        calib_sequences.append(new_seq)

      new_report = TemporalExitDecisionReport(
        analysis=analysis,
        train_dataset=train_dataset_report,
        validation_dataset=valid_dataset_report,
        search_config=summary["search_config"],
        calibration_sequences=calib_sequences,
        deserialize=True
        )

      new_report.acc_report = acc_report
      new_report.ee_report = ee_report
      
      reports.append(new_report)

    return reports

  
  @classmethod
  def submit_to(cls, analysis:aNN.ModelAnalysis, lazy:bool=False) -> TemporalExitDecisionReportSubmitter:
    """
    syntactic sugar for the creation and submission of reports.

    Args:
        analysis (aNN.ModelAnalysis): the ModelAnalysis to which the Report will be submitted.
        lazy (bool, optional): If True, only a reference to the future report will be returned, otherwise the report will be created immediately. Defaults to False.

    Returns:
        TemporalExitDecisionReportSubmitter: An auxiliary object to facilitate the syntax.
    """
    submitter = TemporalExitDecisionReportSubmitter(analysis=analysis, lazy=lazy)
    return submitter
  
  @classmethod
  def create_unique_id(cls, search_config:str, calib_sequences : List[CalibrationSequence]) -> str:
    """
    _summary_

    Args:
        search_config (str): search configuration for the early exit branch architectures.
        calib_sequences (List[CalibrationSequence]): the used calibration sequences.

    Returns:
        str: The unique identifier
    """

    #descr_str = f"TemporalReport-{name}-{search_config}:{str(calib_sequences)}"
    descr_str = f"TemporalReport-{search_config}:{len(calib_sequences)}"
    hashed_str = cls.create_reporter_id(descr_str)

    return hashed_str


  @classmethod
  def closure(cls, create_id:bool=True, calibration_sequences : List[CalibrationSequence]=None, train_dataset : dataset.DatasetReport=None, validation_dataset : dataset.DatasetReport=None, search_config:str = "large"):
    """Closure that should be passed to the ModelAnalysis object"""

    def builder(analysis: aNN.ModelAnalysis):
      return TemporalExitDecisionReport(analysis=analysis, calibration_sequences=calibration_sequences, train_dataset=train_dataset, validation_dataset=validation_dataset, search_config=search_config)
    
    if create_id:
      '''descr_str = f"TDReport_{search_config}" # TODO: handle unqiue identifier for calibration sequences and datasets
      hashed_str = cls.create_reporter_id(descr_str)'''
      hashed_str = cls.create_unique_id(search_config=search_config, calib_sequences=calibration_sequences)

      return builder, hashed_str

    return builder