from typing import List, Union, Set, Dict, Tuple
from typing_extensions import Self
import pathlib
import logging as log
import numpy as np
import pickle
import os

import tensorflow as tf
from jinja2 import Template

from . import base
from . import dataset as dataset
from . import temporal as temp
from . import batch_size as bs

import peax.analysis as aNN
from peax.components import graph_tools as gt
from peax.components import resource as res
from peax.utils import keras_graph

class HistogramReportSubmitter(base.ReportSubmitter):
  """
  This class is used to simplify the submission of the HistogramReport to a ModelAnalysis object.
  """

  def __init__(self, analysis: aNN.ModelAnalysis, lazy: bool = False):
    """
    Creates the submitter object, it is not recommended to use the constructor directly, instead use
    HistogramReport.submit_to(analysis, lazy).
    To create and submit the report, just use HistogramReport.submit_to(analysis, lazy).with_config(...).

    Args:
        analysis (aNN.ModelAnalysis): the ModelAnalysis to which the report(er) will be assigned
        lazy (bool, optional): The submission behavior. lazy will only create the report if it is required, otherwise it will be generated immediately.
            Defaults to False.
    """
    super().__init__(analysis, lazy)

  def with_config(self,
                  train_dataset:dataset.DatasetReport,
                  validation_dataset:dataset.DatasetReport,
                  calibration_sequences:List[temp.CalibrationSequence],
                  max_cost:float=1.0,
                  resolution:str="global"
                ) -> Union[str, base.Report]:
    """
    Creates submission to the ModelAnalysis object.
    If configures as lazy, the unique ID of the report will be returned, but it will not yet be created.
    Otherwise the HistogramReport instance will be created and returned.

    Args:
        train_dataset (dataset.DatasetReport): the training dataset used during the evaluation.
        validation_dataset (dataset.DatasetReport, optional): the validation dataset used during the evaluation. Defaults to None.
        calibration_sequences (List[CalibrationSequence]): the calibration sequences that will be used during the evaluation.
        max_cost (float, optional): the maximum relative cost for the considered IFMs. Defaults to 1.0.
        resolution (str, optional): the histogram resolution, currently only "global" is implemented, which calculates the average value per channel. Defaults to "global".

    Returns:
        Union[str, base.Report]: Returns either the reporter_id of the submitted constructor or the HistogramReport object.
    """
    
    hist_reporter, hist_uid = HistogramReport.closure(
      create_id=True,
      calibration_sequences=calibration_sequences,
      train_dataset=train_dataset,
      validation_dataset=validation_dataset,
      cost_limit=max_cost,
      resolution=resolution
      )
    
    self.analysis.submit_reporter(hist_uid, hist_reporter)
    
    if self.lazy:
      return hist_uid
    else:
      return self.analysis.access_report(hist_uid)

class HistogramReport(base.Report):
  """
  Report to evaluate the quality of IFMs after the trainable layers in the model.
  They are evaluated with regards to the inference cost that is required to acquire them as well as their expressiveness.
  """

  __pkl_name = "report_histogram_<hash>.pkl"
  """ The standard file name used for storing the report. """

  _html_name_template = "report_histogram_<hash>.html"

  #analysis : aNN.ModelAnalysis
  train_dataset : dataset.DatasetReport
  """ Train dataset used to train kMeans classifier to evaluate quality of IFMs. """

  validation_dataset : dataset.DatasetReport
  """ Validation dataset used by kMeans classifier to evaluate quality of IFMs. """

  batch_report : bs.BatchSizeReport
  """ hidden reporter that attempts to find the best training and inference batch size for the currently used system. """

  batch_size : int
  """ Preferred batch size used during the execution of this report. """
  
  calibration_sequences : List[temp.CalibrationSequence]
  """ The calibration data used to decide, if the expressiveness of IFMs is enough to separate sequences in the input data. """

  cost_limit : float
  """ A maximum cost limit for the evaluated IFM locations, given as relative value compared to full model inference. """

  resolution : str
  """ Resolution for the histogram creation step, currently only "global" is supported, which creates one value per channel in the IFM. """

  locations : List[gt.BlockNode]
  """ Locations that have been identified. """

  mac_costs : Dict[gt.BlockNode, float]
  """ Full cost of creating the histogram of the identified locations. """

  subgraph_mac_costs : Dict[gt.BlockNode, float]
  """ Cost of executing the model up to the location. """

  branch_mac_costs : Dict[gt.BlockNode, float]
  """ Cost of executing just the histogram creation step. """

  pred_quality : Dict[gt.BlockNode, float]
  """Prediction quality of the identfied locations."""

  branch_models : Dict[gt.BlockNode, tf.keras.models.Model]
  """Models that only contain the layers required to create the IFM histograms."""

  branches : Dict[gt.BlockNode, tf.keras.models.Model]
  """Models that only contain the layers that are necessary to create the histogram at their locations."""

  sequence_data : Dict[gt.BlockNode, List[Dict[str, np.array]]]
  """The histograms of the IFMs for the scenes and transitions of the provided calibration sequences."""

  def __init__(self,
               analysis: aNN.ModelAnalysis,
               train_dataset : dataset.DatasetReport,
               validation_dataset : dataset.DatasetReport,
               calibration_sequences : List[temp.CalibrationSequence],
               cost_limit:float=1.0,
               resolution:str = "global") -> None:
    """
    Creates the HistogramReport.
    Not recommended to be used directly, use the submitter syntax instead!

    Args:
        analysis (aNN.ModelAnalysis): The ModelAnalysis to which the report will be submitted.
        train_dataset (dataset.DatasetReport): Required for reporting. Can be identical to the original training data of the model.
        validation_dataset (dataset.DatasetReport): Optional. Can be identical to the original training data of the model.
        calibration_sequences (List[temp.CalibrationSequence]): _description_
        cost_limit (float, optional): _description_. Defaults to 1.0.
        resolution (str, optional): _description_. Defaults to "global".
    """
    super().__init__(analysis)

    self.locations = None
    self.subgraph_mac_costs = None
    self.branch_models = None
    self.branches = None
    self.branch_mac_costs = None
    self.mac_costs = None

    self.train_dataset = train_dataset
    if validation_dataset is None:
      self.validation_dataset = train_dataset
      log.warn("no validation set given, will use training data for evaluation, might result in poor quality!")
    else:
      self.validation_dataset = validation_dataset

    self.batch_report = bs.BatchSizeReport.submit_to(analysis=analysis, lazy=False).with_config()
    self.batch_size = self.batch_report.inference

    self.calibration_sequences = calibration_sequences

    self.cost_limit = cost_limit
    self.resolution = resolution

    self.locations = self.identify_locations(cost_limit)
    self.subgraph_mac_costs = self.estimate_subgraph_costs()
    self.branch_models, self.branches = self.create_branch_models()
    self.branch_mac_costs = self.estimate_branch_costs()
    self.dist_mac_costs = self.estimate_distance_calc_costs()
    self.mac_costs = self.estimate_costs()

    self.pred_quality = {}
    self.sequence_data = {}
    #self.pred_quality = self.evaluate_branches()

    return
  
  def identify_locations(self, cost_limit:float=1.0) -> List[gt.BlockNode]:
    """
    Identifies loations whose IFMs could be relevant.
    Limits search to the outputs of trainable layers that are not outputs.
    Can be limited to only account for locations that can be generated with a relative cost of cost_limit.

    Args:
        cost_limit (float, optional): The limit for the relative cost compared to the full inference. Defaults to 1.0.

    Returns:
        List[gt.BlockNode]: A list of suitable locations in the block-level representation of the model.
    """

    block_graph = self.analysis.architecture.block_graph

    #suitable locations would be after the compute blocks

    all_locations = list(block_graph.nodes())

    suitable = [node for node in all_locations if node.dominant_operation in ["compute", "convolution"]]

    # discard output block
    outputs = gt.get_output_nodes(block_graph)
    suitable = [node for node in suitable if node not in outputs]

    # should we enable limiting the locations to be within a given share of operations compared to the full inference?
    total_cost = list(self.analysis.compute.total_mac.values())[-1]
    input_node = gt.get_first_input_node(block_graph)
    rel_cost = [(res.get_subgraph_macs(block_graph, input_node, node)) / total_cost for node in suitable]

    found_options = []
    for node, cost in zip(suitable, rel_cost):
      if cost <= cost_limit:
        found_options.append(node)
      else:
        log.info(f"{node} rejected due to relative cost of {cost}, which is above limit of {cost_limit}")

    return found_options
  
  def estimate_branch_costs(self) -> Dict[gt.BlockNode, float]:
    """
    Estimates the relative inference cost of the histogram branch.
    The values are the relative cost compared to the full inference of the model.

    Returns:
        Dict[gt.BlockNode, float]: A dict that specifies the relative cost of the attached branch for each suitable location.
    """

    mac_costs = {}
    total_cost = list(self.analysis.compute.total_mac.values())[-1]

    if self.locations == None or len(self.locations) == 0:
      self.locations = self.identify_locations()

    if self.branch_models == None or len(self.branch_models) == 0:
      self.branch_models, self.branches = self.create_branch_models()

    mac_costs = {loc:(sum(res.get_model_macs(model).values()) / total_cost) for loc, model in self.branches.items()}

    return mac_costs
  
  def estimate_distance_calc_costs(self) -> Dict[gt.BlockNode, float]:
    """
    Calculates the at-runtime impact of the distance calculation.
    While its overall impact is often negligible, it is important to consider it to create accurate estimates for edge cases (i.e. very large IFMs).

    Returns:
        Dict[gt.BlockNode, float]:  A dict that specifies the relative distance calculation cost for each suitable location.
    """
    costs = {}
    total_cost = list(self.analysis.compute.total_mac.values())[-1]

    if self.locations == None or len(self.locations) == 0:
      self.locations = self.identify_locations()

    if self.branch_models == None or len(self.branch_models) == 0:
      self.branch_models, self.branches = self.create_branch_models()

    def est_ops(dims:int) -> int:
      return 3*dims+1
    
    costs = {loc:(est_ops(model.output_shape[-1]) / total_cost) for loc, model in self.branches.items()}
    return costs
  
  def estimate_subgraph_costs(self) -> Dict[gt.BlockNode, float]:
    """
    Estimates the inference cost in MACs based on the model architecture.
    The values are the relative cost compared to the full inference of the model.

    Returns:
        Dict[gt.BlockNode, float]: A dict that specifies the relative subgraph (input up to branch attachement location) cost for each suitable location.
    """
    mac_costs = {}
    block_graph = self.analysis.architecture.block_graph
    total_cost = list(self.analysis.compute.total_mac.values())[-1]
    input_node = gt.get_first_input_node(block_graph)

    if self.locations == None or len(self.locations) == 0:
      self.locations = self.identify_locations()

    rel_cost = [(res.get_subgraph_macs(block_graph, input_node, node)) / total_cost for node in self.locations]

    for idx, loc in enumerate(self.locations):
      mac_costs[loc] = rel_cost[idx]

    return mac_costs
  
  def estimate_costs(self) -> Dict[gt.BlockNode, float]:
    """
    Estimates the total relative cost for each IFM analysis location.
    Adds the subgraph, branch and distance calcuation cost up.

    Returns:
        Dict[gt.BlockNode, float]: A dict that specifies the relative (to the total cost of the original model) cost for each suitable location.
    """

    self.branch_mac_costs = self.estimate_branch_costs()
    self.subgraph_mac_costs = self.estimate_subgraph_costs()
    self.dist_mac_costs = self.estimate_distance_calc_costs()

    mac_costs = {}

    for loc in self.branch_mac_costs.keys():
      mac_costs[loc] = self.branch_mac_costs[loc] + self.subgraph_mac_costs[loc] + self.dist_mac_costs[loc]

    return mac_costs
  
  def create_branch_models(self, resolution:str=None) -> Tuple[Dict[gt.BlockNode, tf.keras.models.Model], Dict[gt.BlockNode, tf.keras.models.Model]]:
    """
    Creates the models to acquire the IFMs at the different location.
    The added branch performs additional downsmapling to minimize the size of the individual IFMs.
    Currently only "global" is supported, which reduces the IFMs to the average per channel.

    Args:
        resolution (str, optional): The resolution of the acquired IFMs. Defaults to "global".

    Raises:
        AttributeError: raises an error if now suitable downsampling branch could be constructed for an IFM dimensionality.

    Returns:
        Tuple[Dict[gt.BlockNode, tf.keras.models.Model], Dict[gt.BlockNode, tf.keras.models.Model]]: 
        Two Dicts: one contains the keras models that contain all layers necessary to create the IFM from the original input,
        the second contains the models that represent only the attachable branch.
    """
    _global_pool_selection ={
      1 : tf.keras.layers.GlobalAveragePooling1D,
      2 : tf.keras.layers.GlobalAveragePooling2D,
      3 : tf.keras.layers.GlobalAveragePooling3D,
    }

    _pool_selection ={
      1 : tf.keras.layers.AveragePooling1D,
      2 : tf.keras.layers.AveragePooling2D,
      3 : tf.keras.layers.AveragePooling3D,
    }

    _pool_sizes = {
      "coarse" : 2,
      "fine" : 8,
    }

    if self.locations is None:
      self.locations = self.identify_locations(self.cost_limit)

    if resolution is None:
      resolution = self.resolution

    branch_models = {}
    branches = {}
    for node in self.locations:
      #create pooling branch
      adapter_shape = node.output_shape
      inp = tf.keras.layers.Input(shape=adapter_shape)
      if resolution == "global":
        log.debug("going to construct global pooling branches")
        try:
          pool_layer = _global_pool_selection[len(adapter_shape)-1]
          params = {"keepdims" : False}
        except KeyError:
          log.error("no suitable pooling layer can be found for the given shape")
          raise AttributeError("the IFM dimensionality cannot be processed")
      else:
        log.debug("going to construct NONE global pooling branches")
        try:
          pool_layer = _pool_selection[len(adapter_shape)-1]
          params = {}
        except KeyError:
          log.error("no suitable pooling layer can be found for the given shape")
          raise AttributeError("the IFM dimensionality cannot be processed")
        
        pool_sizes = []
        pool_factor = 1
        for dim_idx, dim in enumerate(adapter_shape):
          if dim_idx == len(adapter_shape) -1:
            break
          pool_sizes.append(max(dim // _pool_sizes[resolution],3))
        
        params["pool_size"] = tuple(pool_sizes)

      x = pool_layer(**params)(inp)

      if resolution != "global":
        x = tf.keras.layers.Flatten()(x)

      attach_layer = gt.get_first_output_node(node.subgraph).keras
      branch = tf.keras.models.Model(inputs=[inp], outputs=[x], name=f"{res}-branch_{node.name}")
      branched_model = keras_graph.attach_branch(self.analysis.keras_model, attach_layer, branch, reorder=True)
      branch_only_model = tf.keras.models.Model(inputs=branched_model.inputs, outputs=branched_model.outputs[0], name=f"{branched_model.name}-branch_only")
      branch_models[node] = branch_only_model
      branches[node] = branch

    return branch_models, branches
  
  def get_sequence_delta(self, node : gt.BlockNode, sequence_id:int=-1) -> List[float]:
    """
    Gets the euclidean distances between IFMs for a given calibration sequence.

    Args:
        node (gt.BlockNode): The location that will be used to create the IFMs.
        sequence_id (int, optional): The index of the used calibration sequence. Defaults to -1.

    Returns:
        List[float]: The distances between the subsequent samples.
    """

    samples = self.calibration_sequences[sequence_id].x
    if sequence_id == -1:
      samples = []
      for seq in self.calibration_sequences:
        samples.append(seq.x)
        samples = np.stack(samples)

    preds = self.branch_models[node].predict(samples)
    deltas = [np.linalg.norm(preds[0])]

    for i in range(1, preds.shape[0]):
      distance = np.linalg.norm(preds[i] - preds[i-1])
      deltas.append(distance)

    return deltas

  
  def evaluate_sequences(self, overwrite:bool=False) -> Dict[gt.BlockNode, List[Dict[str, List[float]]]]:
    """
    Evaluates the IFM behaviour for the given calibration sequences.
    Measures the distances between subsequent samples within scenes and transitions.


    Args:
        overwrite (bool, optional): If True, existing results will be overwritten. Defaults to False.

    Returns:
        Dict[gt.BlockNode, List[Dict[str, List[float]]]]: For each evaluated location, the distances for the scenes and transitions of each calibration sequence will be returned.
    """

    if self.branch_models is None:
      self.branch_models = self.create_branch_models()

    if len(self.sequence_data) != 0 and overwrite is False:
      return self.sequence_data

    collected_data = {}
    for branch_location, model in self.branch_models.items():
      log.info(branch_location.name, model.output_shape[-1])

      collected_data[branch_location] = []
      for seq_id, calib_seq in enumerate(self.calibration_sequences):

        collected_data[branch_location].append({})
        scene_deltas = []
        for scene_start, scene_end in calib_seq.scenes():
          samples = calib_seq.x[scene_start:scene_end]
          if len(samples) == 0:
            continue
          ifms = model.predict(samples, verbose=0)

          distances = []
          for i in range(1, ifms.shape[0]):
            distance = np.linalg.norm(ifms[i] - ifms[i-1])
            distances.append(distance)
          
          if len(distances) > 2 and distances[-1] > 1.5 * np.mean(distances):
            # remove last sample to smooth data
            # these samples often already show the influence of the next transition
            distances = distances[0:-1]

          scene_deltas += distances

        log.info(f"scenes:\t{min(scene_deltas)}\t{max(scene_deltas)}\t{np.mean(scene_deltas)}\t{np.median(scene_deltas)}")
        collected_data[branch_location][seq_id]["scenes"] = scene_deltas

        transition_deltas = []
        for trans_start in calib_seq.transitions():
          #print(trans_start)
          samples = calib_seq.x[trans_start:trans_start+2]

          if len(samples) == 0:
            continue
          ifms = model.predict(samples, verbose=0)
          distance = np.linalg.norm(ifms[0] - ifms[1])
          transition_deltas.append(distance)

        log.info(f"trans:\t{min(transition_deltas)}\t{max(transition_deltas)}\t{np.mean(transition_deltas)}\t{np.median(transition_deltas)}")
        collected_data[branch_location][seq_id]["transitions"] = transition_deltas
    
    self.sequence_data = collected_data
    return collected_data

  
  @tf.autograph.experimental.do_not_convert
  def evaluate_branches(self) -> Dict[gt.BlockNode, float]:
    """
    Evaluates the branches in their ability to create expressive IFMs.
    The evaluation only supports classification tasks and uses a KMeansClassifier to evaluate the quality
    by training and evaluating it on the data that was extracted at each location.

    Returns:
        Dict[gt.BlockNode, float]: The validation set accuracy of the KMeansClassifer for each suitable location.
    """
    if self.locations is None:
      self.locations = self.identify_locations(self.cost_limit)
    
    if self.branch_models is None:
      self.branch_models = self.create_branch_models()

    train_sample_dataset = self.train_dataset.data.map(lambda x, y: x).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    train_labels = list(self.train_dataset.data.map(lambda x, y: y))
    train_labels = np.stack([label.numpy() for label in train_labels])
    train_labels = np.argmax(train_labels, axis=-1)

    valid_sample_dataset = self.validation_dataset.data.map(lambda x, y: x).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    valid_labels = list(self.validation_dataset.data.map(lambda x, y: y))
    valid_labels = np.stack([label.numpy() for label in valid_labels])
    valid_labels = np.argmax(valid_labels, axis=-1)

    pred_quality = {}
    for node, model in self.branch_models.items():
      
      train_output = model.predict(train_sample_dataset)
      train_output = np.array(train_output)

      valid_output = model.predict(valid_sample_dataset)
      valid_output = np.array(valid_output)

      usability = self._eval_quality(train_output, train_labels, valid_output, valid_labels)

      print(node, usability)
      pred_quality[node] = usability
    
    self.pred_quality = pred_quality
    return pred_quality

  @staticmethod
  def _eval_quality(x_train : np.array, y_train : np.array, x_valid : np.array, y_valid : np.array) -> float:
    """
    Evaluates the IFM quality by using the accuracy of a KMeansClassifier as proxy.

    Args:
        x_train (np.array): the training samples.
        y_train (np.array): the training labels.
        x_valid (np.array): the validation samples.
        y_valid (np.array): the validation labels.

    Returns:
        float: the accuracy of the classifier on the validation set.
    """
    from sklearn.neighbors import NearestCentroid
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)

    clf = NearestCentroid()
    clf.fit(X_train_scaled, y_train)

    X_valid_scaled = scaler.transform(x_valid, copy=True)

    acc = accuracy_score(clf.predict(X_valid_scaled), y_valid)

    return acc

  def access_id(self) -> str:
    """
    Returns the unique identifier for this report instance.

    Returns:
        str: A unique id that captures the input parameters to enable reusing previously created reports.
    """

    hash_str = self.create_unique_id(
      cost_limit=self.cost_limit,
      resolution=self.resolution,
      train_dataset_name=self.train_dataset.name,
      valid_dataset_name=self.validation_dataset.name,
      calib_sequence=self.calibration_sequences)
    return hash_str
  

  @classmethod
  def create_unique_id(cls,
                       cost_limit:float,
                       resolution:str,
                       train_dataset_name:str,
                       valid_dataset_name:str,
                       calib_sequence:List[temp.CalibrationSequence]
                      ) -> str:
    """
    Creates the unique ID for reports and reporters.

    Args:
        cost_limit (float): the cost limit parameter.
        resolution (str): the resolution parameter.
        train_dataset_name (str): The name of the used training dataset report
        valid_dataset_name (str): The name of the used validation dataset report
        calib_sequence (List[temp.CalibrationSequence]): The list of used calibration sequences.

    Returns:
        str: The unique ID.
    """
    desc_str = f"IFMHistogramReport:{cost_limit}-{resolution}:{train_dataset_name},{valid_dataset_name}:{len(calib_sequence)}"
    hashed_str = cls.create_reporter_id(desc_str)
    return hashed_str

  
  def dump(self, folder_path: Union[str, pathlib.Path] = None):
    if folder_path is None:
        folder_path = pathlib.Path.cwd()

    if isinstance(folder_path, str):
        folder_path = pathlib.Path(folder_path)

    summary = {
      "report_type": "IFM Histogram",
      "report_id": self.access_id(),
      "name" : self.analysis.name,
      "creation_date" : str(self.analysis.creation_time),
      "resolution" : self.resolution,
      "cost_limit" : self.cost_limit,
      "calibration_sequences" : [seq.to_dict() for seq in self.calibration_sequences],
      "train_dataset" : self.train_dataset.access_id(),
      "validation_dataset": self.validation_dataset.access_id(),
      "locations" : [loc.name for loc in self.locations],
      "mac_costs" : dict([(loc.name, data) for loc, data in self.mac_costs.items()]),
      "branch_mac_costs" : dict([(loc.name, data) for loc, data in self.branch_mac_costs.items()]),
      "subgraph_mac_costs" : dict([(loc.name, data) for loc, data in self.subgraph_mac_costs.items()]),
      "distance_mac_costs" : dict([(loc.name, data) for loc, data in self.dist_mac_costs.items()]),
      "pred_quality" : dict([(loc.name, data) for loc, data in self.pred_quality.items()]),
      "sequence_data" : dict([(loc.name, data) for loc, data in self.sequence_data.items()]),
    }

    filename = self.__pkl_name.replace("<hash>", self.access_id())
    with open(folder_path / filename, "wb") as file:
      pickle.dump(summary, file)

    return summary
  
  def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> Tuple[str, pathlib.Path]:
    """
    Creates a HTML-based summary of the report.

    Args:
        folder_path (Union[str, pathlib.Path], optional): The folder where the summary should be stored. Defaults to current working directory if None.

    Returns:
        Tuple[str, pathlib.Path]: The name and path of the summary.
    """
    _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'

    if isinstance(folder_path, str):
        folder_path = pathlib.Path(folder_path)

    with open(_template_path / "histogram_report.html", "r") as file:
        template = Template(file.read())

    file_name = self._html_name_template.replace("<hash>", self.access_id()) #f"report_histogram_{self.access_id()}.html"

    summary = self.dump(folder_path=folder_path)
    summary_seqs = {}
    for loc_name, seqs in summary["sequence_data"].items():
      summary_seqs[loc_name] = []
      for seq in seqs:
        stats = {
          "scenes" : [min(seq["scenes"]), np.mean(seq["scenes"]), np.median(seq["scenes"]), max(seq["scenes"])],
          "transitions" : [min(seq["transitions"]), np.mean(seq["transitions"]), np.median(seq["transitions"]), max(seq["transitions"])]
        }
        summary_seqs[loc_name].append(stats)

    summary["sequence_data"] = summary_seqs

    html = template.render(summary=summary)
    with open(folder_path / file_name, "w") as file:
        file.write(html)

    return "IFM Histograms", file_name
  
  @classmethod
  def load(cls, folder_path: Union[str, pathlib.Path], analysis: aNN.ModelAnalysis) -> Set[Self]:
    """ Loads the report from a serialized file from disk. """
    raise NotImplementedError("This function has not yet been implemented.")
  
    # need to reload dataset reports.

    # need to rerun batch size report as we might deserialize on another system.

    # need to reload calibration sequences.

    # restore locations, costs, quality, branches + branch models and sequence data
    return super().load(folder_path, analysis)
  
  @classmethod
  def submit_to(cls, analysis:aNN.ModelAnalysis, lazy:bool=False) -> HistogramReportSubmitter:
    """
    syntactic sugar for the creation and submission of reports.

    Args:
        analysis (aNN.ModelAnalysis): _description_
        lazy (bool, optional): If True, only a reference to the future report will be returned,
        otherwise the report will be created immediately. Defaults to False.

    Returns:
        HistogramReportSubmitter: An auxiliary object to facilitate the syntax.
    """

    submitter = HistogramReportSubmitter(analysis=analysis, lazy=lazy)
    return submitter

  @classmethod
  def closure(cls,
              create_id:bool=True,
              calibration_sequences : List[temp.CalibrationSequence]=None,
              train_dataset : dataset.DatasetReport=None,
              validation_dataset : dataset.DatasetReport=None,
              cost_limit:float=1.0,
              resolution:str="global"):
    
    """Closure that should be passed to the ModelAnalysis object. """
    
    def builder(analysis:aNN.ModelAnalysis):
      return HistogramReport(analysis=analysis,
                             train_dataset=train_dataset,
                             validation_dataset=validation_dataset,
                             calibration_sequences=calibration_sequences,
                             cost_limit=cost_limit,
                             resolution=resolution
                            )
    
    if create_id:
      hash_str = cls.create_unique_id(
        cost_limit=cost_limit,
        resolution=resolution,
        train_dataset_name=train_dataset.name,
        valid_dataset_name=validation_dataset.name,
        calib_sequence=calibration_sequences
      )

      return builder, hash_str
    
    return builder
  
