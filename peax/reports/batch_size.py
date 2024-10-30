import pathlib
from typing import Dict, Tuple, Union
from typing_extensions import Self
import peax.analysis as aNN
from .base import Report, ReportSubmitter
import peax.analysis as aNN

import socket
import tensorflow as tf
import pathlib
import numpy as np
import time
import json

class BatchSizeReportSubmitter(ReportSubmitter):
  """
  Syntactic sugar to ease process of submitting reporter to ModelAnalysis
  """

  def __init__(self, analysis: aNN.ModelAnalysis, lazy: bool = False):
    """
    Creates the submitter that can be used to submit reports or reporters to the ModelAnalysis

    Args:
        analysis (aNN.ModelAnalysis): the ModelAnalysis to which everything should be assigned
        lazy (bool, optional): If True, only a reference to the future report will be returned, otherwise the report will be created immediately. Defaults to False.
    """
    super().__init__(analysis, lazy)

  def with_config(self, start_size:int=1, end_size:int=4096) -> Union[str, Report]:
    """
    Submits the reporter to the analysis.
    If the submitter was configured to be lazy, only a reference to the future report will be returned,
    otherwise the full report is returned by this function.

    Args:
        start_size (int, optional): start size for the batch size search. Defaults to 1.
        end_size (int, optional): maximum batch size to be evaluated. Defaults to 4096.

    Returns:
        Union[str, Report]: either the unique ID of the report or the report itself
    """
    bs_reporter, bs_uid = BatchSizeReport.closure(create_id=True, start_size=start_size, end_size=end_size)

    self.analysis.submit_reporter(bs_uid, bs_reporter)

    if self.lazy:
      return bs_uid
    else:
      return self.analysis.access_report(bs_uid)

class BatchSizeReport(Report):
  """
  Report that tries to find the best batch size for the current device.
  This decision is mostly based on the available memory and the performance per sample, not the training quality impact.
  """

  inference : int
  training : int
  device_identifier : str

  def __init__(self, analysis: aNN.ModelAnalysis, start:int=1, end:int=4096, check_cache:bool=True) -> None:
    """
    Creates the BatchSizeReport

    Args:
        analysis (aNN.ModelAnalysis): the analysis for which the batch size needs to be found
        start (int, optional): the initial value for the search space. Defaults to 1.
        end (int, optional): the maximum value for the batch size. Defaults to 4096.
    """
    super().__init__(analysis)
    self.hide = True
    self.device_identifier = self._create_device_identifier()

    self.inference_size = None
    self.training_size = None

    if check_cache:
      self._load()

    self.start_size = start
    self.end_size = end

  def _create_device_identifier(self) -> str:
    """
    Tries to find a unique identifier for the device that is currently running the analysis, as the optimal batch size is device specific

    Returns:
        str: a unique identifier for the currently used workstation
    """
    device_id = f"{socket.gethostname()}" # TODO: more info needed?
    return device_id
  
  @property
  def _cache_path(self) -> pathlib.Path:
    """
    the path were the optimal batch sizes should be cached

    Returns:
        pathlib.Path: path to the cache file
    """
    cache_path = self.analysis.cache_dir / self.analysis.keras_model.name
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / "bs_store.json"
  
  def _load(self, overwrite:bool=True, path:pathlib.Path = None) -> Dict[str, int]:
    """
    loads the previous data from a cache file, if a fitting file already exists

    Args:
        overwrite (bool, optional): if True, it overwrites the values within the report with the data loaded from the cache. Defaults to True.
        path (pathlib.Path, optional): path to the folder containing the cache file, uses default path if None. Defaults to None.

    Returns:
        _type_: _description_
    """
    if path is None:
      path = self._cache_path
    else:
      path = path / "ds_store.json"

    if path.exists() and path.is_file:
      with open(path, "r") as file:
        stored = json.load(file)
      
      if overwrite:
        self.inference_size = stored[self.device_identifier][0]
        self.training_size = stored[self.device_identifier][1]
    else:
      stored = {}

    return stored


  def _store(self, path:pathlib.Path = None):
    """
    Stores the found optimal batch sizes in a cache file

    Args:
        path (pathlib.Path, optional): Path to the folder containing the cache file, uses default path if None. Defaults to None.
    """
    if path is None:
      path = self._cache_path
    else:
      path = path / "ds_store.json"

    try:
      stored = self._load(overwrite=False)
    except:
      stored = {}
    stored[self.device_identifier] = (self.inference_size, self.training_size)

    with open(path, "w") as file:
      json.dump(stored, file)

    return

  def _find_inference_batch_size(self) -> int:
    """
    Tries to find the ideal batch size for inference on the current workstation by iteratively trying different configurations

    Returns:
        int: the ideal batch size
    """
    model = self.analysis.keras_model#tf.keras.models.clone_model(self.analysis.keras_model)
    inp_shape = list(model.input_shape)
    dtype = model.input_spec[0].dtype
    if dtype is None:
      dtype = np.float32

    value = self.start_size
    latency = np.Infinity

    while value <= self.end_size:
      inp_shape[0] = value
      inp_tensor = np.random.rand(np.prod(inp_shape)).astype(dtype).reshape(inp_shape)

      try:
        start_time = time.time()
        model.predict(inp_tensor, batch_size=value)
        end_time = time.time()
        new_latency = (end_time-start_time) / value
        if new_latency < latency:
          latency = new_latency
        else:
          break
      except:
        value /= 2
        break
      value *= 2
    
    if value > self.end_size:
        value = self.end_size

    return int(value)
  
  def _find_training_batch_size(self) -> int:
    """
    Tries to find the ideal batch size for training on the current workstation by iteratively trying different configurations

    Returns:
        int: the ideal batch size
    """

    #raise NotImplemented("this function has not yet been implemented")
    model = tf.keras.models.clone_model(self.analysis.keras_model)
    inp_shape = list(model.input_shape)
    dtype = model.input_spec[0].dtype
    if dtype is None:
      dtype = np.float32

    out_shape = list(model.output_shape)

    steps = [32, 64, 128, 256, 512]
    latency = np.Infinity

    #TODO: need to compile the model if not done yet
    model.compile(optimizer='adam', loss='binary_crossentropy')

    for step in steps:
      inp_shape[0] = step
      inp_tensor = np.random.rand(np.prod(inp_shape)).astype(dtype).reshape(inp_shape)

      out_shape[0] = step
      out_tensor = np.random.rand(np.prod(out_shape)).reshape(out_shape)

      try:
        start_time = time.time()
        model.fit(inp_tensor, out_tensor, batch_size=step)
        end_time = time.time()
        new_latency = (end_time-start_time) / step
        if new_latency < latency:
          latency = new_latency
        else:
          break
      except:
        break

    return int(step)

  @property
  def inference(self):
    """
    the optimal inference batch size for the current device
    """
    if self.inference_size is None:
      self.inference_size = self._find_inference_batch_size()
      self._store()
    
    return self.inference_size
  
  @inference.setter
  def inference(self, value):
      raise AttributeError("the inference batch size is a read-only attribute")
  
  @property
  def training(self):
    """
    the optimal training batch size for the current device
    """
    if self.training_size is None:
      self.training_size = self._find_training_batch_size()
      self._store()

    return self.training_size
  
  @training.setter
  def training(self, value):
      self.training_size = int(value)

  def __str__(self):
    return f"BatchSizeReport for {self.analysis.name} on {self.device_identifier}: inference={self.inference_size}, training={self.training_size}"

  def dump(self, folder_path: Union[str, pathlib.Path] = None):
    return self._store(folder_path)
  
  @classmethod
  def load(cls, folder_path: Union[str, pathlib.Path], analysis : aNN.ModelAnalysis) -> Self:
    """
    restores the report from disk.
    As this data is device dependent, it will check the local config cache instead of the project folder.

    Args:
        folder_path (Union[str, pathlib.Path]): project folder path, not used right now, but here due to inheritance
        analysis (aNN.ModelAnalysis): the analysis to which the report will be assigned

    Returns:
        BatchSizeReport: the newly created BatchSizeReport instance, either containing previously cached data or newly created info
    """
    bs_report : Self = BatchSizeReport.submit_to(analysis=analysis, lazy=False).with_config(start_size=1, end_size=4096, check_cache=False)
    bs_report._load(overwrite=True)

    return bs_report
  
  def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> None:
    """
    This report will not render a summary, as it is just an auxiliary tool for the dataset and accuracy reports.

    Args:
        folder_path (Union[str, pathlib.Path], optional): The folder path where the summary shall be stored. Defaults to None.

    Returns:
        None: THIS REPORT WILL NOT RETURN ANYTHING
    """
    return None
  
  @classmethod
  def submit_to(cls, analysis : aNN.ModelAnalysis, lazy:bool=False) -> BatchSizeReportSubmitter:
    """
    Syntactic sugar to simplify submission process: BatchSizeReport.submit_to(analysis=analysis, lazy=False).with_config(start_size=1, end_size=4096)

    Args:
        analysis (aNN.ModelAnalysis): the analysis the report will be submitted to
        lazy (bool, optional): the submission behavior. Defaults to False.

    Returns:
        BatchSizeReportSubmitter: intermediate object
    """
    return BatchSizeReportSubmitter(analysis=analysis, lazy=lazy)
  
  @classmethod
  def closure(cls, create_id:bool=True, start_size:int=1, end_size:int=4096) -> Union[Self, Tuple[callable, str]]:
    """
    Returns the constructor of a new instance of the class as object to enable passing it to the ModelAnalysis instance

    Args:
        create_id (bool, optional): if True, a unique identifier for the instance than can be created from the constructor will be returned alongside the constructor. Defaults to True.
        start_size (int, optional): the minimum value for the batch size search. Defaults to 1.
        end_size (int, optional): the maximum value for the batch size search. Defaults to 4096.
    """
    def builder(analysis: aNN.ModelAnalysis):
      return BatchSizeReport(analysis=analysis, start=start_size, end=end_size)
    
    if create_id:
      descr_str = f"BatchSizeReport_{start_size}_{end_size}"
      hashed_str = cls.create_reporter_id(descr_str)

      return builder, hashed_str
    
    return builder