import hashlib
import logging as log
import pathlib
from typing import Set, Union, Dict, List, Tuple
import numpy as np
import tensorflow as tf
import pathlib

import peax.analysis as aNN
from . import base
import peax.analysis as aNN

from peax.components import graph_tools as gt
from peax.utils.keras_graph import exchange_layer

from ..reports import early_exit as ee
from ..reports import accuracy as acc
from ..reports import dataset as d

from ..components import predictive as prd
from ..components import resource as res

def _is_depthwise_separable_keras_layer(layer : tf.keras.layers.Conv2D):
    # Check if layer is a Conv2D layer
    if not isinstance(layer, tf.keras.layers.Conv2D):
      return False

    # Check if depthwise separable convolution can be used
    if layer.groups != 1 or layer.filters == layer.input_shape[-1] or layer.kernel_size == (1,1):
      return False
    
    # check if the stride constraints can be accounted for
    if layer.strides[0] != layer.strides[-1]:
      log.warn(f"refusing layer {layer.name} due to its stride configuration that is currently not supported by the TensorFlow kernels")
      return False

    return True

def is_suitable_layer_node(layer : gt.LayerNode):
  if layer.layer_type != "Conv2D":
    return False
  
  return _is_depthwise_separable_keras_layer(layer.keras)

def convert2ds_conv2d(layer : tf.keras.layers.Conv2D):
  if layer.strides[0] != layer.strides[1]:
    raise ValueError("Strides must be equal for depthwise-separable convolutions.")
  
  # Depthwise Convolution with depth_multiplier=1, which means the number of output filters is equal to the number of input channels
  depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=layer.kernel_size,
                                    strides=layer.strides,
                                    padding=layer.padding,
                                    depth_multiplier=1,
                                    use_bias=False,
                                    activation=None,
                                    name = f"{layer.name}_depthwise")#(layer.input)

  # Pointwise Convolution with the number of filters equal to the number of output channels of the Depthwise Convolution
  pointwise_conv = tf.keras.layers.Conv2D(filters=layer.filters,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding=layer.padding,
                          use_bias=layer.use_bias,
                          activation=None,
                          name = f"{layer.name}_pointwise")#(depthwise_conv)

  return [depthwise_conv, pointwise_conv]

def compute_ds_macs(layer : tf.keras.layers.Conv2D):
  # Get the input and output shapes of the convolutional layer
  input_shape = layer.input_shape[1:]
  output_shape = layer.output_shape[1:]

  # Get the kernel size of the convolutional layer
  kernel_size = layer.kernel_size[0]

  # Compute the number of MAC operations required by a depthwise-separable convolutional layer
  depthwise_MACs = input_shape[0] * input_shape[1] * input_shape[2] * kernel_size * kernel_size
  pointwise_MACs = output_shape[0] * output_shape[1] * output_shape[2] * input_shape[2]
  total_MACs = depthwise_MACs + pointwise_MACs

  return total_MACs

'''def exchange_layer(old_layer : tf.keras.layers.Layer, new_subgraph : List[tf.keras.layers.Layer], orig_model : tf.keras.models.Model) -> Tuple[tf.keras.models.Model, Set[int]]:
  # Retrieve the optimizer, loss, and metrics from the original model
  optimizer = type(orig_model.optimizer).from_config(orig_model.optimizer.get_config())
  loss = orig_model.loss
  metrics = ["accuracy"]

  model = tf.keras.models.clone_model(orig_model)
  model.build(orig_model.input_shape)
  model.set_weights(orig_model.get_weights())

  # Create a new model with the same architecture as the original model
  model_tensors = {}
  produced = []
  consumed = []

  inputs = [tf.keras.Input(shape=inp.shape[1:], name=inp.name) for inp in model.inputs]#tf.keras.Input(shape=model.input_shape[1:])
  #x = inputs #model.layers[0](inputs)
  for inp in inputs:
    model_tensors[inp.name] = inp
    produced.append(inp)

  for layer in model.layers[1::]:
    
    if isinstance(layer.input, list):
      inp_names = [tensor.node.layer.name for tensor in layer.input]
      inp_tensors = [model_tensors[name] for name in inp_names]
    else:
      inp_name = layer.input.node.layer.name
      inp_tensors = model_tensors[inp_name]
    x = inp_tensors

    if layer.name == old_layer.name:
      # Replace the Conv2D layer with the new DS Conv2D layer
      for new_layer in new_subgraph:
        if isinstance(x, list):
          consumed += x
        else:
          consumed.append(x)
        x = new_layer(x)
        model_tensors[layer.name] = x
        produced.append(x)

    else:
      if isinstance(x, list):
        consumed += x
      else:
        consumed.append(x)
      x = layer(x)
      model_tensors[layer.name] = x
      produced.append(x)

  output_names = set([tens.name for tens in produced]) - set([tens.name for tens in consumed])
  outputs = [tens for tens in produced if tens.name in output_names]
  new_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

  # Compile the modified model using the original compile parameters
  new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  ids = []
  for new_layer in new_subgraph:
    ids.append(new_model.layers.index(new_layer))

  return new_model, set(ids)'''

def update_model_to_ds_version_of(conv_layer : tf.keras.layers.Conv2D, orig_model : tf.keras.models.Model) -> tf.keras.models.Model:

  new_layers = convert2ds_conv2d(conv_layer)
  new_model, new_layer_ids = exchange_layer(old_layer=conv_layer, new_subgraph=new_layers, orig_model=orig_model)

  return new_model, new_layer_ids

class SmartConvSolution(base.Solution):
  """
  A solution for the SmartConvRewriter.
  This solution contains a version of the submitted model where the most expensive Conv2D layer was replaced by a cheaper depthwise separable Conv2D version of it.
  The retraining cost was minimized by focusing the training on the weights of the newly introduced layers.
  """

  model : tf.keras.models.Model
  """The new model created by the rewriter."""

  location : gt.LayerNode
  """The layer node where the Conv2D workload was replaced."""

  train_config : Dict[str, object]
  """The training config that will be applied during finetuning."""

  finetune_epochs: int
  """The finetuning epochs that have been applied to this solution."""

  new_layer_ids : Set[int]
  """The ids of the replaced layers."""

  def __init__(self, rewriter : base.Rewriter, new_model : tf.keras.models.Model, new_layer_ids : Set[int], location: gt.LayerNode, train_config : Dict[str, object] = None) -> None:
    super().__init__()
    self.finetune_epochs = 0
    self.rewriter = rewriter
    self.model = new_model
    self.location = location
    self.new_layer_ids = new_layer_ids

    # enable trainability for all model layers
    for i, layer in enumerate(self.model.layers):
      layer.trainable = True

    if train_config is not None:
      self.train_config = train_config
    else:
      self.train_config = self.rewriter._find_training_config()

  def finetune(self, only_new_layers : bool = False):
    """
    Finetunes the weights of the found solution.
    Can select between just tuning the newly introduced layers or all layers.

    Args:
        only_new_layers (bool, optional): If True, only the depthwise separable section will be trained. Defaults to False.

    Returns:
        object: The training history created by the model.fit function.
    """
    train_dataset = self.rewriter.test_dataset.data.batch(self.train_config["batch_size"])
    self.train_config["epochs"] = 1

    if only_new_layers:
      # store current trainable states as well
      prev_train_state = dict()

      for i, layer in enumerate(self.model.layers):
        # check if the layer corresponds to one of the indices in the "ids" set
        prev_train_state[i] = layer.trainable
        if i not in self.new_layer_ids:
          # if not, freeze the layer's weights
          
          layer.trainable = False
    
    self.finetune_epochs += 1
    history = self.model.fit(train_dataset, **self.train_config)

    if only_new_layers:
      for i, layer in enumerate(self.model.layers):
        layer.trainable = prev_train_state[i]
    
    return history
  
  def evaluate(self) -> Dict[str, object]:
    """
    Evaluates the performance of the solution on the available test data.

    Returns:
        Dict[str, object]: The evaluation results.
    """
    test_dataset = self.rewriter.test_dataset.data.batch(int(self.train_config["batch_size"]*2))
    return self.model.evaluate(test_dataset, return_dict=True)

  def toDict(self, with_model : bool = True):
    data_dict = dict()
    if with_model:
      data_dict["model"] = self.model
    else:
      data_dict["model"] = self.model.name
    data_dict["train_config"] = self.train_config
    data_dict["finetune_epochs"] = self.finetune_epochs
    data_dict["new_layer_ids"] = self.new_layer_ids
    data_dict["location"] = self.location.name
    data_dict["eval"] = self.evaluate()
    data_dict["full_model_footprint"] = int(sum([res.get_layer_macs(layer) for layer in self.model.layers]))

    return data_dict
  
  def dump(self, path : Union[str, pathlib.Path]) -> Dict[str, object]:
    """
    Writes the solution to disk.

    Args:
        path (Union[str, pathlib.Path]): The folder in which the solution will be stored.

    Returns:
        Dict[str, object]: A dict representation of the solution
    """
    import json

    if isinstance(path, str):
      path = pathlib.Path(path)

    model_path = path / f"optimized_{self.location.name}_model.tf"
    self.model.save(model_path)

    data = self.toDict(with_model=False)
    data["model_path"] = str(model_path)
    data["new_layer_ids"] = list(self.new_layer_ids)
    data["evaluation"] = self.evaluate()

    with open(path / f"optimized_{self.location.name}_info.json", "w") as file:
      json.dump(data, file)

    return data
  
  def to_analysis(self) -> List[object]:
    """
    Converts the created Solutions into new ModelAnalysis objects.

    Returns:
        List[object]: The created ModelAnalysis obejcts.
    """
    model = self.model
    name = self.rewriter.analysis.name + ":DSConv2D"
    mac_estimators = self.rewriter.analysis.compute.mac_estimators
    cache_dir = self.rewriter.analysis.cache_dir


    analysis = aNN.ModelAnalysis(model=model,
                                 name=name,
                                 mac_estimators=mac_estimators,
                                 cache_dir=cache_dir
                                )
    
    return [analysis]


class SmartConvRewriter(base.Rewriter):
  """
  A rewriter that replaces the most costly Conv2D layer with a depthwise separable implementation if possible.
  Minimizes the retraining efforts by keeping the weights of the other layers frozen for most of the time.
  """

  train_dataset : d.DatasetReport
  """The dataset reporter used for training"""
  validation_dataset : d.DatasetReport
  """The dataset reporter used for validation"""

  test_dataset : d.DatasetReport
  """The dataset reporter used for evaluation of found solutions"""

  options : Set[gt.BlockNode] # or use keras.layers directly?
  """Locations where a ds-conv2d layer could be introduced into the architecture."""

  model : tf.keras.models.Model
  """The input model."""

  ranking : List[gt.LayerNode]
  """Ranking of the found options by performance (accuracy + efficiency)"""

  old_costs : Dict[gt.LayerNode, int]
  """Cost of the options before the replacement."""

  new_costs : Dict[gt.LayerNode, int]
  """Cost of the options after the replacement."""

  solutions : Dict[gt.LayerNode, SmartConvSolution]
  """Solutions that have been created by the rewriter."""

  def __init__(self, analysis, train_dataset : d.DatasetReport, test_dataset : d.DatasetReport, validation_dataset : d.DatasetReport = None) -> None:
    super().__init__(analysis)
    self.old_costs = {}
    self.new_costs = {}
    self.model = self.analysis.keras_model

    self.train_dataset = train_dataset
    self.validation_dataset = validation_dataset
    self.test_dataset = test_dataset

    if validation_dataset is None:
      self.validation_dataset = train_dataset
    # create search space
    ## identify all Conv2D layers in the model
    self.options = self.__identify_relevant_layers()
    self.ranking = self.rank_options()
    
    self.solutions : Dict[gt.BlockNode, SmartConvSolution] = dict()

  def __identify_relevant_layers(self) -> Dict[gt.BlockNode, Set[gt.LayerNode]]:
    """
    Private function. Identifies suitable Conv2D layers that can be replaced.

    Returns:
        Dict[gt.BlockNode, Set[gt.LayerNode]]: the options that were found in the model architecture.
    """
    self.analysis : aNN.ModelAnalysis
    blocks : List[gt.BlockNode] = [node for node in self.analysis.architecture.block_graph.nodes if node.dominant_operation == "convolution"]

    options = {}
    costs = {}

    for block in blocks:
      # need to see if it contains a Conv2D layer
      # store its cost
      local_options = set()
      for layer in block.subgraph.nodes:
        if is_suitable_layer_node(layer):
          log.debug("found one! : ", layer)
          old_cost = res.get_layer_macs(layer=layer.keras)
          new_cost = compute_ds_macs(layer=layer.keras)

          if new_cost < old_cost:
            local_options.add(layer)
            self.old_costs[layer] = old_cost
            self.new_costs[layer] = new_cost
      if len(local_options) > 0:
        options[block] = local_options

    return options
  
  def _find_training_config(self) -> Dict[str, object]:
    """
    Tries to automatically derive the best training configuration for the newly found solutions.
    Currently fixe to 6 epochs and a batch size of 128.

    Returns:
        Dict[str, object]: the training config as dict that can be passed to the fit function via model.fit(**train_config)
    """
    train_config = {
      "epochs" : 6,
      "batch_size" : 128,
    }

    return train_config

  def rank_options(self) -> List[gt.LayerNode]:
    """
    Ranks the found options based on their maintained accuracy and relative inference cost.

    Returns:
        List[gt.LayerNode]: the lsit of options sorted by their reward.
    """
    option_layers = []
    for block, layer_nodes in self.options.items():
      option_layers += list(layer_nodes)

    return sorted(option_layers, key=lambda x: self.new_costs[x] - self.old_costs[x])

  def search(self, initial_epochs : int = 6, full_epochs : int = 1) -> SmartConvSolution:
    """
    Finds a possible solution for the conv rewrite.

    Args:
        initial_epochs (int, optional): the training epochs spent exclusively on the weights of the replaced layers. Defaults to 6.
        full_epochs (int, optional): the epochs spend optimizing the entire model. Defaults to 1.

    Returns:
        SmartConvSolution: the found solution.
    """

    train_config = self._find_training_config()

    if len(self.ranking) == 0:
      log.warn("No error could be replaced, returning original model as new solution")
      return SmartConvSolution(self, new_model=self.analysis.keras_model, new_layer_ids=[], location=[], train_config=train_config)

    #find best option
    selected_option = self.ranking[0]
    old_layer = selected_option.keras
    new_layers = convert2ds_conv2d(old_layer)
    # replace workload
    new_model, layer_ids = exchange_layer(old_layer, new_layers, self.model)

    # train new model
    
    train_config["epochs"] = initial_epochs

    ## only train new layers to adapt them to existing weights
    for i, layer in enumerate(new_model.layers):
        # check if the layer corresponds to one of the indices in the "ids" set
        if i not in layer_ids:
            # if not, freeze the layer's weights
            layer.trainable = False

    new_model.summary()
    new_model.fit(self.train_dataset.data.batch(train_config["batch_size"]), **train_config)

    ## train full model, nothing frozen
    for i, layer in enumerate(new_model.layers):
      layer.trainable = True

    new_solution = SmartConvSolution(self, new_model=new_model, new_layer_ids=layer_ids, location=selected_option, train_config=train_config)
    new_solution.finetune(only_new_layers=False)
    self.solutions[selected_option] = new_solution

    return self.solutions[selected_option]

  def create_identifier(self) -> str:
    """
    Creates a unique identifier for the rewriter.

    Returns:
        str: The unique ID.
    """
    descriptor = f"SmartConv:{self.analysis.keras_model.name}"
    #TODO: extend if required

    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the input string
    sha256_hash.update(descriptor.encode('utf-8'))
    hashed_representation = sha256_hash.hexdigest()

    return hashed_representation
  
  def dump(self, folder_path: Union[str ,pathlib.Path] = None):
    """Dumps the rewriter to disk."""
    import json

    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    solutions_path = folder_path / f"rewriter_smart_convolution_{self.create_identifier()}/"
    solutions_path.mkdir(parents=True, exist_ok=True)

    summary = {
      "report_type": "Smart-Convolution",
      "name": self.analysis.name,
      "creation_date": str(self.analysis.creation_time),
      "solutions_path" : str(solutions_path),
      "solutions" : {location.name: (values.toDict(with_model=False)) for location, values in self.solutions.items()},
      "options" : [location.name for location in self.options],
      "old_costs": {location.name: int(cost) for location, cost in self.old_costs.items()},
      "new_costs": {location.name: int(cost) for location, cost in self.new_costs.items()},
      "ranking" : [location.name for location in self.ranking],
      "original_footprint" : int(sum([res.get_layer_macs(layer) for layer in self.model.layers])),
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
      elif isinstance(obj, set):
        return list(obj)
      else:
        return obj

    with open(folder_path / f"rewriter_smart_conv_{self.create_identifier()}.json", "w") as file:
      json.dump(summary, file, default=numpy_encoder)
    
    return summary
  
  def render_summary(self, folder_path: Union[str ,pathlib.Path] = None) -> Tuple[str, str]:
    """
    Creates a human-readible summary of the rewriting efforts as HTML file.

    Args:
        folder_path (Union[str ,pathlib.Path], optional): _description_. Defaults to None.

    Returns:
        Tuple[str, str]: the title of the summary when linked from the ModelAnalysis summary and the file name necessary to link it.
    """
    import os
    from jinja2 import Template
    
    _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'

    if folder_path is None:
      folder_path = pathlib.Path.cwd()

    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    with open(_template_path / "smart_conv_rewriter.html", "r") as file:
      template = Template(file.read())

    summary = self.dump(folder_path=folder_path)

    # Render the template with the summary data
    html = template.render(summary=summary)
    # Save the generated HTML to a file
    html_filename = f"rewriter_smart_conv_{self.create_identifier()}.html"
    html_path = folder_path / html_filename
    with open(html_path, "w") as file:
        file.write(html)

    return (
        "Smart Conv",
        html_filename,
    )
  
  @classmethod
  def create_pass(cls, train_data:tf.data.Dataset, valid_data:tf.data.Dataset, test_data:tf.data.Dataset) -> Tuple[str , callable]:
    """
    Creates a pass that can be added to the optimization queue of a model analysis object.
    This pass contains all steps necessary to produce a solution of this rewrite from the ModelAnalysis of the submitted model.

    Returns a name for the pass as well as the function that needs to be called to apply the rewrite to the analysis

    Args:
        train_data (tf.data.Dataset): The training data report wrapper.
        valid_data (tf.data.Dataset): The validation data report wrapper.
        test_data (tf.data.Dataset): The test data report wrapper.

    Returns:
        Tuple[str, callable]: The pass name as well as its callable implementation.
    """
    
    str_id = f"ConvRewrite"

    def rewrite(analysis:aNN.ModelAnalysis) -> SmartConvSolution:

      train_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="train", modality=None).from_source(tf_dataset=train_data)

      valid_data_report = None
      if valid_data is not None:
          valid_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="valid", modality=None).from_source(tf_dataset=valid_data)

      test_data_report = None
      if test_data is not None:
          test_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="test", modality=None).from_source(tf_dataset=test_data)

      rewriter = SmartConvRewriter(analysis=analysis, train_dataset=train_data_report, validation_dataset=valid_data_report, test_dataset=test_data_report)

      solution = rewriter.search()

      return solution
    
    return str_id, rewrite