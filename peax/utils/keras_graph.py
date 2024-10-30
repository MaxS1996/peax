import tensorflow as tf
from typing import Union, Dict, List, Set, Tuple
import numpy as np

from peax.components import graph_tools as gt
from peax.components import architecture as arch

import logging as log

def exchange_layer(old_layer : tf.keras.layers.Layer, new_subgraph : List[tf.keras.layers.Layer], orig_model : tf.keras.models.Model, share_layers:bool=True) -> Tuple[tf.keras.models.Model, Set[int]]:
  # Retrieve the optimizer, loss, and metrics from the original model
  is_compiled = False
  if orig_model.optimizer is not None:
    optimizer = type(orig_model.optimizer).from_config(orig_model.optimizer.get_config())
    is_compiled = True
  if hasattr(orig_model, "loss"):
    loss = orig_model.loss
    is_compiled = True
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

  if not share_layers:
    weights = {}

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

      if share_layers:
        x = layer(x)
      else:
        layer_type = type(layer)
        layer_config = layer.get_config()

        if len(layer.get_weights()) > 0:
          weights[layer.name] = layer.get_weights()
        x = layer_type.from_config(layer_config)(x)

      model_tensors[layer.name] = x
      produced.append(x)

  output_names = set([tens.name for tens in produced]) - set([tens.name for tens in consumed])
  outputs = [tens for tens in produced if tens.name in output_names]
  new_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

  if not share_layers:
    for layer in new_model.layers:
      if layer.name in weights.keys():

        new_weights = layer.get_weights()
        if len(new_weights) != len(weights[layer.name]):
          log.info(f"new layer named {layer.name} has a different number of weight tensors compared to its original version")

        updated_weights = weights[layer.name]

        if isinstance(layer, tf.keras.layers.BatchNormalization):
          for w_id, weight_array in enumerate(weights[layer.name]):
            new_shape = new_weights[w_id].shape
            if new_shape != weight_array.shape:
              log.info(f"weight tensor {w_id} of BatchNorm layer {layer.name} changed shape during layer exchange, attempting to fix it!")

              new_tensor = np.zeros_like(new_weights[w_id])
              new_tensor = weight_array[0:new_shape[0]]
              updated_weights[w_id] = new_tensor

        elif isinstance(layer, (tf.keras.layers.DepthwiseConv1D, tf.keras.layers.Conv1D)):
          weight_array = weights[layer.name][0]
          new_shape = new_weights[0].shape
          old_shape = weight_array.shape

          if new_shape != old_shape:
            log.info(f"filter weights of Conv1D layer {layer.name} changed shape during layer exchange, attempting to fix it!")

            new_tensor = np.zeros_like(new_weights[0])
            # check filter sizes
            if new_shape[0] != old_shape[0] or new_shape[-1] != old_shape[-1]:
              log.error("filter size changed, this should not happen when switching layers!")
            
            if new_shape[1] != old_shape[1]: # filter count changed
              new_tensor = weight_array[:,0:new_shape[1],:]
              updated_weights[0] = new_tensor

          # check bias weights
          weight_array = weights[layer.name][1]
          new_shape = new_weights[1].shape
          old_shape = weight_array.shape

          if new_shape != old_shape:
            log.info(f"bias weights of Conv1D layer {layer.name} changed shape during layer exchange, attempting to fix it!")

            new_bias = weight_array[0:new_shape[0]]
            updated_weights[1] = new_bias


        elif isinstance(layer, (tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2D)):
          # check filter weights
          weight_array = weights[layer.name][0]
          new_shape = new_weights[0].shape
          old_shape = weight_array.shape
          if new_shape != old_shape:
            log.info(f"filter weights of (Depthwise)Conv2D layer {layer.name} changed shape during layer exchange, attempting to fix it!")

            new_tensor = np.zeros_like(new_weights[0])
            # check filter sizes
            if new_shape[0] != old_shape[0] or new_shape[1] != old_shape[1]:
              log.error("filter size changed, this should not happen when switching layers!")
            
            if new_shape[2] != old_shape[2]: # filter count changed
              new_tensor = weight_array[:,:,0:new_shape[2],:]
              updated_weights[0] = new_tensor

          # check bias weights
          weight_array = weights[layer.name][1]
          new_shape = new_weights[1].shape
          old_shape = weight_array.shape

          if new_shape != old_shape:
            log.info(f"bias weights of (Depthwise)Conv2D layer {layer.name} changed shape during layer exchange, attempting to fix it!")

            new_bias = weight_array[0:new_shape[0]]
            updated_weights[1] = new_bias
        elif isinstance(layer, (tf.keras.layers.Dense)):
          # check filter weights
          weight_array = weights[layer.name][0]
          new_shape = new_weights[0].shape
          old_shape = weight_array.shape

          if new_shape != old_shape:
            log.info(f"filter weights of Dense layer {layer.name} changed shape during layer exchange, attempting to fix it!")
            updated_weights[0] = weight_array[0:new_shape[0], :]
          
          # check bias weights
          weight_array = weights[layer.name][1]
          new_shape = new_weights[1].shape
          old_shape = weight_array.shape

          if new_shape != old_shape:
            log.info(f"bias weights of Dense layer {layer.name} changed shape during layer exchange, attempting to fix it!")
            updated_weights[1] = weight_array[0:new_shape[0]]

        else:
          weight_array = weights[layer.name][0]
          new_shape = new_weights[0].shape
          old_shape = weight_array.shape
          if new_shape != old_shape:
            raise NotImplementedError(f"layer {layer.name} changed its weight tensor shapes, but its adaption has not yet been implemented")

        layer.set_weights(updated_weights) # weights[layer.name]

  # Compile the modified model using the original compile parameters
  if is_compiled:
    new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  ids = []
  for new_layer in new_subgraph:
    ids.append(new_model.layers.index(new_layer))

  return new_model, set(ids)

def attach_branch(original_model : tf.keras.models.Model, attachment_layer : Union[str, tf.keras.layers.Layer], branch : tf.keras.models.Model, reorder:bool=False) -> tf.keras.models.Model:
  """Function to attach a single branch to your base model

  Args:
      original_model (tf.keras.models.Model): The backbone model to which the branch should be attached
      attachment_layer (Union[str, tf.keras.layers.Layer]): The name or layer of the backbone model after which the branch should be attached
      branch (tf.keras.models.Model): the model that describes the added branch
      reorder(bool): if True, the added outputs will come before the original classifer when creating result arrays or calling model.outputs.

  Returns:
      tf.keras.models.Model: the new model with an additional exit
  """
  backbone = tf.keras.models.clone_model(original_model)
  backbone.build(original_model.input_shape)
  backbone.set_weights(original_model.get_weights())

  input_tensors = backbone.inputs
  all_outputs = backbone.outputs

  if isinstance(attachment_layer, tf.keras.layers.Layer):
    attachment_layer = attachment_layer.name

  attach_layer = backbone.get_layer(attachment_layer)
  x = attach_layer.output

  source_weights : Dict[str, np.array] = dict()
  for layer in branch.layers[1::]:
    config = layer.get_config()
    config["name"] = f"{attach_layer.name}_{config['name']}"
    source_weights[config["name"]] = layer.get_weights()

    x = type(layer).from_config(config)(x)
  all_outputs.append(x)

  if reorder:
    all_outputs.reverse()

  new_model = tf.keras.Model(inputs=input_tensors, outputs=all_outputs)
  for layer_name, weights in source_weights.items():
    new_model.get_layer(layer_name).set_weights(weights)

  return new_model

def attach_branches(original_model : tf.keras.models.Model, attachment_layers : List[Union[str, tf.keras.layers.Layer]], branches : List[tf.keras.models.Model], reorder:bool=False) -> tf.keras.models.Model:
  """Function to attach multiple branches to your base model

  Args:
      original_model (tf.keras.models.Model): The backbone model to which the branches will be attached
      attachment_layers (List[Union[str, tf.keras.layers.Layer]]): the locations where the branches are going to be attached
      branches (List[tf.keras.models.Model]): the configurations of the attached branches described by compiled keras models
      reorder (bool): if True, the outputs will be sorted in the order of occurrence in the network graph
  Returns:
      tf.keras.models.Model: the newly created model with additional branches
  """
  
  backbone = tf.keras.models.clone_model(original_model)
  backbone.build(original_model.input_shape)
  backbone.set_weights(original_model.get_weights())

  input_tensors = backbone.inputs
  all_outputs = backbone.outputs

  if reorder:
     all_outputs = []

  source_weights : Dict[str, np.array] = dict()
  for attach_layer, ee_branch in zip(attachment_layers, branches):
    attach_layer = backbone.get_layer(attach_layer)
    
    x = attach_layer.output
    for layer in ee_branch.layers[1::]:
        config = layer.get_config()
        config["name"] = f"{attach_layer.name}_{config['name']}"
        source_weights[config["name"]] = layer.get_weights()

        x = type(layer).from_config(config)(x)

    # Add the exit branch to the model
    all_outputs.append(x)

  # Create the new Keras model
  if reorder:
     all_outputs += backbone.outputs
     
  new_model = tf.keras.Model(inputs=input_tensors, outputs=all_outputs)

  # write weights back
  for layer_name, weights in source_weights.items():
      new_model.get_layer(layer_name).set_weights(weights)

  return new_model

def split_eenn_model(model : tf.keras.models.Model, split_points : List[gt.BlockNode]) -> List[tf.keras.models.Model]:
  eenn_model = tf.keras.models.clone_model(model)
  eenn_model.build(model.input_shape)
  eenn_model.set_weights(model.get_weights())

  split_layer_names = [gt.get_output_nodes(loc.subgraph)[0].keras.name for loc in split_points]
  inp_layers = eenn_model.inputs

  connection_layers = []
  for name in split_layer_names:
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

    sub_model = tf.keras.Model(inputs=inp_layers, outputs=out_layers, name=f"{model.name}-{idx}")
    new_submodel = tf.keras.models.clone_model(sub_model)
    new_submodel.build(sub_model.input_shape)
    new_submodel.set_weights(sub_model.get_weights())

    subgraph_models.append(new_submodel)

    connection_layer = eenn_model.get_layer(connect.node.layer.name)
    inp_layers = [tf.keras.layers.Input(shape=connection_layer.output_shape, tensor=connection_layer.output)]

  #last submodel
  final_submodel = tf.keras.Model(inputs=inp_layers, outputs=[exit_layers[-1]], name=f"{model.name}-{idx+1}")

  new_submodel = tf.keras.models.clone_model(final_submodel)
  new_submodel.build(final_submodel.input_shape)
  new_submodel.set_weights(final_submodel.get_weights())
  subgraph_models.append(new_submodel)

  return subgraph_models

