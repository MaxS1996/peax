import tensorflow as tf
import numpy as np

import logging as log

def _transfer_arrays(old_weights:np.array, new_weights:np.array) -> np.array:
  # get the minimum shape between old_weights and new_weights
  min_shape = tuple(min(a, b) for a, b in zip(old_weights.shape, new_weights.shape))

  # slice old_weights to the minimum shape
  old_weights_slice = old_weights[tuple(slice(0, s) for s in min_shape)]

  # copy the sliced old_weights into new_weights
  new_weights[tuple(slice(0, s) for s in min_shape)] = old_weights_slice

  return new_weights

def copy_weights(old_layer:tf.keras.layers.Layer, new_layer:tf.keras.layers.Layer) -> None:
  if len(old_layer.get_weights()) == 0:
    return new_layer
  
  if isinstance(old_layer, tf.keras.layers.BatchNormalization):
    log.info("skipping copying of BatchNorm config, as it will break Normalization behavior")
    return new_layer
  
  old_weights = old_layer.get_weights()
  new_weights = new_layer.get_weights()

  updated_weights = []
  for old, new in zip(old_weights, new_weights):
    updated_weights.append(_transfer_arrays(old_weights=old, new_weights=new))

  new_layer.set_weights(updated_weights)
  return new_layer

