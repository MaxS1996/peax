import tensorflow as tf
from typing import List, Tuple, Union

class SlimmableConv2d(tf.keras.layers.Layer):
  def __init__(
    self,
    filters_list:List[int],
    kernel_size:Union[int, Tuple[int]],
    strides:Union[int, Tuple[int]]=(1, 1),
    padding="same",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    activation="relu",
    **kwargs
  ):
    super(SlimmableConv2d, self).__init__(**kwargs)

    self.filters_list = filters_list
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding

    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation

    self.active_indices = [1 for _ in filters_list]
    self.conv_layers = []
    for filters in filters_list:
      new_layer = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        activation=activation,
        **kwargs,
      )
      self.conv_layers.append(new_layer)

  def call(self, inputs, training=False):
    while len(self.active_indices) < len(self.conv_layers):
      self.active_indices.append(0)

    outs = []
    for idx in range(len(self.conv_layers)):
      tmp_conv = self.conv_layers[idx](inputs, training=training)
      if self.active_indices[idx] >= 1:
        outs.append(tmp_conv)
      else:
        # Create a zero tensor with the same shape as the output of the convolution
        zero_tensor = tf.zeros_like(tmp_conv)
        outs.append(zero_tensor)

    outs = tf.concat(outs, axis=-1)

    return outs

  def set_active_indices(self, indices):
    self.active_indices = indices

    active_filters = [
      state * filters
      for (state, filters) in zip(self.active_indices, self.filters_list)
    ]

    print(f"ACTIVE_WIDTH:{self.active_indices} - equals filters {active_filters}")

  def get_active_indices(self):
    return self.active_indices

  def freeze_active(self):
    for idx, conv_layer in enumerate(self.conv_layers):
      if self.active_indices[idx] >= 1:
        conv_layer.trainable = False
        print(f"FROZE:   {conv_layer.name}")
      else:
        conv_layer.trainable = True
        print(f"UNFROZE: {conv_layer.name}")

class SlimmableDense(tf.keras.layers.Layer):
  def __init__(
      self,
      units_list:List[int],
      activation:str="relu",
      kernel_initializer:str = "glorot_uniform",
      bias_initializer:str = "zeros",
      **kwargs) -> None:
    super(SlimmableDense, self).__init__(**kwargs)

    self.units_list = units_list
    self.activation = activation

    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

    self.active_indices = [1 for _ in units_list]
    self.dense_layers = []
    for units in units_list:
      new_layer = tf.keras.layers.Dense(
        units=units,
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        **kwargs,
      )
      self.dense_layers.append(new_layer)

    pass

  def call(self, inputs, training=False):
    while len(self.active_indices) < len(self.dense_layers):
      self.active_indices.append(0)

    outs = []
    for idx in range(len(self.dense_layers)):
      tmp_dense = self.dense_layers[idx](inputs, training=training)
      if self.active_indices[idx] >= 1:
        outs.append(tmp_dense)
      else:
        # Create a zero tensor with the same shape as the output of the dense layer
        zero_tensor = tf.zeros_like(tmp_dense)
        outs.append(zero_tensor)

    outs = tf.concat(outs, axis=-1)
    return outs

  def set_active_indices(self, indices):
    self.active_indices = indices

    active_units = [
        state * units
        for (state, units) in zip(self.active_indices, self.units_list)
    ]

    print(f"ACTIVE_UNITS:{self.active_indices} - equals units {active_units}")

  def get_active_indices(self):
    return self.active_indices

  def freeze_active(self):
    for idx, dense_layer in enumerate(self.dense_layers):
      if self.active_indices[idx] >= 1:
        dense_layer.trainable = False
        print(f"FROZE:   {dense_layer.name}")
      else:
        dense_layer.trainable = True
        print(f"UNFROZE: {dense_layer.name}")