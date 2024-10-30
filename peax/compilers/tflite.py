import tensorflow as tf
from .compile_backend import Backend

import pathlib
import logging as log

class TFLiteBackend(Backend):
  #converter : tf.lite.TFLiteConverter = None
  ops : tf.lite.OpsSet = None
  opt : tf.lite.Optimize = tf.lite.Optimize.DEFAULT

  supported_ops = {
    "int16-int8" : tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
    "default" : tf.lite.OpsSet.TFLITE_BUILTINS,
    "int8-int8": tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
  }

  opt_targets = {
    "default" : tf.lite.Optimize.DEFAULT,
    "sparsity" : tf.lite.Optimize.EXPERIMENTAL_SPARSITY,
    "latency" : tf.lite.Optimize.OPTIMIZE_FOR_LATENCY,
    "size" : tf.lite.Optimize.OPTIMIZE_FOR_SIZE,
  }
      

  def __init__(self, ops: str = "default", optimize_for : str = None) -> None:
    if ops in self.supported_ops.keys():
      self.ops = self.supported_ops[ops]
    else:
      self.ops = self.supported_ops["default"]
      log.warn(f"unknown OpsSet {ops}, using 'default' instead")
    
    if optimize_for in self.opt_targets.keys():
      self.opt = self.opt_targets[optimize_for]
      if optimize_for in ("latency", "size"):
        log.warn("optimize for latency or size are deprecated since TF 2.??, same behavior as default flag")
      if optimize_for == "sparsity":
        log.warn("optimize for sparsity is an experimental feature of the TFLite converter, expect the unexpected!")
    else:
      self.opt = None
      log.info(f"unknown OptimizationSpec {optimize_for} or no optimization set, using None instead")

    super().__init__()

  def compile(self, model : tf.keras.models.Model, calibration_data : tf.data.Dataset = None, full_quantization : bool = False):
    converter = tf.lite.TFLiteConverter.from_model(model)

    # Set the opsSet
    converter.target_spec.supported_ops = self.ops

    if calibration_data != None:
      def representative_data_gen():
        for input_value in calibration_data.unbatch().batch(1).take(400):
          # Model has only one input so each data point has one element.
          yield [input_value]
      converter.representative_dataset = representative_data_gen

    # Set optimization options
    if self.opt != None:
      converter.optimizations = [self.opt]

    if full_quantization:
      converter.inference_input_type = tf.uint8
      converter.inference_output_type = tf.uint8
      
    # Additional conversion and compilation steps can be added here
    compiled_model = converter.convert()
    return compiled_model

  def quantize(self, model: tf.keras.models.Model, calibration_data : tf.data.Dataset = None, full_quantization : bool = False):
    log.info("quantize wraps the compile function for the TFLite backend")
    return self.compile(model=model, calibration_data=calibration_data)
  
  def optimize(self, model: tf.keras.models.Model, calibration_data : tf.data.Dataset = None, full_quantization : bool = False):
    log.info("optimize wraps the compile function for the TFLite backend")
    return self.compile(model=model, calibration_data=calibration_data)
  
  def invoke(self, model: tf.keras.models.Model, calibration_data : tf.data.Dataset = None, full_quantization : bool = False):
    log.info("invoke wraps the compile function for the TFLite backend")
    return self.compile(model=model, calibration_data=calibration_data)
  
  def store(self, model: tf.keras.models.Model, path : pathlib.Path, calibration_data : tf.data.Dataset = None, full_quantization : bool = False):
    result = self.compile(model=model, calibration_data=calibration_data)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the converted model:
    tflite_model_file = path
    tflite_model_file.write_bytes(result)

    return