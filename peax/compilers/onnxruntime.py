import tensorflow as tf
from .compile_backend import Backend

import onnx
import tf2onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, MinMaxCalibrater, QuantFormat, CalibrationMethod

import tempfile
import pathlib
import logging as log


class ONNXRuntime(Backend):
    
  '''
  model_fp32 = 'path/to/the/model.onnx'
  model_quant = 'path/to/the/model.quant.onnx'
  quantized_model = quantize_dynamic(model_fp32, model_quant)
  '''
    
  def compile(self, model : tf.keras.models.Model):
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model,
                input_signature=None, opset=None, custom_ops=None,
                custom_op_handlers=None, custom_rewriter=None,
                inputs_as_nchw=None, outputs_as_nchw=None, extra_opset=None,
                shape_override=None, target=None, large_model=False, output_path=None)
    return model_proto, external_tensor_storage

  def quantize(self, model : tf.keras.models.Model, dynamic:bool=False):
    onnx_model, external_tensor_storage = self.compile(model=model)
    #do we need to write the models to storage? sadly yes
    tmp_folder = tempfile.gettempdir()
    inp_model_path = tmp_folder / f"float_{model.name}.onnx"
    out_model_path = tmp_folder / f"quant_{model.name}.onnx"

    onnx_model.save(inp_model_path)
    
    if dynamic:
      quantize_dynamic(
        inp_model_path,
        out_model_path,
        op_types_to_quantize=None,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        optimize_model=True,
        use_external_data_format=False,
        extra_options=None,
      )
    else:
      calibration_reader = MinMaxCalibrater(
                          onnx_model,
                          augmented_model_path="calibration_model.onnx",
                          symmetric=False,
                          use_external_data_format=False)
      
      quantize_static(
        inp_model_path,
        out_model_path,
        calibration_reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=None,
        per_channel=True,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        optimize_model=False,
        use_external_data_format=False,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options=None,
    )
      
    quant_model = onnx.load(out_model_path)

    return quant_model

  def optimize(self, model : tf.keras.models.Model):
    pass

  def store(self, compiled_model : object, path : pathlib.Path):
    pass

  def invoke(self, model : tf.keras.models.Model):
    """Wraps the entire optimization and conversion process into one function

    Args:
        model (tf.keras.models.Model): The (sub)model that needs to be compiled
    """
    pass
