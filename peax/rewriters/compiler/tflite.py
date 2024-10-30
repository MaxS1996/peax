import enum
from typing import Dict, List, Union, Tuple
import pathlib
import json
import hashlib

import copy
import json
import os

from jinja2 import Template

import tensorflow as tf

from . import rewriter as compiler

from peax.reports import dataset as d
import peax.analysis as aNN

from enum import Enum
class QuantizationMode(Enum):
  """
  The possible Quantization modes for the TFLite Compiler Interface.
  """
  FLOAT = 1
  """
  This mode will convert the model, while maintaining the weights, activations and calculations in a float format.
  """
  INTERNAL = 2
  """
  This mode will convert the model and convert internal weights, activations and calculations into the int8 format, a quantization and deqauntization step will be added to the input and output of the model.
  Can reduce the latency and memory footprint of the model inference, even on platforms with a hardware FPU.
  """
  EXTERNAL = 3
  """
  The model will be converted to be entirely in an int8 format, the inputs need to be provided in the quantized format and the output will also be returned quantized.
  Will achieve best latency on MCUs.
  Beneficial, if an MCU without FPU is supposed to be targeted.
  """

class TFLiteCompilerSolution(compiler.CompilerSolution):
  """
  The solution created by the TFLiteCompilerInterface.
  """

  def __init__(self, rewriter : compiler.CompilerInterface, model:tf.keras.models.Model, name:str, config:Dict[str, object], flatbuffer):
    """
    The constructor for the TFLiteSolution class, should only be called by the TFLiteCompilerInterface.

    Args:
        rewriter (compiler.CompilerInterface): The CompilerInterface that created the solution object.
        model (tf.keras.models.Model): The original of the model that has been compiled.
        name (str): A name for the created solution
        config (Dict[str, object]): The compiler parameters that have been used to create the solution.
        flatbuffer (_type_): The created flatbuffer file, which will be loaded and executed by the TFLite Interpreter at runtime.
    """
    super().__init__(rewriter=rewriter, model=model, name=name, config=config)
    self.flatbuffer = flatbuffer
    self.id = self.get_identifier()

  def get_identifier(self):
    """
    Returns the unique identifier of the solution.

    Returns:
        str: The unique identifier of the solution.
    """
    config = {
      "rewriter" : self.rewriter.create_identifier(),
      "config" : dict([(key, str(x)) for key, x in self.config.items()])
    }
    sorted_config_str = json.dumps(config, sort_keys=True)
    hash_object = hashlib.sha256()
    hash_object.update(sorted_config_str.encode('utf-8'))
    config_hash = hash_object.hexdigest()

    return config_hash
  
  def convert_bytes_to_c_array(self) -> Tuple[str, str]:
    """
    Creates the C-source and -header file necessary to include the model source into an external IDE.
    Often required for TFLite for Microcontroller-based projects and bare-metal projects.

    Returns:
        Tuple[str, str]: The c-file containing the model data as array and a header file to be included into the project.
    """
    # Convert to C array
    c_array = ', '.join([f'0x{byte:02X}' for byte in self.flatbuffer])
    
    # Determine the array length
    array_length = len(self.flatbuffer)

    cleaned_name = self.rewriter.analysis.keras_model.name.replace(" ", "_")
    cleaned_name = self.rewriter.analysis.keras_model.name.replace(":", "_")
    cleaned_name = self.rewriter.analysis.keras_model.name.replace("-", "_")
    cleaned_name = self.rewriter.analysis.keras_model.name.replace(".", "_")
    
    # Generate the C and H file content
    c_file_content = f'''
#include "{cleaned_name}.h"

const unsigned char {cleaned_name}_tflite[] = {{
    {c_array}
}};

const unsigned int {cleaned_name}_tflite_len = {array_length};
'''

    h_file_content = f'''
#ifndef {cleaned_name.upper().replace(".", "_")}
#define {cleaned_name.upper().replace(".", "_")}

extern const unsigned char {cleaned_name}_tflite[];
#define MODEL_LEN {array_length}

#endif // {cleaned_name.upper().replace(".", "_")}
'''
    return c_file_content, h_file_content

  def get_config(self) -> Dict[str, str]:
    """
    Returns the config, which was used to create this solution.

    Returns:
        Dict[str, str]: A dict of the config parameters.
    """

    config = {
      "name" : self.name,
    }

    for key, val in self.config.items():
      config[key] = str(val)
      if isinstance(val, list):
        config[key] = [str(x) for x in val]

    return config

  def dump(self, folder_path : Union[str, pathlib.Path]):
    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    config_hash = self.id

    rel_path = f"compiled_{self.name}_{config_hash}"

    folder_path = folder_path / rel_path

    folder_path.mkdir(exist_ok=True, parents=True)
    flatbuf_path = folder_path / f"{self.model.name}.tflite"
    flatbuf_path.write_bytes(self.flatbuffer)

    config = {}

    for key, val in self.config.items():
      config[key] = str(val)
      if isinstance(val, list):
        config[key] = [str(x) for x in val]

    with open(folder_path / "config.json", "w") as file:
      json.dump(config, file)

    cleaned_name = self.rewriter.analysis.keras_model.name.replace(" ", "_")
    cleaned_name = self.rewriter.analysis.keras_model.name.replace(":", "_")
    cleaned_name = self.rewriter.analysis.keras_model.name.replace("-", "_")
    c_content, h_content = self.convert_bytes_to_c_array()

    with open(folder_path / f"{cleaned_name}.c", "w") as file:
      file.write(c_content)

    with open(folder_path / f"{cleaned_name}.h", "w") as file:
      file.write(h_content)

    return folder_path


class TFLiteCompilerInterface(compiler.CompilerInterface):
  """
  The CompilerInterface to convert a ModelAnalysis object using TensorFlow Lite.
  Can be used to target Cortex-A and Cortex-M-based targets or similar.
  """

  solutions : List[TFLiteCompilerSolution]
  """
  The solutions created by the CompilerInterface.
  """

  quantization_data : d.DatasetReport
  """
  Dataset that will be used for the post-training quanitzation step.
  """

  def __init__(self, analysis :aNN.ModelAnalysis, quantization_data:d.DatasetReport=None):
    """
    The constructor for the TFLiteCompilerInterface

    Args:
        analysis (aNN.ModelAnalysis): The ModelAnalysis to which the CompilerInterface object will be submitted
        quantization_data (d.DatasetReport, optional): The dataset object that will be used to guide the post-training quantization. Only needed, if you plan to create quantized solutions. Defaults to None.
    """
    super().__init__(analysis=analysis)
    self.quantization_data = quantization_data
    self.solutions = []

  def compile(self,
    optimizations: List[tf.lite.Optimize] = [tf.lite.Optimize.DEFAULT],
    target_spec: tf.lite.TargetSpec = None,
    quantize: QuantizationMode = QuantizationMode.FLOAT,
    quant_ops: List[tf.lite.OpsSet] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
  ) -> TFLiteCompilerSolution:
    """
    The compile command, which creates the TFLite solution.

    Args:
        optimizations (List[tf.lite.Optimize], optional): The TFLite optimization flags. Defaults to [tf.lite.Optimize.DEFAULT].
        target_spec (tf.lite.TargetSpec, optional): An optional TFLite target spec. Defaults to None.
        quantize (QuantizationMode, optional): The intended quantization mode, supports FLOAT, INTERNAL and EXTERNAL. Defaults to QuantizationMode.FLOAT.
        quant_ops (List[tf.lite.OpsSet], optional): The Ops that are supposed to be used, if the model will be quantized, defaults to int8, as it is the only non-experimential that is supported by CMSIS. Defaults to [tf.lite.OpsSet.TFLITE_BUILTINS_INT8].

    Returns:
        TFLiteCompilerSolution: The resulting solution that wraps the TFLite flattbuffer file.
    """
    
    model = self.analysis.keras_model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = optimizations
    
    if target_spec is not None:
      converter.target_spec = target_spec
    
    if quantize != QuantizationMode.FLOAT:
      def representative_data_gen():
        for input_value, y in self.quantization_data.data.batch(1).take(250):
          # Model has only one input so each data point has one element.
          yield [input_value]

      converter.representative_dataset = representative_data_gen
      converter.target_spec.supported_ops = quant_ops
      if quantize == QuantizationMode.EXTERNAL:
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
      quant_ops = None

    tflite_model_data = converter.convert()

    config = {
      "optimizations" : [str(opt) for opt in optimizations],
      "target_spec" : target_spec,
      "quantize" : quantize,
      "quant_ops" : quant_ops
    }

    if quantize == QuantizationMode.FLOAT:
      del config["quant_ops"]

    sol = TFLiteCompilerSolution(
      self,
      model=model,
      name=f"TFLITE_{model.name}_{len(self.solutions)}",
      config=config,
      flatbuffer=tflite_model_data
    )

    self.solutions.append(sol)

    return sol

  def create_identifier(self) -> str:

    identifer = f"{super().create_identifier()}-TFLite:{self.analysis.name}"

    hash_object = hashlib.sha256()
    hash_object.update(identifer.encode('utf-8'))
    config_hash = hash_object.hexdigest()

    return config_hash

  '''def dump(self, folder_path : Union[str, pathlib.Path]) -> Dict[str, object]:
    sol_paths = {}
    for sol in self.solutions:
      sol_paths[sol.id] = sol.dump(folder_path)

    data = {
      "rewriter_type" : "TFLiteCompilerInterface",
      "solutions" : sol_paths,
      "identifier" : self.create_identifier()
    }

    with open(folder_path / "tflite_interface.json", "w") as file:
      json.dump(data, file)

    return data'''
  
  def dump(self, folder_path : Union[str, pathlib.Path]) -> Dict[str, object]:
    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)
    
    solutions_path = folder_path / f"compiler_tflite_{self.create_identifier()}"
    solutions_path.mkdir(parents=True, exist_ok=True)

    sols = []
    for sol in self.solutions:
      path: pathlib.Path = solutions_path #/ f"tflite_{sol.name}"
      path = sol.dump(path).relative_to(folder_path)
      sols.append((str(path), sol))

    if self.quantization_data is not None:
      quant_data = (self.quantization_data.name, self.quantization_data.access_id())
    else:
      quant_data = None

    data = {
      "report_type" : "TFLiteCompilerInterface",
      "name" : self.analysis.name,
      "identifier" : self.create_identifier(),
      "solutions" : dict([(str(path), sol.get_config()) for (path, sol) in sols]),
      "quant_data" : quant_data,
    }

    with open(folder_path / f"compiler_tflite_{self.create_identifier()}.json", "w") as file:
      json.dump(data, file)

    return data

  def render_summary(self, folder_path : Union[str, pathlib.Path]) -> Tuple[str, str]:
    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / '..' / 'templates'

    with open(_template_path / "tflite_compiler.html", "r") as file:
      template = Template(file.read())

    summary = self.dump(folder_path=folder_path)

    summary["text"] = "The TFLite compiler interface was used to create these versions of the model:"
    #sol_data = [(sol.get_config(), sol_path) for (sol, sol_path) in zip(self.solutions, summary["solutions"])]
    #summary["solutions"] = sol_data

    html = template.render(summary=summary)
    html_filename = f"compiler_tflite_{self.create_identifier()}.html"
    html_path = folder_path / html_filename
    with open(html_path, "w") as file:
      file.write(html)
    
    return (
            "Compiled:TFLite",
            html_filename,
        )
  
  @classmethod
  def create_pass(
      cls,
      optimizations: List[tf.lite.Optimize] = [tf.lite.Optimize.DEFAULT],
      target_spec: tf.lite.TargetSpec = None,
      quantize: QuantizationMode = QuantizationMode.FLOAT,
      quant_ops: List[tf.lite.OpsSet] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
      quant_data:tf.data.Dataset=None
    ) -> Tuple[str, callable]:
    """
    Creates a closure that wraps the entire pass to create a solution using the TFLiteCompilerInterface.
    This closure can be added to the queue of a ModelAnalysis object and abstracts the internal complexity of the CompilerInterface away, which simplifies its usage.

    Args:
        optimizations (List[tf.lite.Optimize], optional): The TFLite optimization flags. Defaults to [tf.lite.Optimize.DEFAULT].
        target_spec (tf.lite.TargetSpec, optional): An optional TFLite target spec. Defaults to None.
        quantize (QuantizationMode, optional): The intended quantization mode, supports FLOAT, INTERNAL and EXTERNAL. Defaults to QuantizationMode.FLOAT.
        quant_ops (List[tf.lite.OpsSet], optional): The Ops that are supposed to be used, if the model will be quantized, defaults to int8, as it is the only non-experimential that is supported by CMSIS. Defaults to [tf.lite.OpsSet.TFLITE_BUILTINS_INT8].
        quant_data (tf.data.Dataset, optional): The dataset that will be used to guide the post-training quantization. Only needed, if you plan to create quantized solutions. Defaults to None.

    Returns:
        Tuple[str, callable]: _description_
    """
    
    str_id = f"TFLiteCompilerInterface:{str(optimizations)}_{target_spec}_{quantize}_{quant_ops}_{id(quant_data)}"

    def rewrite(analysis : aNN.ModelAnalysis) -> TFLiteCompilerSolution:
      quant_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="quantization", modality=None).from_source(tf_dataset=quant_data)

      tfl_interface = TFLiteCompilerInterface(analysis=analysis, quantization_data=quant_data_report)

      sol = tfl_interface.compile(
        optimizations=optimizations,
        target_spec=target_spec,
        quantize=quantize,
        quant_ops=quant_ops
      )

      return sol

    return str_id, rewrite