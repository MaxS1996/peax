import enum
from pdb import run
from typing import Dict, List, Union, Tuple
import pathlib
import json
import hashlib

import shutil

import copy
import json
import os
import uuid

from jinja2 import Template

import tensorflow as tf

import tvm
from tvm.driver import tvmc

from . import rewriter as compiler
from . import tflite as tflc

from peax.reports import dataset as d
import peax.analysis as aNN

class MicroTVMSolution(compiler.CompilerSolution):
  """
  A Solution created by the microTVM CompilerInterface
  """

  def __init__(self, rewriter: compiler.CompilerInterface, model:tf.keras.models.Model, name:str, config:Dict[str, object], package, tflite_sol : tflc.TFLiteCompilerSolution):
    """
    The constructor for the MicroTVMSolution class, should only be called by the MicroTVMCompilerInterface

    Args:
        rewriter (compiler.CompilerInterface): The CompilerInterface that created the solution object.
        model (tf.keras.models.Model): The original of the model that has been compiled.
        name (str): A name for the created solution
        config (Dict[str, object]): The compiler parameters that have been used to create the solution.
        package (_type_): The output of the TVM compiler.
        tflite_sol (tflc.TFLiteCompilerSolution): The TFLiteCompilerInterface which is used to initially optimize the model graph and optionally to quantize the model.
    """
    super().__init__(rewriter=rewriter, model=model, name=name, config=config)
    self.package = package
    self.id = self.get_identifier()

    self.package = package
    self.tflite_solution = tflite_sol

  def get_identifier(self) -> str:
    """
    Returns the unique identifier of the solution.

    Returns:
        str: The unique identifier of the solution.
    """
    config = {
      "rewriter" : self.rewriter.create_identifier(),
      "config": dict([(key, str(x)) for key, x in self.config.items()]),
      "runtime" : self.rewriter.runtime.name,
      "executor" :self.rewriter.executor.name,
    }

    sorted_config_str = json.dumps(config, sort_keys=True)
    hash_object = hashlib.sha256()
    hash_object.update(sorted_config_str.encode('utf-8'))
    config_hash = hash_object.hexdigest()

    return config_hash
  
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
  
  def dump(self, folder_path : Union[str, pathlib.Path]) -> pathlib.Path:
    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)
    
    config_hash = self.id
    rel_path = f"compiled_{self.name}_{config_hash}"
    
    folder_path = folder_path / rel_path
    folder_path.mkdir(exist_ok=True, parents=True)

    #TODO: dump config json
    with open(folder_path / "config.json", "w") as file:
      json.dump(self.config, file)

    #TODO: create target folder, dump everything in there, copy module.tar there
    src_path = self.package.package_path
    dst_path = folder_path / "module.tar"
    shutil.copy(src_path, dst_path)

    return folder_path
  

class MicroTVMCompilerInterface(compiler.CompilerInterface):
  """
  The CompilerInterface to convert a ModelAnalysis object using microTVM as a compiler.
  """

  solutions : List[MicroTVMSolution]
  """
  The solutions created by the CompilerInterface.
  """

  quantization_data : d.DatasetReport
  """
  Dataset that will be used for the post-training quanitzation step.
  """

  quantize : bool
  """
  Bool, is True, if the created solutions will use integer quantization.
  """

  target : str
  """
  A string that describes the target device and configuration, default value creates usable results on Cortex-M-based platforms.
  """

  runtime : str
  """
  The microTVM runtime that should be used during the execution of the compiled network.
  The default value is 'c', which describes the C-based runtime, which is preferred for embedded applications.
  """

  executor : str
  """
  The TVM executor that is supposed to plan/control the execution of the inference.
  The default value is 'aot', which references the Ahead-of-Time executor that does not require an at-runtime interpreter and performs all possible operations,
  like memory planning and function calling for the model at compile time instead.
  """

  tflite_interface : tflc.TFLiteCompilerInterface
  """
  The current implementation relies on the TFLite converter to optimize and quantize the models to 8-bit integer formats.
  To facilitate this, the microTVM CompilerInterface wraps the TFLite CompilerInterface, which will also be added to the ModelAnalysis Rewriter registry.
  """

  def __init__(self, analysis : aNN.ModelAnalysis, target:str="cmsis-nn,c", runtime:str="crt", executor:str="aot", quantization_data:d.DatasetReport=None):
    """
    The constructor for the MicroTVMCompilerInterface class.

    Args:
        analysis (aNN.ModelAnalysis): The ModelAnalysis to which the CompilerInterface object will be submitted
        target (str, optional): A target string that describes the target configuration for TVM's flow. The default is optimal for Cortex-M-based platforms. Defaults to "cmsis-nn,c".
        runtime (str, optional): The TVM runtime that is supposed to be used. The default is the best option for embedded applications. Defaults to "crt".
        executor (str, optional): The TVM executor that will control the inference.
        The ahead-of-time has the minimizes the at-runtime overhead, by resolving problems like scheduling and memory planning at compile time. Defaults to "aot".
        quantization_data (d.DatasetReport, optional): The dataset object that will be used to guide the post-training quantization. Only needed, if you plan to create quantized solutions. Defaults to None.
    """
    

    self.quantize = False
    self.quantization_data = quantization_data
    self.solutions = []

    self.target = target
    self.runtime = tvm.relay.backend.Runtime(name=runtime, options={"system-lib": True})
    self.executor = tvm.relay.backend.Executor(name=executor, options={"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8})

    super().__init__(analysis=analysis)

    if quantization_data is not None:
      self.quantize = True

    self.tflite_interface = tflc.TFLiteCompilerInterface(analysis=analysis, quantization_data=quantization_data)

  def compile(self, mcpu:str="cortex-m55", ethos_u:bool=False, ethos_u_config:str="ethos-u55-128") -> MicroTVMSolution:
    """
    The compile command, which creates the microTVM solution.

    Args:
        mcpu (str, optional): The string that describes the target CPU, follows the llvm -mcpu attribute style. Defaults to "cortex-m55".
        ethos_u (bool, optional): If True, Ethos-U will be added as target and subgraphs will be mapped to it. Defaults to False.
        ethos_u_config (str, optional): The hardware configuration of the Ethos-U, if it is used. Defaults to "ethos-u55-128".

    Returns:
        MicroTVMSolution: The solution that has been created by compiling the model with microTVM.
    """
    model:tf.keras.models.Model = self.analysis.keras_model

    target = self.target

    add_targets :dict[str, dict[str, str]] = {}

    pass_context_configs = [
    "tir.disable_vectorize=1",
    ]

    if ethos_u:
      add_targets["ethos-u"] = {
        'accelerator_config': ethos_u_config
      }
      target = "ethos-u,"+target

      pass_context_configs.append("tir.usmp.enable=1")
      pass_context_configs.append("tir.usmp.algorithm=hill_climb")
      pass_context_configs.append("tir.disable_storage_rewrite=1")

    add_targets["c"] = {
      "mcpu" : mcpu,
    }

    quant_mode = tflc.QuantizationMode.FLOAT
    if self.quantize:
      quant_mode = tflc.QuantizationMode.EXTERNAL
    tfl_sol = self.tflite_interface.compile(quantize=quant_mode)

    model_path = tfl_sol.dump(self.analysis.cache_dir / f"{str(uuid.uuid4())}") / f"{model.name}.tflite"
    output_path :pathlib.Path = self.analysis.cache_dir / "mTVM" / f"{str(uuid.uuid4())} / module.tar"
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / "module.tar"

    model = tvmc.load(model_path)

    output = tvmc.compile(
      tvmc_model=model,
      target=target,
      additional_target_options=add_targets,
      runtime=self.runtime,
      executor=self.executor,
      output_format="mlf",
      pass_context_configs=pass_context_configs,
      package_path=output_path,
    )

    config = {
      "runtime" : self.runtime.name,
      "executor" : self.executor.name,
      "target" : target,
      "mcpu" : mcpu,
      "ethos_u" : ethos_u,
    }

    # TODO: delete the temporary TFLiteSolution dump
    for tmp_file_path in model_path.parent.iterdir():
      tmp_file_path.unlink(missing_ok=True)
    
    model_path.parent.rmdir()
    model_path.parent.parent.rmdir()

    if ethos_u:
      config["ethos_u_config"] = ethos_u_config

    microtvm_sol = MicroTVMSolution(
      rewriter=self,
      model=model,
      name=f"MicroTVM_{len(self.solutions)}_{self.analysis.keras_model.name}_{target.replace(',', '')}",
      config=config,
      package=output,
      tflite_sol=tfl_sol)

    self.solutions.append(microtvm_sol)

    return microtvm_sol

  def create_identifier(self)-> str:
    """
    Returns the identifier of the microTVM CompilerInterface.

    Returns:
        str: the identifier of the CompilerInterface object.
    """
    
    identifer = f"{super().create_identifier()}-microTVM:{self.analysis.name}-{self.runtime.name}-{self.executor.name}-{self.target}"

    hash_object = hashlib.sha256()
    hash_object.update(identifer.encode('utf-8'))
    config_hash = hash_object.hexdigest()

    return config_hash
  
  def dump(self, folder_path: Union[str,pathlib.Path]) -> Dict[str, object]:
    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    solutions_path = folder_path / f"compiler_microTVM_{self.create_identifier()}"
    solutions_path.mkdir(parents=True, exist_ok=True)

    sols = []
    for sol in self.solutions:
      path: pathlib.Path = solutions_path #/ f"tflite_{sol.name}"
      path = sol.dump(path).relative_to(folder_path)
      sols.append((str(path), sol))

    if self.quantization_data is not None:
      quant_data = (self.quantization_data.name, self.quantization_data.create_unique_id())
    else:
      quant_data = None

    data = {
      "report_type" : "MicroTVMCompilerInterface",
      "name" : self.analysis.name,
      "runtime" : self.runtime.name,
      "executor" : self.executor.name,
      "identifier" : self.create_identifier(),
      "solutions" : dict([(str(path), sol.get_config()) for (path, sol) in sols]),
      "quant_data" : quant_data,
    }

    with open(folder_path / f"compiler_microTVM_{self.create_identifier()}.json", "w") as file:
      json.dump(data, file)

    return data
  
  def render_summary(self, folder_path: Union[str,pathlib.Path]) -> Tuple[str, str]:
    if isinstance(folder_path, str):
      folder_path = pathlib.Path(folder_path)

    _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / '..' / 'templates'

    with open(_template_path / "microTVM_compiler.html", "r") as file:
      template = Template(file.read())

    summary = self.dump(folder_path=folder_path)

    summary["text"] = "The microTVM compiler interface was used to create these versions of the model:"
    #sol_data = [(sol.get_config(), sol_path) for (sol, sol_path) in zip(self.solutions, summary["solutions"])]
    #summary["solutions"] = sol_data

    html = template.render(summary=summary)
    html_filename = f"compiler_microTVM_{self.create_identifier()}.html"
    html_path = folder_path / html_filename
    with open(html_path, "w") as file:
      file.write(html)
    
    return (
            "Compiled:microTVM",
            html_filename,
        )
  
  @classmethod
  def create_pass(cls, target:str="cmsis-nn,c", runtime:str="crt", executor:str="aot", mcpu:str="cortex-m55", ethos_u:bool=False, ethos_u_config:str="ethos-u55-128", quant_data:tf.data.Dataset=None) -> Tuple[str, callable]:
    """
    Creates a closure that wraps the entire pass to create a solution using the microTVMCompilerInterface.
    This closure can be added to the queue of a ModelAnalysis object and abstracts the internal complexity of the interface away, to simplify the usage.

    Args:
        target (str, optional): A target string that describes the target configuration for TVM's flow. The default is optimal for Cortex-M-based platforms. Defaults to "cmsis-nn,c".
        runtime (str, optional): The TVM runtime that is supposed to be used. The default is the best option for embedded applications. Defaults to "crt".
        executor (str, optional): The TVM executor that will control the inference.
        The ahead-of-time has the minimizes the at-runtime overhead, by resolving problems like scheduling and memory planning at compile time. Defaults to "aot".
        mcpu (str, optional): The string that describes the target CPU, follows the llvm -mcpu attribute style. Defaults to "cortex-m55".
        ethos_u (bool, optional): If True, Ethos-U will be added as target and subgraphs will be mapped to it. Defaults to False.
        ethos_u_config (str, optional): The hardware configuration of the Ethos-U, if it is used. Defaults to "ethos-u55-128".
        quant_data (tf.data.Dataset, optional): The dataset object that will be used to guide the post-training quantization. Only needed, if you plan to create quantized solutions. Defaults to None.. Defaults to None.

    Returns:
        Tuple[str, callable]: _description_
    """
    
    str_id = f"microTVMCompilerInterface:{target}_{runtime}_{executor}_{mcpu}_{ethos_u}_{ethos_u_config}_{id(quant_data)}"

    def rewrite(analysis : aNN.ModelAnalysis) -> MicroTVMSolution:
      quant_data_report = d.DatasetReport.submit_to(analysis, lazy=False).with_config(name="quantization", modality=None).from_source(tf_dataset=quant_data)

      microTVM_interface = MicroTVMCompilerInterface(
        analysis=analysis,
        target=target,
        runtime=runtime,
        executor=executor,
        quantization_data=quant_data_report
        )
      
      sol = microTVM_interface.compile(mcpu=mcpu, ethos_u=ethos_u, ethos_u_config=ethos_u_config)

      return sol

    return str_id, rewrite