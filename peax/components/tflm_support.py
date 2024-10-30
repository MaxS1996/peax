from tensorflow import keras
import tensorflow as tf
import numpy as np

import logging as log
from typing import List, Tuple
import requests
import re

from . import architecture as arch

_known_tflm_branches = ["main"]
"""known branch names that can be looked up directly in the repository"""

_translate_table = {
    "Abs": keras.layers.ReLU,
    "Add": keras.layers.Add,
    "AddN": keras.layers.Add,
    "AveragePool2D": keras.layers.AveragePooling2D,
    "BatchToSpaceNd": keras.layers.Reshape,
    "Concatenation": keras.layers.Concatenate,
    "Conv2D": keras.layers.Conv2D,
    "DepthwiseConv2D": keras.layers.DepthwiseConv2D,
    "Elu": keras.layers.ELU,
    "FullyConnected": keras.layers.Dense,
    "L2Pool2D": keras.layers.MaxPooling2D,
    "LeakyRelu": keras.layers.LeakyReLU,
    "Logistic": keras.layers.Activation,
}
"""helper to translate the downloaded layer names to Keras layers classes"""


def _request_tflm_resolver(branch: str = "main") -> str:
    """Downloads the relevant source code from the official TFLM repository.
    It returns the raw source code, which can then be used for further analysis

    Args:
        branch (str, optional): the branch or commit from which the source should be taken. Defaults to "main".

    Returns:
        str: the raw source code
    """
    uri = f"https://raw.githubusercontent.com/tensorflow/tflite-micro/{branch}/tensorflow/lite/micro/all_ops_resolver.cc"
    r = requests.get(uri)
    source_code = ""

    if r.status_code == 200:
        source_code = r.content.decode()
    else:
        log.error(
            f"Failed to retrieve ops support from TFLite Github repository: {uri}"
        )

    return source_code


def _analyze_tflm_resolver(source: str) -> List[str]:
    """Extracts the supported layer types from the source code

    Args:
        source (str): the previously acquired source code

    Returns:
        List[str]: A list of the found function/layer names
    """
    functions = re.findall(r"Add(\w+)\(\)", source)
    return functions


def _get_tflm_layers(branch: str = "main") -> List[keras.layers.Layer]:
    """Returns the Keras layers that are supported by TFLM based on the downloaded AllOpsResolver

    Args:
        branch (str, optional): The TFLM GitHub branch or commit that will be used as reference. Defaults to "main".

    Returns:
        List[keras.layers.Layer]: List of supported Keras layers
    """
    '''if not branch in _known_tflm_branches:
        log.warning(
            f"{branch} is not a known branch of the official TFLite-micro repository, it might not exit"
        )

    source = _request_tflm_resolver(branch=branch)
    functions = _analyze_tflm_resolver(source=source)

    supported_layers = [keras.layers.InputLayer]

    for layer_name in functions:
        if layer_name in _translate_table:
            supported_layers.append(_translate_table[layer_name])'''

    supported_layers = list(_translate_table.values())
    supported_layers.append(keras.layers.InputLayer)
    return supported_layers


def check_model_support(
    model: tf.keras.Model, tflm_branch: str = "main"
) -> Tuple[List[keras.layers.Layer], List[keras.layers.Layer]]:
    """Checks, if the layers in the given model are supported by the TFLM branch.
    Only checks for matches in layer classes, no analysis of hyperparameters takes place!

    Args:
        model (tf.keras.Model): The model that will be analyzed
        tflm_branch (str, optional): The reference branch from the official TFLM Github repository. Defaults to "main".

    Returns:
        Tuple[List[keras.layers.Layer], List[keras.layers.Layer]]: List of the model layers that are supported and a second list containing all unsupported layers
    """
    tflm_layers = _get_tflm_layers(branch=tflm_branch)

    supported = []
    unsupported = []
    for layer in model.layers:
        if type(layer) in tflm_layers:
            supported.append(layer)
        else:
            unsupported.append(layer)
    return supported, unsupported
