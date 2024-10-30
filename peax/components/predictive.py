from tensorflow import keras
import tensorflow as tf
import numpy as np

import logging as log
import enum
from typing import Dict

from . import architecture as arch


class Modality(enum.Enum):
    """enumeration of data modalities that can be recognized"""

    AUDIO = 0
    IMAGE = 1
    VIDEO = 2
    NLP = 3
    SERIES = 4
    OTHER = 5


def recognize_modality(model: tf.keras.models.Model) -> Dict[str, Modality]:
    """Tries to guess the used data modality for each input, can be completely wrong

    Args:
        model (tf.keras.models.Model): The model that will be analyzed

    Returns:
        Dict[str, Modality]: dict, keys are the input names, values are the modalities
    """

    inp_dict = {}
    inputs = model.inputs
    for inp in inputs:
        inp_dict[inp.name] = _recognize_input(input_tensor=inp)
    return inp_dict


def _recognize_input(input_tensor) -> Modality:
    """tries to guess the used data modality of the input tensor based on its shape

    Args:
        input_tensor (_type_): the keras tensor that will be analyzed

    Returns:
        Modality: the modality that might be used for this input
    """
    input_shape = input_tensor.shape[1::]
    if len(input_shape) == 2 and input_shape[1] > 1:
        return Modality.AUDIO
    elif len(input_shape) == 2 and input_shape[1] == 1:
        return Modality.NLP
    elif len(input_shape) == 3 and input_shape[2] == 3:
        return Modality.IMAGE
    elif len(input_shape) == 4 and input_shape[3] == 3:
        return Modality.VIDEO
    elif len(input_shape) == 3 and input_shape[2] == 1:
        return Modality.SERIES
    else:
        return Modality.OTHER


class Task(enum.Enum):
    """enumeration of tasks that can be recognized"""

    BINARY_CLASSIFICATION = 1
    CLASSIFICATION = 2
    REGRESSION = 3
    SEGMENTATION = 4
    UNKNOWN = 5


def _detect_activation(layer: keras.layers.Layer) -> str:
    """Detects the used activation function of a Keras layer,
    necessary, as they can be a parameter in a compute layer, their own layer or a parameter for an Acitvation Layer

    Args:
        layer (keras.layers.Layer): The layer, in which we suspect an activation function

    Returns:
        str: the name of the used activation function, None if no activation has been found
    """
    activation = None
    if isinstance(layer, keras.layers.Softmax):
        activation = "softmax"
    elif isinstance(layer, keras.layers.ReLU):
        activation = "relu"
    elif isinstance(layer, keras.layers.LeakyReLU):
        activation = "lrelu"
    elif isinstance(layer, keras.layers.PReLU):
        activation = "prelu"
    elif isinstance(layer, keras.layers.ELU):
        activation = "elu"
    elif isinstance(layer, keras.layers.ThresholdedReLU):
        activation = "threshold_relu"

    if activation != None:
        return activation

    log.info(f"layer {layer} is not a preconfigured activation layer")

    try:
        activation = layer.activation.__name__
    except Exception as e:
        log.warning(f"no activation found in layer {layer}")

    return activation


def recognize_task(model: tf.keras.Model) -> Dict[str, Task]:
    """Tries to recognize the task that the model is trying to solve.
    Can detect the Tasks that are described in the Task Enum.
    (i.e. Classification, Regression, Segmentation, etc)

    Args:
        model (tf.keras.Model): The Model that will be analyzed

    Returns:
        Dict[str, Task]: A dict that uses the output names as keys and their task as values
    """

    outputs = arch.identify_output_layers(model=model)

    task = {}
    for name, layer in outputs.items():
        task[name] = Task.UNKNOWN

        activation = _detect_activation(layer)
        if len(layer.output_shape) <= 2:
            if activation is None:
                task[name] = Task.UNKNOWN
                continue

            if activation in ["softmax"]:
                task[name] = Task.CLASSIFICATION
            elif activation in ["sigmoid"]:
                task[name] = Task.BINARY_CLASSIFICATION
            elif isinstance(layer, keras.layers.Dense) and layer.units > 1:
                task[name] = Task.CLASSIFICATION
            elif (
                activation in ["softplus", "linear", None]
                or "elu" in activation
            ):
                task[name] = Task.REGRESSION
            else:
                task[name] = Task.UNKNOWN
        elif len(layer.output_shape) > 2 and activation in [
            "softmax",
            "sigmoid",
        ]:
            task[name] = Task.SEGMENTATION
        else:
            task[name] = Task.UNKNOWN

    return task
