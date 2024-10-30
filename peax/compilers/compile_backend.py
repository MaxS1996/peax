from typing import Dict
import pathlib
from abc import ABC, abstractmethod

import tensorflow as tf

class Backend(ABC):

  @abstractmethod
  def compile(self, model : tf.keras.models.Model):
    pass

  @abstractmethod
  def quantize(self, model : tf.keras.models.Model):
    pass

  @abstractmethod
  def optimize(self, model : tf.keras.models.Model):
    pass

  @abstractmethod
  def store(self, compiled_model : object, path : pathlib.Path):
    pass

  @abstractmethod
  def invoke(self, model : tf.keras.models.Model):
    """Wraps the entire optimization and conversion process into one function

    Args:
        model (tf.keras.models.Model): The (sub)model that needs to be compiled
    """
    pass


