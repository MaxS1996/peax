import tensorflow as tf
import numpy as np

import pytest

from peax.components import architecture as arch
import peax.analysis as aNN

from peax.components import predictive as prd
from peax.components import resource as res
from peax.components import tflm_support as tflm_sup


class TestCheckModelSupport:
    def test_check_model_support_mixed_model(self):
        # Create a model with supported and unsupported layers
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation="softmax"),
                tf.keras.layers.BatchNormalization(),
            ]
        )

        # Test with default TFLM branch
        supported, unsupported = tflm_sup.check_model_support(model)

        assert len(supported) == 3
        assert len(unsupported) == 3

    '''def test_check_model_support_wrong_ref(self):
        # Create a model with supported and unsupported layers
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation="softmax"),
                tf.keras.layers.BatchNormalization(),
            ]
        )

        # Test with non-existing TFLM branch
        supported, unsupported = tflm_sup.check_model_support(
            model, "non-existing"
        )
        assert len(supported) == 0
        assert len(unsupported) == 6'''

    def test_check_model_support_full_support(self):
        # Create a model with only supported layers
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
            ]
        )

        # Test with default TFLM branch
        supported, unsupported = tflm_sup.check_model_support(model)
        assert len(supported) == 2
        assert len(unsupported) == 0
