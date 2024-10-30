import tensorflow as tf
import numpy as np

import pytest

from peax.components import architecture as arch
import peax.analysis as aNN

from peax.components import predictive as prd
from peax.components import tflm_support as tflm


class TestActivationDetection:
    def test_softmax_activation(self):
        layer = tf.keras.layers.Softmax()
        result = prd._detect_activation(layer)
        assert result == "softmax"

    def test_relu_activation(self):
        layer = tf.keras.layers.ReLU()
        result = prd._detect_activation(layer)
        assert result == "relu"

    def test_lrelu_activation(self):
        layer = tf.keras.layers.LeakyReLU()
        result = prd._detect_activation(layer)
        assert result == "lrelu"

    def test_prelu_activation(self):
        layer = tf.keras.layers.PReLU()
        result = prd._detect_activation(layer)
        assert result == "prelu"

    def test_elu_activation(self):
        layer = tf.keras.layers.ELU()
        result = prd._detect_activation(layer)
        assert result == "elu"

    def test_elu_activation(self):
        layer = tf.keras.layers.ThresholdedReLU()
        result = prd._detect_activation(layer)
        assert result == "threshold_relu"

    def test_layer_with_tanh_activation(self):
        layer = tf.keras.layers.Dense(10, activation=tf.nn.tanh)
        result = prd._detect_activation(layer)
        assert result == "tanh"

    def test_layer_with_linear_activation(self):
        layer = tf.keras.layers.Dense(10)
        result = prd._detect_activation(layer)
        assert result == "linear"

    def test_layer_with_unknown_activation(self):
        layer = tf.keras.layers.Flatten()
        assert prd._detect_activation(layer) == None


class TestRecognitionTask:
    def test_classification(self):
        input_layer = tf.keras.layers.Input(shape=(10,))
        x = tf.keras.layers.Dense(32, activation="relu")(input_layer)
        output_layer = tf.keras.layers.Dense(5, activation="softmax")(x)
        model = tf.keras.Model(input_layer, output_layer)
        assert prd.recognize_task(model) == {
            model.layers[-1].name: prd.Task.CLASSIFICATION
        }

    def test_binary_classification(self):
        input_layer = tf.keras.layers.Input(shape=(10,))
        x = tf.keras.layers.Dense(32, activation="relu")(input_layer)
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(input_layer, output_layer)
        assert prd.recognize_task(model) == {
            model.layers[-1].name: prd.Task.BINARY_CLASSIFICATION
        }

    def test_regression(self):
        input_layer = tf.keras.layers.Input(shape=(10,))
        x = tf.keras.layers.Dense(32, activation="relu")(input_layer)
        output_layer = tf.keras.layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(input_layer, output_layer)
        assert prd.recognize_task(model) == {
            model.layers[-1].name: prd.Task.REGRESSION
        }

    def test_segmentation(self):
        input_layer = tf.keras.layers.Input(shape=(255, 255, 3))
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, activation="relu"
        )(input_layer)
        output_layer = tf.keras.layers.Conv2D(
            filters=5, kernel_size=3, activation="softmax"
        )(x)
        model = tf.keras.Model(input_layer, output_layer)
        assert prd.recognize_task(model) == {
            model.layers[-1].name: prd.Task.SEGMENTATION
        }

    def test_unknown_task(self):
        input_shape = (32, 32, 3)
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Dense(10, activation="tanh")(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        task = prd.recognize_task(model)
        assert task == {model.layers[-1].name: prd.Task.UNKNOWN}

    def test_unknown_layer(self):
        input_shape = (32, 32, 3)
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        task = prd.recognize_task(model)

        #print(task)
        assert task == {model.layers[-1].name: prd.Task.UNKNOWN}
