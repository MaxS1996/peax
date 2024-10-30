import tensorflow as tf
from tensorflow import keras
import numpy as np

import pytest

from peax.components import resource as res
from peax.hardware.processor import Processor


class TestProcessor:
    @pytest.fixture
    def processor(self):
        return Processor(
            name="Test Processor", compute_macs=10**9, memory_size=10**9
        )

    def test_init(self, processor):
        assert processor.name == "Test Processor"
        assert processor._MACs == 10**9
        assert processor._MEM == 10**9
        assert processor._unsupported_layers == []
        assert processor._ruleset == {}

    def test_check_supported_layer(self, processor):
        model = keras.Sequential(
            [keras.layers.Dense(units=64, activation="relu")]
        )

        # needs to be done to have weights
        x = tf.ones((1, 4))
        y = model(x)

        supported, delay, weight_alloc, macs = processor.check(layer=model.layers[0])
        assert supported == True
        assert delay == 2.56e-7
        assert weight_alloc["float32"] == 1.28e-06
        assert macs == 256

    def test_check_unsupported_layer_false(self, processor):
        layer = keras.layers.Conv1D(
            input_shape=(50,50,3), filters=32, kernel_size=3, activation="relu"
        )

        model = keras.Sequential(
            [layer]
        )

        processor._unsupported_layers.append(keras.layers.Conv1D)
        # needs to be done to have weights
        x = tf.ones((1, 50, 50, 3))
        y = model(x)

        supported, delay, weight_alloc, layer_cost = processor.check(layer=model.layers[0])
        assert supported == False
        assert delay == float("inf")
        assert weight_alloc["float32"] == float("inf")

    def test_check_unsupported_layer_true(self, processor):
        layer = keras.layers.Conv1D(
            input_shape=(50,50,3), filters=32, kernel_size=3, activation="relu"
        )

        model = keras.Sequential(
            [layer]
        )

        processor._unsupported_layers.append(keras.layers.Conv2D)
        # needs to be done to have weights
        x = tf.ones((1, 50, 50, 3))
        y = layer(x)

        supported, delay, weight_alloc, layer_cost = processor.check(layer=model.layers[0])
        assert supported == True

    def test_check_custom_rule_false(self, processor):
        layer = keras.layers.Dense(units=128, activation="tanh")
        # needs to be done to have weights
        x = tf.ones((1, 4))
        y = layer(x)

        def custom_rule(test_layer):
            return test_layer.units < 100

        processor._ruleset[type(layer)] = custom_rule
        supported, delay, weight_alloc, layer_cost = processor.check(layer=layer)
        assert supported == False
        assert delay == float("inf")
        assert weight_alloc["float32"] == float("inf")

    def test_check_custom_rule_true(self, processor):
        layer = keras.layers.Dense(units=64, activation="tanh")
        # needs to be done to have weights
        x = tf.ones((1, 4))
        y = layer(x)

        def custom_rule(test_layer):
            return test_layer.units < 100

        processor._ruleset[type(layer)] = custom_rule
        supported, delay, weight_alloc, macs = processor.check(layer=layer)
        assert supported == True

    def test_str(self, processor):
        assert (
            str(processor)
            == "Processor Test Processor with 1000000000 MAC/s and 1000000000 bytes memory"
        )
