import tensorflow as tf
import numpy as np

import pytest
import logging

import os
import sys

from peax.components import architecture as arch
import peax.analysis as aNN

from peax.components import predictive as prd
from peax.components import tflm_support as tflm


class TestExtractEdges:
    def test_empty_routes(self):
        routes = {}
        assert arch.extract_edges(routes) == []

    def test_single_route(self):
        input_layer = tf.keras.layers.Input(
            shape=(28, 28, 1), name="input_layer"
        )
        conv_layer = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            name="conv_layer",
        )(input_layer)
        flatten_layer = tf.keras.layers.Flatten(name="flatten_layer")(
            conv_layer
        )
        output_layer = tf.keras.layers.Dense(
            units=10, activation="softmax", name="output_layer"
        )(flatten_layer)

        routes = {
            "output": [input_layer, conv_layer, flatten_layer, output_layer]
        }
        expected_edges = [
            ("input_layer", "conv_layer/Relu:0"),
            ("conv_layer/Relu:0", "flatten_layer/Reshape:0"),
            ("flatten_layer/Reshape:0", "output_layer/Softmax:0"),
        ]
        assert arch.extract_edges(routes) == expected_edges

    def test_multiple_routes(self):
        input_layer_1 = tf.keras.layers.Input(
            shape=(28, 28, 1), name="input_layer_1"
        )
        conv_layer_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            name="conv_layer_1",
        )(input_layer_1)
        flatten_layer_1 = tf.keras.layers.Flatten(name="flatten_layer_1")(
            conv_layer_1
        )
        output_layer_1 = tf.keras.layers.Dense(
            units=10, activation="softmax", name="output_layer_1"
        )(flatten_layer_1)

        input_layer_2 = tf.keras.layers.Input(
            shape=(32, 32, 3), name="input_layer_2"
        )
        conv_layer_2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation="relu",
            name="conv_layer_2",
        )(input_layer_2)
        flatten_layer_2 = tf.keras.layers.Flatten(name="flatten_layer_2")(
            conv_layer_2
        )
        output_layer_2 = tf.keras.layers.Dense(
            units=5, activation="softmax", name="output_layer_2"
        )(flatten_layer_2)

        routes = {
            "output_1": [
                input_layer_1,
                conv_layer_1,
                flatten_layer_1,
                output_layer_1,
            ],
            "output_2": [
                input_layer_2,
                conv_layer_2,
                flatten_layer_2,
                output_layer_2,
            ],
        }
        expected_edges = [
            ("input_layer_1", "conv_layer_1/Relu:0"),
            ("conv_layer_1/Relu:0", "flatten_layer_1/Reshape:0"),
            ("flatten_layer_1/Reshape:0", "output_layer_1/Softmax:0"),
            ("input_layer_2", "conv_layer_2/Relu:0"),
            ("conv_layer_2/Relu:0", "flatten_layer_2/Reshape:0"),
            ("flatten_layer_2/Reshape:0", "output_layer_2/Softmax:0"),
        ]

        assert arch.extract_edges(routes) == expected_edges


class TestGetPredecessor:
    def test_predecessor_is_none(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8, input_shape=(16,)))

        pred = arch.get_predecessor(model.input)
        assert pred is None

    def test_predecessor_is_correct(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(10,)))
        model.add(tf.keras.layers.Dense(32))
        model.add(tf.keras.layers.Dense(64))

        pred = arch.get_predecessor(model.layers[-1])
        assert pred == model.layers[-2]

    @pytest.mark.usefixtures("caplog")
    def test_warning_is_logged(self, caplog):
        # TODO: fix test
        input_layer = tf.keras.Input(shape=(10,))
        layer1 = tf.keras.layers.Dense(32)(input_layer)
        layer2 = tf.keras.layers.Dense(64)(layer1)
        layer3 = tf.keras.layers.Dense(128)(layer2)

        model = tf.keras.Model(inputs=input_layer, outputs=layer3)
        pred = arch.get_predecessor(model.layers[-1])

        #print(caplog.records)

        assert pred == model.layers[-2]
        # assert "more than one inbound node detected" in caplog.text


class TestIdentifyOutputLayers:
    @pytest.fixture
    def single_output_model(self):
        inputs = tf.keras.layers.Input(shape=(10,))
        x = tf.keras.layers.Dense(64, activation="relu")(inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation=None, use_bias=False)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @pytest.fixture
    def multi_output_model(self):
        inputs = tf.keras.layers.Input(shape=(10,))
        x = tf.keras.layers.Dense(64, activation="relu")(inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        output1 = tf.keras.layers.Dense(
            1, activation="sigmoid", name="output1"
        )(x)
        output2 = tf.keras.layers.Dense(
            1, activation="sigmoid", name="output2"
        )(x)
        return tf.keras.Model(inputs=inputs, outputs=[output1, output2])

    def test_identify_single_output_layer(self, single_output_model):
        expected_output_layer_names = [
            single_output_model.outputs[0].name.split("/")[0]
        ]
        output_layers = arch.identify_output_layers(single_output_model)

        assert isinstance(output_layers, dict)
        assert len(output_layers) == len(expected_output_layer_names)

        for layer_name in expected_output_layer_names:
            assert layer_name in output_layers.keys()
            assert isinstance(output_layers[layer_name], tf.keras.layers.Layer)

    def test_identify_output_layers_multi_output(self, multi_output_model):
        expected_output_layer_names = ["output1", "output2"]
        output_layers = arch.identify_output_layers(multi_output_model)

        assert isinstance(output_layers, dict)
        assert len(output_layers) == len(expected_output_layer_names)

        for layer_name in expected_output_layer_names:
            assert layer_name in output_layers
            assert isinstance(output_layers[layer_name], tf.keras.layers.Layer)


class TestIdentifyRoutes:
    def test_identify_routes_reverse(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
        model.add(tf.keras.layers.Dense(4))

        routes = arch.identify_routes(model)
        assert list(routes.keys())[0] == model.layers[-1].name
        assert routes[model.layers[-1].name][0] == model.layers[1]
        assert routes[model.layers[-1].name][1] == model.layers[0]

    def test_identify_routes_forward(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
        model.add(tf.keras.layers.Dense(4))

        # akwardly written because of the Input Layer that is added by the Sequential API
        routes = arch.identify_routes(model, reverse=False)
        assert list(routes.keys())[0] == model.layers[-1].name
        assert routes[model.layers[-1].name][-1] == model.layers[-1]
        assert routes[model.layers[-1].name][-2] == model.layers[-2]
