import tensorflow as tf
import numpy as np

import pytest

from peax.components import architecture as arch
import peax.analysis as aNN

from peax.components import predictive as prd
from peax.components import resource as res


class TestMAC:
    class TestDefaultEstimators:
        def test__mac_stub(self):
            layer = tf.keras.layers.InputLayer()
            assert 0 == res._mac_stub(layer)

        def test__mac_max_pool2d(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.MaxPool2D(pool_size=2, input_shape=(10, 10, 3))
            )
            assert 300 == res._mac_max_pool2d(model.layers[-1])

        def test__mac_dense(self):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(units=2, input_shape=(10,)))
            assert 20 == res._mac_dense(model.layers[-1])

        def test__mac_dw_conv2d(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, input_shape=(10, 10, 3)
                )
            )
            assert 1728 == res._mac_dw_conv2d(model.layers[-1])

        def test__mac_conv2d(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, input_shape=(10, 10, 3)
                )
            )
            assert 13824 == res._mac_conv2d(model.layers[-1])

        def test__mac_conv3d(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Conv3D(
                    filters=8, kernel_size=3, input_shape=(10, 10, 10, 3)
                )
            )
            assert 41472 == res._mac_conv3d(model.layers[-1])

    class TestLayerEstimation:
        def test_get_layer_macs_stub_default(self):
            layer = tf.keras.layers.InputLayer()
            assert 0 == res.get_layer_macs(layer)

        def test_get_layer_macs_max_pool2d_default(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.MaxPool2D(pool_size=2, input_shape=(10, 10, 3))
            )
            assert 300 == res.get_layer_macs(model.layers[-1])

        def test_get_layer_macs_dense_default(self):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(units=2, input_shape=(10,)))
            assert 20 == res.get_layer_macs(model.layers[-1])

        def test_get_layer_macs_dw_conv2d_default(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, input_shape=(10, 10, 3)
                )
            )
            assert 1728 == res.get_layer_macs(model.layers[-1])

        def test_get_layer_macs_conv2d_default(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, input_shape=(10, 10, 3)
                )
            )
            assert 13824 == res.get_layer_macs(model.layers[-1])

        def test_get_layer_macs_conv3d_default(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Conv3D(
                    filters=8, kernel_size=3, input_shape=(10, 10, 10, 3)
                )
            )
            assert 41472 == res.get_layer_macs(model.layers[-1])

        def test_get_layer_macs_custom_estimator(self):
            estimation_functions = {tf.keras.layers.Dense: lambda layer: 64}

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(units=2, input_shape=(10,)))

            assert (
                res.get_layer_macs(model.layers[-1], estimation_functions)
                == 64
            )

    class TestModelEstimation:
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, input_shape=(50, 50, 3)
            )
        )
        model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3))
        model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3))
        model.add(tf.keras.layers.MaxPool2D(pool_size=4))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=50))
        model.add(tf.keras.layers.Dense(units=10))

        model_eval = res.get_model_macs(model)
        assert len(model.layers) == len(model_eval)

        for name, macs in model_eval.items():
            truth_macs = res.get_layer_macs(model.get_layer(name))
            assert macs == truth_macs


class TestStorage:
    class TestLayerWeights:
        def test_get_layer_weight_size_no_weights(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.MaxPool2D(pool_size=4, input_shape=(10, 10, 3))
            )

            assert (
                0
                == res.get_layer_weight_size(model.layers[-1], ["float"])[
                    "float"
                ]
            )

        def test_get_layer_weight_size_dense(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Dense(
                    units=4, input_shape=(10,), use_bias=False
                )
            )

            assert (
                160
                == res.get_layer_weight_size(model.layers[-1], ["float"])[
                    "float"
                ]
            )

        def test_get_layer_weight_size_dense_bias(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Dense(
                    units=4, input_shape=(10,), use_bias=True
                )
            )

            assert (
                176
                == res.get_layer_weight_size(model.layers[-1], ["float"])[
                    "float"
                ]
            )

        def test_get_layer_weight_size_conv2d(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    input_shape=(10, 10, 3),
                    use_bias=False,
                )
            )

            assert (
                3456
                == res.get_layer_weight_size(model.layers[-1], ["float"])[
                    "float"
                ]
            )

        def test_get_layer_weight_size_conv2d_bias(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    input_shape=(10, 10, 3),
                    use_bias=True,
                )
            )

            assert (
                3456 + 128
                == res.get_layer_weight_size(model.layers[-1], ["float"])[
                    "float"
                ]
            )

    class TestModelWeights:
        def test_get_model_weight_size(self):
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Dense(
                    units=50, input_shape=(100,), use_bias=False
                )
            )
            model.add(tf.keras.layers.Dense(units=25, use_bias=False))
            model.add(tf.keras.layers.Dense(units=10, use_bias=False))

            assert (
                26000 == res.get_model_weight_size(model, ["float"])["float"]
            )
