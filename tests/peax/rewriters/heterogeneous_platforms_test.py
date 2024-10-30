from typing import List, Dict, Union, Tuple, Set
import logging as log
import numbers
import copy

import numpy as np
import networkx as nx
import tensorflow as tf
import networkx as nx

from peax.hardware import processor as pr
from peax.hardware import connection as con

from peax.reports import early_exit as ee
from peax.reports import hw_checker as hw_check

from peax.components import resource as res
from peax.components import graph_tools as gt

import peax.analysis as aNN
from peax.rewriters.heterogeneous_platforms import MappingOption
from peax.rewriters.heterogeneous_platforms import HeterogeneousPlatformRewriter

import pytest


class TestMappingOption:
    @pytest.fixture
    def model(self):
        # Define model

        filters = 64
        output_size = 5

        # Define FFNN
        sample_shape = (10, 16, 64)
        inputs = tf.keras.Input(shape=sample_shape)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding="same", activation="relu"
        )(inputs)
        x = tf.keras.layers.Conv2D(
            filters=filters // 2, kernel_size=1, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=filters * 2, kernel_size=1, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(filters * 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        final_output = tf.keras.layers.Dense(
            units=output_size, activation="softmax"
        )(x)

        model = tf.keras.Model(
            inputs=inputs, outputs=[final_output], name="radar_model"
        )

        return model

    def test_add_mapping_valid(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations
        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)
        # Test adding a mapping that is valid
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        assert mapping_option.add_mapping(processor, node, costs) == True
        assert mapping_option.mappings == {processor: node}
        assert mapping_option.cost_map == {(processor, node): costs}

    def test_add_mapping_invalid(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()
        options = ee_report.recommendations
        #options = analysis.access_report(ee.EarlyExitReport)[0].recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        #print(options, processors)
        mapping_option = MappingOption(options, processors)

        #print(mapping_option)
        # Test adding a mapping that has already been added
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.add_mapping(processor, node, costs) == False

    def test_add_mapping_invalid(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        #print(options, processors)
        mapping_option = MappingOption(options, processors)

        #print(mapping_option)

        node = gt.BlockNode(options[0].subgraph)
        processor = processors[0]
        costs = {"delay": 1}

        # Test adding a mapping with an invalid node

        assert mapping_option.add_mapping(processor, node, costs) == False

    def test_add_mapping_invalid(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        #print(options, processors)
        mapping_option = MappingOption(options, processors)

        #print(mapping_option)

        node = options[0]
        processor = pr.Processor("invalid")
        costs = {"delay": 1}
        # Test adding a mapping with an invalid processor

        assert mapping_option.add_mapping(processor, node, costs) == False

    def test_get_mappings_empty(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting mappings when there are no mappings
        assert mapping_option.get_mappings() == []

    def test_get_mappings_one(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)
        # Test getting mappings when there is one mapping
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)
        assert mapping_option.get_mappings() == [(processor, node, costs)]

    def test_get_mappings_multiple(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)
        # Test getting mappings when there are multiple mappings
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        node = options[1]
        processor = processors[1]
        costs = {"delay": 2}
        mapping_option.add_mapping(processor, node, costs)

        mappings = mapping_option.get_mappings()
        assert mappings[0] == (processors[0], options[0], {"delay": 1})
        assert mappings[1] == (processors[1], options[1], {"delay": 2})

    def test_get_assigned_processors_empty(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors when there are no assigned processors
        assert mapping_option.get_assigned_processors() == []

    def test_get_assigned_processors_one(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.get_assigned_processors() == [processor]

    def test_get_assigned_processors_more(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        # Test getting assigned processors, when there is more than one processor assigned
        node = options[1]
        processor = processors[1]
        costs = {"delay": 2}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.get_assigned_processors() == processors

    def test_get_assigned_nodes_empty(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors when there are no assigned processors
        assert mapping_option.get_assigned_nodes() == []

    def test_get_assigned_nodes_one(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.get_assigned_nodes() == [node]

    def test_get_assigned_nodes_more(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        # Test getting assigned processors, when there is more than one processor assigned
        node = options[1]
        processor = processors[1]
        costs = {"delay": 2}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.get_assigned_nodes() == options

    '''def test_get_open_nodes_empty(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        assert mapping_option.get_open_nodes() == []'''

    def test_get_open_nodes_one_early(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.get_open_nodes() == [options[1]]

    def test_get_open_nodes_one_late(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[1]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.get_open_nodes() == []

    def test_accumulate_cost_empty(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        assert mapping_option.accumulate_cost("delay") == None

    def test_accumulate_cost_single_elem(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.accumulate_cost("delay") == 1

    def test_accumulate_cost_multi_elem(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        node = options[1]
        processor = processors[1]
        costs = {"delay": 2}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.accumulate_cost("delay") == 3

    def test_accumulate_cost_wrong_cost(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1}
        mapping_option.add_mapping(processor, node, costs)

        node = options[1]
        processor = processors[1]
        costs = {"delay": 2}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.accumulate_cost("time") == None

    def test_accumulate_cost_multiple_keys(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"delay": 1, "memory": 50}
        mapping_option.add_mapping(processor, node, costs)

        node = options[1]
        processor = processors[1]
        costs = {"delay": 2, "memory": 25}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.accumulate_cost("delay") == 3
        assert mapping_option.accumulate_cost("memory") == 75

    def test_accumulate_cost_boolean_cost_true(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"support": True}
        mapping_option.add_mapping(processor, node, costs)

        node = options[1]
        processor = processors[1]
        costs = {"support": True}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.accumulate_cost("support") == True

    def test_accumulate_cost_boolean_cost_false(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option = MappingOption(options, processors)

        # Test getting assigned processors, when there is one processor assigned
        node = options[0]
        processor = processors[0]
        costs = {"support": True}
        mapping_option.add_mapping(processor, node, costs)

        node = options[1]
        processor = processors[1]
        costs = {"support": False}
        mapping_option.add_mapping(processor, node, costs)

        assert mapping_option.accumulate_cost("support") == False

    # TODO: add more tests for accumulate_cost, str

    def test_create_copy_empty(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option1 = MappingOption(options, processors)

        mapping_option2 = mapping_option1.create_copy()

        assert mapping_option1 == mapping_option2

    def test_create_copy_mapped(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option1 = MappingOption(options, processors)
        node = options[0]
        processor = processors[0]
        costs = {"support": True}
        mapping_option1.add_mapping(processor, node, costs)

        mapping_option2 = mapping_option1.create_copy()

        assert mapping_option1.get_mappings() == mapping_option2.get_mappings()
        assert mapping_option1 == mapping_option2

    def test_create_copy_mapped_same(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option1 = MappingOption(options, processors)
        node = options[0]
        processor = processors[0]
        costs = {"support": True}
        mapping_option1.add_mapping(processor, node, costs)

        mapping_option2 = mapping_option1.create_copy()

        node = options[0]
        processor = processors[0]
        costs = {"support": True}
        mapping_option2.add_mapping(processor, node, costs)

        assert mapping_option1.get_mappings() == mapping_option2.get_mappings()
        assert mapping_option1 == mapping_option2

    def test_create_copy_mapped_different(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        ee_report = ee.EarlyExitReport.submit_to(analysis=analysis, lazy=False).with_config(search_config="small")
        #analysis.submit_reporter(ee.EarlyExitReport.closure())
        #analysis.create_reports()

        options = ee_report.recommendations

        processors = [pr.Processor("1"), pr.Processor("2")]

        mapping_option1 = MappingOption(options, processors)
        node = options[0]
        processor = processors[0]
        costs = {"support": True}
        mapping_option1.add_mapping(processor, node, costs)

        mapping_option2 = mapping_option1.create_copy()

        node = options[1]
        processor = processors[1]
        costs = {"support": False}
        mapping_option2.add_mapping(processor, node, costs)

        assert mapping_option1.get_mappings() != mapping_option2.get_mappings()
        assert mapping_option1 != mapping_option2


class TestHeterogeneousPlatformRewriter:
    @pytest.fixture
    def model(self):
        # Define model

        filters = 64
        output_size = 5

        # Define FFNN
        sample_shape = (10, 16, 64)
        inputs = tf.keras.Input(shape=sample_shape)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding="same", activation="relu"
        )(inputs)
        x = tf.keras.layers.Conv2D(
            filters=filters // 2, kernel_size=1, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=filters * 2, kernel_size=1, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(filters * 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        final_output = tf.keras.layers.Dense(
            units=output_size, activation="softmax"
        )(x)

        model = tf.keras.Model(
            inputs=inputs, outputs=[final_output], name="radar_model"
        )

        return model

    @pytest.fixture
    def analysis(self, model):
        analysis = aNN.ModelAnalysis(model=model)
        return analysis

    @pytest.fixture
    def processors(self):
        processors = [pr.Processor("1"), pr.Processor("2")]
        return processors

    @pytest.fixture
    def connections(self, processors):
        connections = [
            con.Connection(
                start=processors[0], end=processors[1], bandwidth=268435456
            ),
            con.Connection(
                start=processors[1], end=processors[0], bandwidth=268435456
            ),
        ]
        return connections

    def test_init_raises_error_if_no_processors(self, analysis, connections):
        with pytest.raises(ValueError, match="processors cannot be None"):
            HeterogeneousPlatformRewriter(
                analysis=analysis,
                latency=500,
                processors=None,
                connections=connections,
                dtypes={"int8", "float32"},
            )

    def test_init_raises_error_if_empty_processors(
        self, analysis, connections
    ):
        with pytest.raises(ValueError, match="processors cannot be empty"):
            HeterogeneousPlatformRewriter(
                analysis=analysis,
                latency=500,
                processors=[],
                connections=connections,
                dtypes={"int8", "float32"},
            )

    def test_init_raises_error_if_no_connections(self, analysis, processors):
        with pytest.raises(ValueError, match="connections cannot be None"):
            HeterogeneousPlatformRewriter(
                analysis=analysis,
                latency=500,
                processors=processors,
                connections=None,
                dtypes={"int8", "float32"},
            )

    def test_init_raises_error_if_empty_connections(
        self, analysis, processors
    ):
        with pytest.raises(ValueError, match="connections cannot be empty"):
            HeterogeneousPlatformRewriter(
                analysis=analysis,
                latency=500,
                processors=processors,
                connections=[],
                dtypes={"int8", "float32"},
            )

    '''def test_extract_all_options(self, analysis, processors, connections):
        optimizer = HeterogeneousPlatformRewriter(
            analysis=analysis,
            latency=500,
            processors=processors,
            connections=connections,
            dtypes={"int8", "float32"},
        )
        options = optimizer.extract_all_options()
        assert isinstance(options, list)'''
