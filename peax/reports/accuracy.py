import pathlib
import glob
import logging as log
from typing import List, Union, Tuple, Dict, Set, Iterable
from typing_extensions import Self
import math
import json
import socket
import hashlib
import pickle as pkl
import os
import copy

from jinja2 import Template
#import dill

import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, explained_variance_score, max_error
import numpy as np

import peax.analysis as aNN

from ..components import predictive as prd

from . import base
from . import dataset as data
from . import batch_size as bs

class AccuracyReportSubmitter(base.ReportSubmitter):
    """
    Class that acts as syntactic sugar for the submission of reporters to the ModelAnalysis object
    """

    def __init__(self, analysis: aNN.ModelAnalysis, lazy: bool = False):
        """
        Constructor for the Submitter object

        Args:
            analysis (aNN.ModelAnalysis): the ModelAnalysis to which the reporter will be submitted
            lazy (bool, optional): The submission behavior. lazy will only create the report if it is required, otherwise it will be generated immediately.
            Defaults to False.
        """
        super().__init__(analysis, lazy)

    def with_config(self, datasets : Set[data.DatasetReport], metrics:Dict[str, callable]=None, max_batch_size:int=256) -> Union[base.Report, str]:
        """
        Creates the reporter with the given parameters and submits it to the analysis object that has been assigned to the submitter

        Args:
            datasets (Set[data.DatasetReport]): The datasets that will be evaluated by the AccuracyReport
            metrics (Dict[str, callable], optional): The metrics that will be evaluated, use None for the default metrics. Defaults to None.
            max_batch_size (int, optional): The maximum batch size that will be used while creating the AccuracyReport. Defaults to 256.

        Returns:
            Union[base.Report, str]: Returns the report object if lazy is False, 
            otherwise the key of the reporter within the analysis object will be returned and can be used to create the report if needed
        """
        acc_report, acc_r_uid = AccuracyReport.closure(
            datasets=datasets,
            metrics=metrics,
            max_batch_size=max_batch_size,
            create_id=True
            )
        
        self.analysis.submit_reporter(acc_r_uid, acc_report)

        if self.lazy:
            return acc_r_uid
        else:
            return self.analysis.access_report(acc_r_uid)



class AccuracyReport(base.Report):
    """Report that analysis the prediction performance (accuracy) of the base model"""

    __pkl_name = "report_accuracy_<hash>.pkl"

    metrics : Dict[str, callable]
    """The metrics that will be evaluated"""

    datasets : Dict[data.DatasetReport, str]
    """The datasets that have been used for this evaluation, mapped to their IDs in the analysis object"""

    results : Dict[data.DatasetReport, Dict[str, object]]
    """The results that have been achieved on the dataset"""

    __classification_metrics = {
        "top1_accuracy" : lambda y_true, y_pred: 100*accuracy_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)),
        "top5_accuracy" : lambda y_true, y_pred: 100*top_k_accuracy_score(np.argmax(y_true, axis=-1), y_pred, k=5),
        "precision" : lambda y_true, y_pred: 100*precision_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average="weighted"),
        "recall" : lambda y_true, y_pred: 100*recall_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average="weighted"),
        "f1" : lambda y_true, y_pred: 100*f1_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average="weighted"),
        "confusion_matrix" : lambda y_true, y_pred: confusion_matrix(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)),
        #"roc" : roc_curve,
        #"auc_roc" : roc_auc_score,
    }
    """Default classification metrics"""

    __binary_classification_metrics = {
        "top1_accuracy" : lambda y_true, y_pred: 100*accuracy_score(y_true, y_pred),
    }
    """Default binary classification metrics"""

    __regression_metrics = {
        "mean_squared_error" : mean_absolute_error,
        "mean_absolut_error" : mean_absolute_error,
        "root_mean_squared_error" : lambda y_true, y_pred:  np.sqrt(mean_squared_error(y_true, y_pred)),
        "median_absolut_error" : median_absolute_error,
        "r_squared" : r2_score,
        "explained_variance" : explained_variance_score,
        "max_error" : max_error
    }
    """Default regression metrics"""

    def __init__(
        self,
        analysis: aNN.ModelAnalysis,
        datasets : Set[data.DatasetReport],
        metrics : Dict[str, callable] = None,
        max_batch_size : int = 128,
        deserialize:bool = False
    ) -> None:
        """
        Constructor of the AccuracyReport class

        Args:
            analysis (aNN.ModelAnalysis): The ModelAnalysis to which the report will be assigned
            datasets (Set[data.DatasetReport]): The datasets that will be used to evaluate the accuracy of the model
            metrics (Dict[str, callable], optional): The metrics and their functions that will be used to evaluate the model. Defaults to None.
            max_batch_size (int, optional): The maximum batch size used for the creation of the report. Defaults to 128.
        """
        super().__init__(analysis)

        self.bs_estimator : bs.BatchSizeReport = bs.BatchSizeReport.submit_to(analysis=analysis, lazy=False).with_config(end_size=max_batch_size)
        self.max_batch_size = max_batch_size
        self.batch_size = self.__determine_batch_size(max_size=self.max_batch_size)

        if not deserialize:
            if list(self.analysis.tasks.values())[-1] in [prd.Task.CLASSIFICATION]:
                self.metrics = copy.deepcopy(self.__classification_metrics)

            if list(self.analysis.tasks.values())[-1] in [prd.Task.BINARY_CLASSIFICATION]:
                self.metrics = copy.deepcopy(self.__binary_classification_metrics)
            
            elif list(self.analysis.tasks.values())[-1] in [prd.Task.REGRESSION]:
                self.metrics = copy.deepcopy(self.__regression_metrics)

            if metrics is not None:
                self.metrics.update(metrics)
        else:
            self.metrics = {}
        
        self.datasets = {}
        self.results = {}

        if not deserialize:
            for dataset in datasets:
                # should also be accessible via self.analysis.access_report(id), but this is quicker and just pointers anyway
                self.datasets[dataset] = dataset.access_id()
                if not self._check_cache(dataset):
                    self.results[dataset] = self._evaluate_metrics(dataset)
                else:
                    self.results[dataset] = self.__load_cache_config(dataset)
        self.__write_cache()

        pass

    def _evaluate_metrics(self, dataset : data.DatasetReport) -> Dict[str, object]:
        """
        Protected function used to evaluate the model on a given dataset.
        Creates the model predictions for that dataset, evaluates the performance by comparing the predictions
        with the ground truth data using the metric functions.
        TODO: add support for regression and other tasks.
        TODO: improve performance!

        Args:
            dataset (data.DatasetReport): the dataset used for the evaluation

        Raises:
            NotImplementedError: The function does not yet support multi-exit models

        Returns:
            Dict[str, object]: a dict containing the evaluation results for the evaluated metrics
        """
        #return self.__experimential_evaluate_model(dataset=dataset) # experimential for large datasets like augmented COCO VWW 96

        results = {}
        model = self.analysis.keras_model
        # Initialize lists to store y_true and y_pred
        y_true_list = []
        y_pred_list = []

        # Iterate over the dataset to get y_true and y_pred
        data = dataset.data
        batched_dataset = data.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        for features, labels in batched_dataset:
            y_true_list.append(labels.numpy())
            predictions = model.predict(features)  # Make predictions using the model
            y_pred_list.append(predictions)  # Store the predictions

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        if len(self.analysis.tasks) > 1:
            raise NotImplementedError("this report was not implemented with multi-output models in mind, please contact the developer")

        if list(self.analysis.tasks.values())[0] == prd.Task.CLASSIFICATION:
            # multi-class problem with different representations for y_pred and y_true
            if len(y_pred.shape) > len(y_true.shape) or y_pred.shape[-1] != y_true.shape[-1]:
                y_true = to_categorical(y_true, num_classes=y_pred.shape[-1])

        elif list(self.analysis.tasks.values())[0] == prd.Task.BINARY_CLASSIFICATION:

            if len(y_pred.shape) != len(y_true.shape):
                threshold = 0.5
                y_pred = np.reshape(y_pred, y_true.shape) #np.argmax(y_pred, axis=-1)
                y_pred = (y_pred > threshold).astype(int)

        for name, func in self.metrics.items():
            try:
                results[name] = func(y_true, y_pred)
            except:
                results[name] = None

        return results

    def __experimential_evaluate_model(self, dataset : data.DatasetReport) -> Dict[str, object]:

        results = {name: [] for name in self.metrics}
        model: tf.keras.models.Model = self.analysis.keras_model

        data: tf.data.Dataset = dataset.data
        batched_dataset: tf.data.ShuffleDataset = data.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Evaluate the model on the dataset pipeline
        for batch in batched_dataset:
            inputs, labels = batch
            predictions = model(inputs, training=False)
            for metric_name, metric_fn in self.metrics.items():
                result = metric_fn(labels, predictions)
                results[metric_name].append(result)

        # Compute the final results by aggregating the list
        final_results = {}
        for metric_name, metric_values in results.items():
            final_results[metric_name] = sum(metric_values) / len(metric_values)  # compute the mean

        return results

    @property
    def __cache_path(self) -> pathlib.Path:
        """Function to generate a standardized path to the cached files

        Returns:
            pathlib.Path: the path where the necessary files should be located
        """
        # Compute a hash of the weights
        weights = self.analysis.keras_model.get_weights()
        hasher = hashlib.sha256()
        for w in weights:
            hasher.update(w.tobytes())

        weight_hash = hasher.hexdigest()

        hash_string = hashlib.sha256(f"acc_report_{self.analysis.name}".encode()).hexdigest()
        #dataset_string = hashlib.sha256(f"{[(key, dataset.name) for key, dataset in self.datasets.items()]}".encode()).hexdigest()
        metric_string = hashlib.sha256(f"{[key for key in self.metrics.keys()]}".encode()).hexdigest()
        hash_string = hashlib.sha256(f"{hash_string}_{metric_string}_v{weight_hash}".encode()).hexdigest()

        path = self.analysis.cache_dir / hash_string

        return path

    def __write_cache(self) -> None:
        """Writes the accuracy report to the cache_dir that has been defined by the ModelAnalysis object
        """
        path = self.__cache_path
        path.mkdir(parents=True, exist_ok=True)

        for dataset, results in self.results.items():
            config_path = path / f"accuracy_{dataset.access_id()}.data"

            with open(config_path, "wb") as f:
                pkl.dump(results, f)

        return

    def __load_cache_config(self, dataset) -> Dict[str, str]:
        """loads the configuration from the cache

        Returns:
            Dict[str, str]: the configuration and report parameters
        """
        results = None
        path = self.__cache_path

        config_path = path / f"accuracy_{dataset.access_id()}.data"

        if path.exists() and path.is_dir() and config_path.exists() and config_path.is_file():
            # load config
            with open(config_path, "rb") as f:
                results = pkl.load(f)
        
        return results

    def _check_cache(self, dataset) -> bool:
        """check, if files have been cached for the current configuration

        Returns:
            bool: True, if cache files are present
        """
        path = self.__cache_path / f"accuracy_{dataset.access_id()}.data"

        return path.exists() and path.is_file()

    def __determine_batch_size(self, max_size=128) -> int:
        """determines the optimal batch size for the current computer to speed up the evaluation.
        The system utilizes a trial-and-error appraoch that ensures that the batch_size configuration
        will fit into the available memory and results in the most optimal runtime per sample

        Returns:
            int: optimal batch size
        """
        return self.bs_estimator.inference
    
    def access_id(self) -> str:
        
        return self.create_unique_id(
            datasets=self.datasets,
            metrics=self.metrics
        )

    def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> Tuple[str, pathlib.Path]:
        """Creates a HTML file that visualizes all informations of the current Report

        Args:
            folder_path (Union[str, pathlib.Path], optional): The target folder path, where the file will be stored. Defaults to None.

        Returns:
            Tuple[str, pathlib.Path]: the displayed title for links to this summary and the file path to the summary HTML file
        """

        _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'

        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)
        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        file_name = f"report_accuracy_{self.access_id()}.html"

        summary = self.dump(folder_path)

        with open(_template_path / "accuracy_report.html", "r") as file:
            template = Template(file.read())

        # Render the template with the summary data
        html = template.render(summary=summary)
        # Save the generated HTML to a file
        with open(folder_path / file_name, "w") as file:
            file.write(html)

        return "Accuracy Report", file_name
    
    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> Dict[str, object]:
        """Dumps the information from the report into a file and returns a dict with the most important information

        Args:
            folder_path (Union[str, pathlib.Path], optional): The target folder path, where the file will be stored. Defaults to None.

        Returns:
            Dict[str, object]: the relevant information as a dict.
        """
        
        summary = {
            "report_type": "Accuracy",
            "name": self.analysis.name,
            "creation_date": str(self.analysis.creation_time),
            "results" : {(dataset.name, dataset.access_id()): results for dataset, results in self.results.items()},
            "batch_size" : self.batch_size,
            "max_batch_size" : self.max_batch_size,
            #"metrics" : self.metrics,
        }

        file_name = self.__pkl_name.replace("<hash>", self.access_id())
        with open(folder_path / file_name, "wb") as file:
            pkl.dump(summary, file)

        return summary
    
    @classmethod
    def create_unique_id(cls, datasets, metrics) -> str:
        descr_str = f"AccReport-{datasets}-{metrics}"

        hashed_str = cls.create_reporter_id(descr_str)

        return hashed_str
    
    @classmethod
    def load(cls, folder_path: Union[str, pathlib.Path], analysis) -> Set[Self]:
        """
        Experimential function to load AccuracyReports back from a dump on disk

        Args:
            folder_path (Union[str, pathlib.Path]): the folder in which the method will search for AccuracyReportDumps
            analysis (_type_): The ModelAnalysis to which the AccuracyReports will be assigned

        Returns:
            Set[Self]: A set of all found AccuracyReports
        """
        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        file_pattern = cls.__pkl_name.replace("<hash>", "*")
        files = glob.glob(str(folder_path) + "/" + file_pattern)

        dataset_reports = data.DatasetReport.load(folder_path=folder_path, analysis=analysis)

        reports = list()
        for file_path in files:
            file_path = pathlib.Path(file_path)

            with open(file_path, "rb") as file:
                summary = pkl.load(file)

            
            #TODO: create AccuracyReport from summary
            new_report = AccuracyReport(analysis, None, None, max_batch_size=summary["max_batch_size"], deserialize=True)
            #new_report.metrics = summary["metrics"]
            new_report.metrics = None #copy.deepcopy(cls.__classification_metrics)

            if list(new_report.analysis.tasks.values())[-1] in [prd.Task.CLASSIFICATION]:
                new_report.metrics = copy.deepcopy(cls.__classification_metrics)

            if list(new_report.analysis.tasks.values())[-1] in [prd.Task.BINARY_CLASSIFICATION]:
                new_report.metrics = copy.deepcopy(cls.__binary_classification_metrics)
            
            elif list(new_report.analysis.tasks.values())[-1] in [prd.Task.REGRESSION]:
                new_report.metrics = copy.deepcopy(cls.__regression_metrics)
            log.warn("unable to restore custom metrics, no metrics have been restored!")

            #TODO: convert the dataset hash_ids back to the datasets, associate dataset reports with the results in the AccuracyReport
            for dataset_descriptor, results in summary["results"].items():
                #new_report.datasets
                dataset = [ds for ds in dataset_reports if ds.access_id() == dataset_descriptor[-1]][0]
                new_report.datasets[dataset] = dataset.access_id()

                new_report.results[dataset] = results

            reports.append(new_report)

        return reports
    
    def __str__(self):
        return f"AccuracyReport for {self.analysis.name} on {[dataset.name for dataset in self.results.keys()]}"
    
    def __repr__(self):
        return f"AccuracyReport(analysis={repr(self.analysis)}, datasets={repr(self.datasets)}, metrics={repr(self.metrics)}, max_batch_size={self.max_batch_size})"
    
    @classmethod
    def submit_to(cls, analysis : aNN.ModelAnalysis, lazy:bool=False) -> AccuracyReportSubmitter:
        """
        Syntactic sugar to simplify submission of new AccuracyReports to the ModelAnalysis
        i.e.: AccuracyReport.submit_to(analysis).with_config(...)

        Args:
            analysis (aNN.ModelAnalysis): the analysis to which the reporter should be submitted
            lazy (bool, optional): The submission behavior:
                lazy will only create the report if it is required, otherwise it will be generated immediately.
                Defaults to False.

        Returns:
            AccuracyReportSubmitter: The submitter auxiliary object that is used to provide the necessary syntax
        """
        return AccuracyReportSubmitter(analysis=analysis, lazy=lazy)
    
    @classmethod
    def data_closure(
        cls,
        analysis,
        paths : Union[pathlib.Path, Iterable[pathlib.Path]],
        names : Union[str, Iterable[str]],
        modality : prd.Task,
        metrics : Dict[str, callable] = None,
        max_batch_size : int = 256,
        input_preprocessors: Union[callable, Iterable[callable]] = None,
        label_preprocessors: Union[callable, Iterable[callable]] = None,
        create_id:bool=True,
    ):
        """An additional way to create a closure for an AccuracyReport, if no DatasetReports have been created yet

        Args:
            analysis (analysis.ModelAnalysis): the ModelAnalysis object to which the DatasetReports will be submitted
            paths (Union[pathlib.Path, Iterable[pathlib.Path]]): the paths to the datasets on your file system
            names (Union[str, Iterable[str]]): the names for the datasets (i.e. "training", "validation", "test")
            modality (prd.Task): the modality of the input data (i.e. IMAGE)
            metrics (Dict[str, callable], optional): The metrics you want to benchmark. Defaults to None.
            max_batch_size (int, optional): The maximum batch size that should be used for the processing. Defaults to 256.
            input_preprocessors (Union[callable, Iterable[callable]], optional): A preprocessing function for the input samples. Defaults to None.
            label_preprocessors (Union[callable, Iterable[callable]], optional): A postprocessing function for the output data. Defaults to None.
            create_id (bool, optional): if a reference ID should be created. Defaults to True.

        Returns:
            callable: The AccuracyReport closure
            (optional) int: the unique ID for this closure
        """
        if isinstance(paths, Iterable):
            assert isinstance(names, Iterable), "if multiple dataset paths are submitted, a list for the names need to be provided as well"
            assert len(paths) == len(names), "need to provide as many names as paths!"
        else:
            paths = [paths]

        if isinstance(names, Iterable):
            assert isinstance(paths, Iterable), "if multiple dataset paths are submitted, a list for the names need to be provided as well"
            assert len(paths) == len(names), "need to provide as many names as paths!"
        else:
            names = [names] * len(paths)

        if isinstance(input_preprocessors, Iterable):
            assert len(paths) == len(input_preprocessors), "need to provide as many preprocessors as datasets, or single preprocessor for input on all datasets!"
        else:
            input_preprocessors = [input_preprocessors] * len(paths)
        
        if isinstance(label_preprocessors, Iterable):
            assert len(paths) == len(label_preprocessors), "need to provide as many preprocessors as datasets, or single preprocessor for samples on all datasets!"
        else:
            label_preprocessors = [label_preprocessors] * len(paths)

        data_closures_ids = []
        for path, name, sample_prepro, label_prepro in zip(paths, names, input_preprocessors, label_preprocessors):
            data_reporter, reporter_id = data.DatasetReport.closure(create_id=True,
                                                                    data_preprocessor=sample_prepro,
                                                                    label_preprocessor=label_prepro,
                                                                    path=path,
                                                                    modality=modality,
                                                                    name=name
                                                                )
            data_closures_ids.append(reporter_id)
            analysis.submit_reporter(reporter_id, data_reporter)
        analysis.create_reports()

        datasets = set()
        for id in data_closures_ids:
            datasets.add(analysis.access_report(id))

        return cls.closure(datasets=datasets, metrics=metrics, max_batch_size=max_batch_size, create_id=create_id)

    @classmethod
    def closure(
        cls,
        datasets : Set[data.DatasetReport],
        metrics = None,
        max_batch_size = 256,
        create_id:bool=True
    ):
        """
        Closure that should be passed to the ModelAnalysis object.
        This function is also called Reporter when being passed around.
        The reporter creates the report if it is evaluated.
        """

        def builder(analysis: aNN.ModelAnalysis):
            return AccuracyReport(
                analysis=analysis,
                datasets=datasets,
                metrics=metrics,
                max_batch_size=max_batch_size,
            )
        
        if metrics is None:
            metrics = {}
        #metrics.update(cls.__classification_metrics)
        if create_id:
            metric_str = f"{[str(name) for name in metrics.keys()]}"
            dataset_str = f"{[dataset.access_id() for dataset in datasets]}"
            descr_str = f"AccReport:{dataset_str}-{metric_str}"
            hashed_str = cls.create_reporter_id(descr_str)

            return builder, hashed_str

        return builder