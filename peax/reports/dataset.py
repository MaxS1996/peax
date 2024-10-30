import json
import pathlib
from typing import List, Tuple, Union, Optional, Dict, Set
from typing_extensions import Self
import hashlib
import inspect
import socket
import os
import glob
import logging as log

import tensorflow as tf
import numpy as np
from jinja2 import Template

import peax.analysis as aNN
from . import base

from ..components import predictive as prd
from peax import reports

class DatasetReporterSubmitter(base.ReportSubmitter):
    """
    Syntactic Sugar class for easy submission of Report(ers) to the ModelAnalysis
    """

    def __init__(self, analysis: aNN.ModelAnalysis, lazy: bool = False, name:str=None, modality:prd.Modality=None, data_preprocessor:callable=None, label_preprocessor:callable=None):
        super().__init__(analysis, lazy)
        self.name = name
        self.modality = modality
        self.data_preprocessor = data_preprocessor
        self.label_preprocessor = label_preprocessor

    def with_config(self, name:str, modality:prd.Modality, data_preprocessor:callable=None, label_preprocessor:callable=None) -> base.ReportSubmitter:
        """
        Returns a submitter where these additional parameters have been set.

        Args:
            name (str): _description_
            modality (prd.Modality): _description_
            data_preprocessor (callable, optional): _description_. Defaults to None.
            label_preprocessor (callable, optional): _description_. Defaults to None.

        Returns:
            base.ReportSubmitter: _description_
        """
        return DatasetReporterSubmitter(analysis=self.analysis, lazy=self.lazy, name=name, modality=modality, data_preprocessor=data_preprocessor, label_preprocessor=label_preprocessor)
    
    def from_source(self, path:Union[str, pathlib.Path]=None, data_loader:callable=None, tf_dataset:tf.data.Dataset=None, numpy_arrays:Set[np.array]=None):
        if numpy_arrays is not None:
            data_reporter, data_rep_id = DatasetReport.from_numpy(
                create_id=True,
                data_preprocessor=self.data_preprocessor,
                label_preprocessor=self.label_preprocessor,
                numpy_arrays=numpy_arrays,
                modality=self.modality, name=self.name
            )
            ###TODO: complete!
        elif tf_dataset is not None:
            data_reporter, data_rep_id = DatasetReport.from_tf_dataset(
                create_id=True,
                data_preprocessor=self.data_preprocessor,
                label_preprocessor=self.label_preprocessor,
                modality=self.modality,
                name=self.name,
                dataset=tf_dataset)
        elif path is not None:
            data_reporter, data_rep_id = DatasetReport.closure(
                create_id=True,
                data_preprocessor=self.data_preprocessor,
                label_preprocessor=self.label_preprocessor,
                modality=self.modality,
                name=self.name,
                path=path,
                data_loader=data_loader)
            
        self.analysis.submit_reporter(data_rep_id, data_reporter)

        if self.lazy:
            return data_rep_id
        else:
            return self.analysis.access_report(data_rep_id)


class DatasetReport(base.Report):
    """
    Contains information about the wrapped dataset as well as the data itself.
    The data can be provided as numpy array, tf.data.Dataset or by using a (custom) dataloader.
    tf.data.Dataset is the best tested solution here.
    """
    #TODO: currently only covering Classification Use Cases, need to generalize to regression, segmentation, etc
    __file_prefix = "report_dataset"
    
    path: pathlib.Path
    """ Path from where the dataset was loaded """
    modality: prd.Modality
    """ Data modality of the used dataset"""
    data_preprocessor: callable
    """ Preprocessing function that will be applied elementwise to the dataset samples"""
    label_preprocessor : callable
    """ Preprocessing function that will be applied elementwise to the output labels"""
    name : str
    """ Name of this specific dataset"""
    size : int
    """ Number of samples in dataset"""
    input_shape : Tuple[int, ...]
    """ The shape of the input sample after applying the preprocessor"""
    output_shape : Tuple[int, ...]
    """ The shape of the output label after applying the preprocessor"""
    label_distribution : Tuple[int, ...]
    """ The distribution of samples across the classes/labels"""
    cached : bool
    """ True, if this is cached data that is reused from previous runs """
    shuffled : bool
    """ True, if the data has been shuffled, False otherwise """

    def __init__(self,
                 analysis: aNN.ModelAnalysis,
                 data_preprocessor : callable,
                 label_preprocessor : callable,
                 modality : prd.Modality,
                 name : str,
                 path : str = None,
                 shuffle : bool = True,
                 data_loader:callable=None,
                 tf_dataset:tf.data.Dataset = None,
                 deserialize:bool=False,
                ) -> None:
        
        """
        Creates a new instance of the DatasetReport class.
        The __init__ function should not be used directly,
        instead the Submitter or _from_<datatype> functions are the recommended way to create the instance
        """
        
        if analysis is not None:
            super().__init__(analysis)
        else:
            self.analysis = None
            self.hide = False
            self._graph = None #self.analysis.architecture.network_graph
            self._block_graph = None #self.analysis.architecture.block_graph

        self.name = None
        self.modality = None
        self.data_preprocessor = None
        self.label_preprocessor = None

        if not deserialize:
            self.name = name
            self.modality = modality
            self.data_preprocessor = data_preprocessor
            '''def identify(x):
                    return x
            if data_preprocessor is None:
                self.data_preprocessor = identify'''
            self.label_preprocessor = label_preprocessor
            '''if label_preprocessor is None:
                self.label_preprocessor = identify'''

        self.cached = None
        self.path = None
        self.data_loader = None
        self.data = None
        self.from_tf_data = None

        if not deserialize:
            self.cached = False
            if tf_dataset is None:
                log.info(f"loading data from {path}")
                if isinstance(path, str):
                    path = pathlib.Path(path)
                
                self.path = path
                self.data_loader = data_loader
                self.data = self._load_data(data_loader=self.data_loader)
                self.from_tf_data = False
            else:
                self.data = tf_dataset
                self.from_tf_data = True
                self.data = self._preprocess().prefetch(tf.data.AUTOTUNE)

                if self.data.element_spec[0].shape[0] is None:
                    self.data = self.data.unbatch()

                self.path = str(tf_dataset)

            self.cached = self._check_cache()
        
        self.size = None
        self.shuffled = None

        if not deserialize:
            try:
                self.size = len(self.data)
            except TypeError:
                self.size = self.data.reduce(0, lambda x, _: x + 1).numpy()
            
            self.shuffled = False
            if shuffle:
                self.data = self.data.shuffle(buffer_size=self.size*2, reshuffle_each_iteration=False)
                self.shuffled = True

        self.input_shape = None
        self.output_shape = None
        self.label_distribution = None

        if not deserialize:
            self.input_shape, self.output_shape = self._get_shapes()
            self.label_distribution = self._get_label_distribution()

            if list(analysis.tasks.values())[-1] == prd.Task.CLASSIFICATION:
                log.debug("need to check, if y needs to be converted to one-hot representation")

                # check shape of output y-vectors
                if len(self.output_shape) < 1 or self.output_shape[-1] == 1:
                    # need to convert to one-hot
                    #num_classes = len(self.label_distribution) #here lies the issue
                    num_classes = self.analysis.keras_model.output_shape[-1]
                    def one_hot_encode(inputs, labels):
                        return inputs, tf.one_hot(labels, depth=num_classes)

                    self.data = self.data.map(one_hot_encode)
                    self.input_shape, self.output_shape = self._get_shapes()
                    log.info("converted labels to one-hot")

            self.__write_cache(self.data)

        pass

    def shuffle(self) -> None:
        """enables dataset to be shuffled after loading it
        """
        if self.shuffled:
            return
        
        self.shuffled = True
        self.data = self.data.shuffle(self.size*2, reshuffle_each_iteration=False)
        return
        

    def _load_data(self, data_loader : callable = None) -> tf.data.Dataset:
        """
        Loads the dataset as tf.data.Dataset into self.data.
        If the data has been loaded and preprocessed before, this version will be loaded from cache.
        You can pass a custom dataloader, which requires to have directory as input argument,
        which points to the directory were the data is stored.

        Args:
            data_loader (callable, optional): A function that will be called to load the data from directory into a tf.data.Dataset object. Defaults to None.

        Raises:
            NotImplementedError: Currently raises this error if you try to load non-image data

        Returns:
            tf.data.Dataset: the data as a Dataset object
        """ 
        if data_loader is None:
            def loader(directory):
                data = tf.keras.utils.image_dataset_from_directory(
                    directory=directory,
                    labels="inferred",
                    label_mode="int",
                    class_names=None,
                    color_mode="rgb",
                    batch_size=1,
                    image_size=self.analysis.keras_model.inputs[0].shape._dims[1:-1],
                    shuffle=False,
                    seed=None,
                    validation_split=None,
                    subset=None,
                    interpolation="bilinear",
                    follow_links=False,
                    crop_to_aspect_ratio=False,
                )
                return data
            data_loader = loader

            self.data = data_loader(self.path)
            self.data = self.data.unbatch()

        data = self._preprocess()

        '''data = None
        if self.cached == False:
            if self.modality == prd.Modality.IMAGE:
                data = data_loader(self.path)
                data = data.map(lambda x, y: (self.data_preprocessor(x), self.label_preprocessor(y)))

            else:
                raise NotImplementedError("Support for loading datasets of none-image data has not yet been implemented")
        
            self.__write_cache(dataset=data)
        
        else:
            data = self.__load_cached()'''

        return data
    
    def _preprocess(self) -> None:
        """
        Applies the contained preprocessing functions to the samples and labels of the dataset

        Returns:
            None
        """
        if self.cached == False:
            if self.data_preprocessor is not None and self.label_preprocessor is not None:
                data = self.data.map(lambda x, y: (self.data_preprocessor(x), self.label_preprocessor(y)))
            elif self.data_preprocessor is not None and self.label_preprocessor is None:
                data = self.data.map(lambda x, y: (self.data_preprocessor(x), y))
            elif self.data_preprocessor is None and self.label_preprocessor is not None:
                data = self.data.map(lambda x, y: (x, self.label_preprocessor(y)))
            else:
                data = self.data
            #self.__write_cache(dataset=data)
        else:
            data = self.__load_cached()

        return data
    
    def _provide_cache_path(self) -> pathlib.Path:
        """Function to generate a standardized path to the cached files

        Returns:
            pathlib.Path: the path where the necessary files should be located
        """
        hostname_hash = hashlib.sha256(socket.gethostname().encode()).hexdigest()

        if self.data_preprocessor is not None:
            source_code = inspect.getsource(self.data_preprocessor)
        else:
            source_code = "no_preprocessing"
        dp_hash = hashlib.sha256(source_code.encode()).hexdigest()
        
        if self.label_preprocessor is not None:
            source_code = inspect.getsource(self.label_preprocessor)
        else:
            source_code = "no_postprocessing"
        lp_hash = hashlib.sha256(source_code.encode()).hexdigest()

        if hasattr(self, "path") and not self.from_tf_data:
            identifier = self.path
        else:
            identifier = self.name
            log.warn("dataset has been created from tf.data.Dataset object, cannot detect changes to input dataset and might reuse stale cache data")

        hash_string = hashlib.sha256(f"dataset_report_{self.analysis.name}_{identifier}_{dp_hash}_{lp_hash}".encode()).hexdigest()
        hash_string = f"{hash_string}_{hostname_hash}"

        path = self.analysis.cache_dir / hash_string

        return path
    
    def _check_cache(self) -> bool:
        """check, if files have been cached for the current configuration

        Returns:
            bool: True, if cache files are present
        """
        path = self._provide_cache_path()

        return path.exists() and path.is_dir()
    
    def __write_cache(self, dataset) -> None:
        """Writes the accuracy report to the cache_dir that has been defined by the ModelAnalysis object
        """
        path = self._provide_cache_path()
        path.mkdir(parents=True, exist_ok=True)

        dataset_path = path / "dataset.tfds"

        # store dataset
        if dataset is not None:
            dataset.save(str(dataset_path))

        self.cached = True

        return
    
    def __load_cached(self) -> tf.data.Dataset:
        """loads the already preprocessed data from cache

        Returns:
            tf.data.Dataset: the preprocessed data that was generated in a previous run
        """

        dataset = None

        path = self._provide_cache_path() #what is going on?

        dataset_path = path / "dataset.tfds"

        if path.exists() and path.is_dir() and dataset_path.exists():
            # load dataset
            dataset = tf.data.Dataset.load(str(dataset_path)).prefetch(tf.data.AUTOTUNE)

        return dataset
    
    def _get_shapes(self) -> Tuple[Tuple[int]]:
        """Extracts the input and output tensor shapes after applying
        the preprocessing functions from the wrapped tf.data.Dataset object.

        Returns:
            Tuple[Tuple[int]]: input_tensor_shape, output_tensor_shape
        """
        sample = next(iter(self.data))
        images, labels = sample

        return list(images.shape), list(labels.shape)
    
    def _get_label_distribution(self) -> Tuple[int, ...]:
        """Generates a overview of the distribution of the used class labels.
        Useful to check if the dataset has been balanced

        Returns:
            Tuple[int, ...]: A tuple with as many elements as there are classes in the dataset.
            Each element equals the total number of samples belonging to that class.
        """
        labels = np.array([label.numpy() for _, label in self.data])

        label_counts = np.unique(labels, return_counts=True, axis=0)
        return tuple(label_counts[-1])

    '''def assess_quality(self) -> Dict[str, object]:
        """This function is intended to give you feedback on the quality of your dataset.
        EXPERIMENTIAL FEATURE, WIP!

        Returns:
            Dict[str, object]: _description_
        """

        class_distribution = self._get_label_distribution()

        if self.modality == prd.Modality.IMAGE:
            # we need to handle image data
            channel_count = self.input_shape[-1]
            
            # value distribution per channel
            #pixel_mean = tf.math.reduce_mean(image, axis=(0, 1))
            #TODO
            print("not implemented yet")'''

    def dump(self, folder_path: Union[str, pathlib.Path] = None) -> Dict[str, object]:
        """
        Writes the report object to disk

        Args:
            folder_path (Union[str, pathlib.Path], optional): the folder in which the report file will be created. Defaults to None.

        Returns:
            Dict[str, object]: _description_
        """

        if self.data_preprocessor is None:
            data_preprocess = "identity"
            data_preprocess_source = ""
        else:
            data_preprocess = self.data_preprocessor.__name__
            data_preprocess_source = inspect.getsource(self.data_preprocessor).lstrip().rstrip()

        if self.label_preprocessor is None:
            label_preprocess = "identity"
            label_preprocess_source = ""
        else:
            label_preprocess = self.label_preprocessor.__name__
            label_preprocess_source = inspect.getsource(self.label_preprocessor).lstrip().rstrip()
        
        summary = {
            "report_type": "Dataset",
            "analysis_name": self.analysis.name,
            "name": self.name,
            "creation_date": str(self.analysis.creation_time),
            "path" : str(self.path),
            "from_tf_data" : self.from_tf_data,
            "size" : int(self.size),
            "input_shape" : [int(x) for x in self.input_shape],
            "output_shape" : [int(x) for x in self.output_shape],
            "modality" : str(self.modality),
            "data_preprocessor_name" : data_preprocess,
            "label_preprocessor_name" : label_preprocess,
            "data_preprocessor_source" : data_preprocess_source,
            "label_preprocessor_source" : label_preprocess_source,
            "cached" : self.cached,
            "shuffled" : self.shuffled,
            "label_distribution" : [(idx, int(count)) for idx, count in enumerate(self.label_distribution)],
        }

        with open(folder_path / f"{self.__file_prefix}_{self.name}_{hash(self)}.json", "w") as file:
            json.dump(summary, file)

        return summary
    
    @classmethod
    def load(cls, folder_path: Union[str, pathlib.Path], analysis : aNN.ModelAnalysis) -> Set[Self]:
        """
        loads the dataset report from disk. UNTESTED function, preprocessing functions might be a significant security and stability issues.

        Args:
            folder_path (Union[str, pathlib.Path]): the folder in which the dataset reports are searched
            analysis (aNN.ModelAnalysis): the analysis to which the found reports will be assigned

        Returns:
            Set[DatasetReport]: The found dataset reports
        """
        
        if not isinstance(folder_path, pathlib.Path):
            folder_path = pathlib.Path(folder_path)

        file_pattern = cls.__file_prefix + "_*"
        files = glob.glob(str(folder_path) + "/" + file_pattern)

        reports = list()
        for file_path in files:
            file_path = pathlib.Path(file_path)
            with open(file_path, "r") as file:
                summary = json.load(file)

            new_report = DatasetReport(analysis=analysis,
                data_preprocessor=None,
                label_preprocessor=None,
                modality=None,
                name=None,
                path=None, 
                shuffle=None,
                deserialize=True)

            new_report.name = summary["name"]

            new_report.from_tf_data = summary["from_tf_data"]
            if new_report.from_tf_data:
                new_report.path = summary["path"]
            else:
                new_report.path = pathlib.Path(summary["path"])
                
            new_report.size = summary["size"]
            new_report.cached = summary["cached"]
            new_report.shuffled = summary["shuffled"]

            new_report.input_shape = tuple(summary["input_shape"])
            new_report.output_shape = tuple(summary["output_shape"])

            if summary["data_preprocessor_name"] == "identity":
                new_report.data_preprocessor = None
            else:
                compiled_data_preprocessor = compile(source=summary["data_preprocessor_source"], filename="<string>", mode="exec")
                def func(x):
                    return exec(compiled_data_preprocessor)
                    
                new_report.data_preprocessor = func
                log.warn("restoring preprocessing functions has not yet been implemented")
            
            if summary["label_preprocessor_name"] == "identity":
                new_report.label_preprocessor = None
            else:
                compiled_label_preprocessor = compile(source=summary["label_preprocessor_source"], filename="<string>", mode="exec")
                def func(x):
                    return exec(compiled_label_preprocessor)
                new_report.label_preprocessor = func
                log.warn("restoring preprocessing functions has not yet been implemented")

            new_report.label_distribution = tuple(summary["label_distribution"])

            new_report.cached = new_report._check_cache()
            try:
                new_report.data = new_report.__load_cached()
            except:
                log.warn("the contained data could not be restored, please submit the data as tf.data.Dataset manually to the report")

            new_report.analysis = analysis
            new_report.hide = False
            new_report._graph = analysis.architecture.network_graph
            new_report._block_graph = analysis.architecture.block_graph

            reports.append(new_report)

        return reports
    
    def render_summary(self, folder_path: Union[str, pathlib.Path] = None) -> Tuple[str, str]:
        """
        Creates a HTML-based summary of the information from this report

        Args:
            folder_path (Union[str, pathlib.Path], optional): Folder in which the summary will be created, uses the current working directory if None. Defaults to None.

        Returns:
            Tuple[str, str]: Title for the link that can be created in the ModelAnalysis summary, and the path to the summary of this report
        """

        _template_path = pathlib.Path(os.path.dirname(__file__)) / '..' / 'templates'
        
        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)
        if folder_path is None:
            folder_path = pathlib.Path.cwd()

        file_name = f"report_dataset_{self.name}_{self.access_id()}.html"

        summary = self.dump(folder_path)

        with open(_template_path / "dataset_report.html", "r") as file:
            template = Template(file.read())

        # Render the template with the summary data
        html = template.render(summary=summary)
        # Save the generated HTML to a file
        with open(folder_path / file_name, "w") as file:
            file.write(html)

        return f"{self.name} Dataset", file_name
    
    def access_id(self) -> str:
        """
        returns the unique ID of this report

        Returns:
            str: the unique identifier
        """
        # this should be rewritten to use the hash dunder
        return self.create_unique_id(
            data_preprocessor=self.data_preprocessor,
            label_preprocessor=self.label_preprocessor,
            path=self.path,
            modality=self.modality,
            name=self.name,
            shuffle=self.shuffled,
            )
    
    def __hash__(self):
        return hash(self.access_id())
    
    @classmethod
    def submit_to(cls, analysis:aNN.ModelAnalysis, lazy:bool=False) -> DatasetReporterSubmitter:
        """
        Syntactic sugar for the submission of reports to the analysis

        Args:
            analysis (aNN.ModelAnalysis): The ModelAnalysis
            lazy (bool, optional): The submission behavior. lazy will only create the report if it is required, otherwise it will be generated immediately. Defaults to False.

        Returns:
            DatasetReporterSubmitter: the auxiliary class that is used to simplify the submission process.
        """
        return DatasetReporterSubmitter(analysis=analysis, lazy=lazy)

    @classmethod
    def create_unique_id(cls,
                data_preprocessor : callable = None,
                label_preprocessor : callable = None,
                path : str = None,
                modality : prd.Modality = None,
                name : str = None,
                shuffle : bool = True) -> str:
        """
        Function to create a unique ID for an instance of this class.
        Implemented as class method to enable the creation of IDs for instances that do not yet exist.

        Args:
            data_preprocessor (callable, optional): The preprocessing function for the samples of the dataset. Defaults to None.
            label_preprocessor (callable, optional): The preprocessing function for the labels of the dataset. Defaults to None.
            path (str, optional): The path where the data is stored, not a path for tf.data.Datasets. Defaults to None.
            modality (prd.Modality, optional): currently unsed. Defaults to None.
            name (str, optional): name of the dataset. Defaults to None.
            shuffle (bool, optional): boolean, describes if the dataset has been shuffled. Defaults to True.

        Returns:
            str: the unique ID of the dataset report
        """
        
        descr_str = f"DatasetReport-{path}-{name}-{shuffle}"
        if data_preprocessor is not None:
            source_code = inspect.getsource(data_preprocessor).strip()
            dp_hash = hashlib.sha256(source_code.encode()).hexdigest()
            descr_str = f"{descr_str}:{dp_hash}"
        if label_preprocessor is not None:
            source_code = inspect.getsource(label_preprocessor).strip()
            lp_hash = hashlib.sha256(source_code.encode()).hexdigest()
            descr_str = f"{descr_str}:{lp_hash}"
        hashed_str = cls.create_reporter_id(descr_str)

        return hashed_str

    @classmethod
    def from_numpy(
        cls,
        create_id:bool=True,
        data_preprocessor : callable = None,
        label_preprocessor : callable = None,
        numpy_arrays : List[np.array] = None,
        modality : prd.Modality = None,
        name : str = None
    ) -> Tuple[callable, Optional[str]]:
        """
        Creates a DatasetReport from a tuple of numpy arrays.
        Both arrays should have the same length, the first contains the samples, the second the ground truth labels.
        The reporter can be passed to the ModelAnalysis object to create the report on demand (and as a singleton).

        Args:
            create_id (bool, optional): If True, not only the reporter, but also the unique ID of the report that can be created from it will be returned. Defaults to True.
            data_preprocessor (callable, optional): The preprocessing function that should be applied to the samples. Defaults to None.
            label_preprocessor (callable, optional): The preprocessing function that should be applied to the labels. Defaults to None.
            numpy_arrays (List[np.array], optional): The data. Defaults to None.
            modality (prd.Modality, optional): A description of the data modality (i.e. image, time-series, etc.). Defaults to None.
            name (str, optional): The name of the dataset, i.e. "training data", "validation data". Defaults to None.

        Returns:
            Tuple[callable, Optional[str]]: The reporter (constructor of the report) and optionally the reports unique ID
        """
        
        # create TF dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices(numpy_arrays)
        # create DatasetReporterClosure from TFDataset object
        return cls.from_tf_dataset(
            create_id=create_id,
            data_preprocessor=data_preprocessor,
            label_preprocessor=label_preprocessor,
            dataset=dataset,
            modality=modality,
            name=name
        )

    @classmethod
    def from_tf_dataset(cls,
                create_id:bool=True,
                data_preprocessor : callable = None,
                label_preprocessor : callable = None,
                dataset : tf.data.Dataset = None,
                modality : prd.Modality = None,
                name : str = None) -> Tuple[callable, Optional[str]]:
        """
        Creates a DatasetReport from a tensorflow dataset object.
        The dataset should contain samples and labels.
        The preprocessing functions can be passed alongside, if they have not yet been mapped to the dataset.
        The reporter can be passed to the ModelAnalysis object to create the report on demand (and as a singleton).

        Args:
            create_id (bool, optional): If True, not only the reporter, but also the unique ID of the report that can be created from it will be returned. Defaults to True.
            data_preprocessor (callable, optional): The preprocessing function that should be applied to the samples. Defaults to None.
            label_preprocessor (callable, optional): The preprocessing function that should be applied to the labels. Defaults to None.
            dataset (tf.data.Dataset, optional): The data. Defaults to None.
            modality (prd.Modality, optional): A description of the data modality (i.e. image, time-series, etc.). Defaults to None.
            name (str, optional): The name of the dataset, i.e. "training data", "validation data". Defaults to None.

        Returns:
            Tuple[callable, Optional[str]]: The reporter (constructor of the report) and optionally the reports unique ID
        """

        def builder(analysis: aNN.ModelAnalysis):
            return DatasetReport(analysis=analysis,
                            data_preprocessor=data_preprocessor,
                            label_preprocessor=label_preprocessor,
                            tf_dataset=dataset,
                            modality=modality,
                            name=name,
                            shuffle=False
                                )
        
        if create_id:
            hashed_str = cls.create_unique_id(
                data_preprocessor=data_preprocessor,
                label_preprocessor=label_preprocessor,
                path=hashlib.sha256(str(dataset).encode()).hexdigest(),
                modality=modality,
                name=name
            )

            return builder, hashed_str

        return builder

    @classmethod
    def closure(cls,
                create_id:bool=True,
                data_preprocessor : callable = None,
                label_preprocessor : callable = None,
                path : str = None,
                modality : prd.Modality = None,
                data_loader : callable = None,
                name : str = None) -> Tuple[callable, Optional[str]]:
        """
        returns the constructor of a new instance of this class as object.
        Can be used to pass it to the ModelAnalysis object.
        Not recommended for direct use!

        Args:
            create_id (bool, optional): If True, not only the reporter, but also the unique ID of the report that can be created from it will be returned. Defaults to True.
            data_preprocessor (callable, optional): The preprocessing function that should be applied to the samples. Defaults to None.
            label_preprocessor (callable, optional): The preprocessing function that should be applied to the labels. Defaults to None.
            path (str, optional): the path were the data is stored. Defaults to None.
            modality (prd.Modality, optional): A description of the data modality (i.e. image, time-series, etc.). Defaults to None.
            data_loader (callable, optional): the function that loads the data from the given path and turns it into a tensorflow dataset object. Defaults to None.
            name (str, optional): The name of the dataset, i.e. "training data", "validation data". Defaults to None.
        Returns:
            Tuple[callable, Optional[str]]: The reporter (constructor of the report) and optionally the reports unique ID
        """

        def builder(analysis: aNN.ModelAnalysis):
            return DatasetReport(analysis=analysis,
                            data_preprocessor=data_preprocessor,
                            label_preprocessor=label_preprocessor,
                            path=path,
                            modality=modality,
                            data_loader=data_loader,
                            name=name
                                )
        
        if create_id:
            hashed_str = cls.create_unique_id(
                data_preprocessor=data_preprocessor,
                label_preprocessor=label_preprocessor,
                path=path,
                modality=modality,
                name=name
            )

            return builder, hashed_str

        return builder