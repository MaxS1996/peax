# PEAX Framework: Enhancing Latency and Efficiency in Model Architectures

- [PEAX Framework: Enhancing Latency and Efficiency in Model Architectures](#peax-framework-enhancing-latency-and-efficiency-in-model-architectures)
  - [Currently Implemented Rewrites](#currently-implemented-rewrites)
  - [Architecture](#architecture)
  - [TODOs:](#todos)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
    - [Creating the ModelAnalysis object](#creating-the-modelanalysis-object)
    - [Create a Summary](#create-a-summary)
    - [Submit Reporters](#submit-reporters)
    - [Create a Rewriter](#create-a-rewriter)


Our PEAX Framework is a framework designed to optimize the latency and efficiency of your trained model architecture for embedded platforms.
By incorporating various rewrites, PEAX enhances the latency and efficiency of your model, with a focus on adaptive inference techniques such as Early Exit Neural Networks.

What sets PEAX apart from Deep Learning compilers like TVM, Glow, or the TFLite converter is its unique ability to significantly modify the network architecture and perform training steps for finetuning or retraining the altered parts of the model.
This adaptability ensures that your model maintains high prediction performance while significantly improving its computational and latency footprint.

A pioneering feature of PEAX is its automatic application of (adaptive) techniques.
These techniques are capable of substantially reducing the resource footprint of inference without permanently compromising the prediction quality.
This innovation allows PEAX to deliver high-performance models that are both efficient and reliable, making it the ideal choice for a wide range of applications, without need for expert knowledge on how to apply these methods to the model.

## Currently Implemented Rewrites

Our PEAX Framework offers a wide array of implemented rewrites, each designed to enhance the performance and efficiency of your model architecture.
These rewrites include:

* Confidence-based Early Exit Neural Networks for Heterogeneous or Distributed Devices:
This rewrite, also known as the Heterogeneous Platforms rewrite, optimizes performance for your heterogeneous (or distributed) hardware, leveraging the special architecture for efficient inference.
* Model Right-Sizing: 
This feature allows PEAX to adapt your model to the specific requirements of your application and the capabilities of your target device by pruning the deeper layers of your model to find a compromise between prediction quality and the resource footprint, all changed layers are retrained automatically in a way that is supposed to minimize the cost while maximizing the predicition quality.
* Depthwise-separable Convolution Rewrite: By rewriting the most expensive convolutions in your model into depthwise separable version, PEAX significantly reduces the peak computational and memory footprint, enabling you to deploy your model on even smaller target devices.
* Temporal-Decision Early Exit Neural Networks for Sensor Data Monitoring:
This rewrite is specifically designed for sensor data monitoring applications, allowing for more efficient processing by leveraging the correlation that is present within the data for its efficiency.
* IFM-based Temporal Early Termination for Sensor Data Monitoring:
This feature enables the model to exit early based on the Intermediate Feature Map (IFM), reducing the mean computational footprint and latency in sensor data monitoring applications.

These rewrites, along with others, make PEAX a powerful tool for optimizing and enhancing the performance of your model architectures.

## Architecture

The PEAX software architecture is inspired by the principles of compiler design, but has been carefully crafted to promote modularity and facilitate future contributions.
This design allows for easy implementation of new rewrite passes while maximizing code reuse.

At the heart of PEAX lies the ModelAnalysis class, which encapsulates all information and steps related to optimizing a model.
This class contains the submitted model as well as essential information about it, such as:

* Computational, memory, and storage footprints of individual layers
* Estimated task of the model, per output (e.g., binary classification, segmentation, regression, etc.)
* Two graph-level representations of the network architecture in different granularities (called Layer-level and Block-level representations)

This information is always extracted from the model, as it is both inexpensive to acquire and crucial for most rewrite passes.
If additional information is required, custom Report objects can be created and associated with the ModelAnalysis object.
These reports serve as the equivalent of intermediate representations used in traditional compilers and can be reused across runs and between rewriting passes.

Rewrites are implemented through their own Rewriter classes, which leverage the information from ModelAnalysis and Reports to perform the rewrites and generate one or more Solution objects.
These Solution objects contain the modified models, optional additional configuration information, and mappings of subgraphs to processors and functions for evaluating or storing the found solutions.

A key feature of PEAX is its ability to generate HTML-based summaries of ModelAnalysis, Reports, and Rewriters.
These visualizations provide a human-readable overview of the network architecture, layer footprints, extracted information, search spaces, and findings, helping users understand the changes applied to their model and the reasons behind these changes.
This transparency is essential for identifying potential limitations in deploying models to resource-limited embedded devices.

## TODOs:

* Documentation
* Test Coverage
* Simple Examples

## Installation

Currently, PEAX is a poetry-based Python module.
To install it, create a virtual environment for your project, navigate to this directory and install peax as a module using ```pip install .```.
Note, that this might requires an updated pip version as older versions might not yet support poetry-based setups.

If you are using poetry yourself, you should be able to install peax in your venv by running ```poetry add .```.

PEAX has mostly been tested with Python 3.9/3.10 and TensorFlow 2.13.
We are planning to implement more integration tests in the future.

## Basic Usage

PEAX is currently limited to keras models.

### Creating the ModelAnalysis object
Assuming that you already have a trained model loaded, you can create a ModelAnalysis object by running:

```python

from peax.analysis import ModelAnalysis

analysis = ModelAnalysis(model=model)

```

### Create a Summary

To create the HTML-based summary of your analysis object and all reports and rewrites you might have submitted to it, you just need to run

```python

analysis.summary("./summary") # you can either use strings or pathlib.Path objects here

```

The command will create a new folder that is named after the ModelAnalysis (usually the ```model.name``` property, but can be changed by the user).
The folder will contain the HTML files (at least one per report and rewriter and one for the analysis), the original model as keras file and the created solutions in separate subfolders.

### Submit Reporters

A reporter is a closure that is used to create a new report object by the ModelAnalysis.
This ensures that each report configuration is only created once and reused across different passes.
To enable this behaviour, each Report class should implement the closure function and an additional helper class that is called ```Submitter```

One example is the ```DatasetReport``` that is used to introduce training, validation and test sets that are needed for certain rewrite passes into the optimization flow.
These reports can be created using Numpy arrays, TensorFlow Dataset objects or DataLoaders like ```keras.utils.image_dataset_from_directory```.

To submit a TFDataset-based Dataset report to your model analysis, you need to load the TFDataset object and than use the ```Submitter``` syntax:

```python

train_id = (
    dataset.DatasetReport.submit_to(analysis, lazy=True)
    .with_config("training_data", modality=None)
    .from_source(numpy_arrays=(train_images, y_train))
)

```

This is a "lazy" submission, which will only evaluate the report, if it is requested by a rewrite pass, the return value of this function is the unique identfier of the report that would be created, if its closure function is executed in the context of the ModelAnalysis through

```python 

analysis.create_reports() # this creates all reports from submitted reporters that have not yet been executed

anaylsis.access_report(train_id) # returns the report for the given ID, if it already has been created, this version will be returned, otherwise the reporter closure will be executed firstly

```

If you directly want to use the report object that you submitted, you can also just disable the lazy evaluation mode.

```python

train_data_report = (
    dataset.DatasetReport.submit_to(analysis, lazy=False)
    .with_config("training_data", modality=None)
    .from_source(numpy_arrays=(train_images, y_train))
)

```

This will directly evaluate the reporter and return the report object instead of the unique identifer.
However, most of the time, the required reports are submitted by the the rewriter objects and you do not need to submit them manually.

### Create a Rewriter

There are multiple ways to submit a Rewriter to the ModelAnalysis, depending on how much you want to influence the rewrite pass.

If you just want to perform the rewrite and do not want to handtune any parameters, you can directly submit it to the ModelAnalyiss OptimizationQueue.
This will create a closure, that creates the required reports, rewrite object and will execute the functions that are required to create a solution for this rewrite pass.

```python

from peax.rewriters import smart_convolutions as sc

train_ds : tf.data.Dataset
val_ds : tf.data.Dataset
test_ds : tf.data.Dataset

pass_name, pass_func = sc.SmartConvRewriter.create_pass(
    train_data=train_ds, valid_data=val_ds, test_data=test_ds
)

analysis.optimization_queue.add_pass(pass_func, pass_name)

results = analysis.optimization_queue.step(analysis, return_analysis=False)
```