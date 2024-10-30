# Reports

Reports are a way for developers to additional information that is required for rewrite operations to the ModelAnalysis object.
The idea of this modular design is to reduce the information that needs to be acquired to create the ModelAnalysis object to a minimum, which is used often and cheap to collect.
Reports can be used to collect more rewrite-specific information, which can also be more costly to gather.

Another benefit of this modular structure is that different rewriters and other reports can share the already acquired information within an report, reducing the amount of redundant operations needed during the model augmentation.

## Usage

Reports are not supposed to be created through their constructors, instead, a closure that wraps their constructor function will be submitted to the relevant ModelAnalysis object.
This is often done using the ```submit_to``` syntax that consists of the static function as part of the implementation of the Report classes and the ReportSubmitter class, which acts as a factory for the construction closures that will be submitted to the ModelAnalysis object.
This syntax is recommended and acts as syntactic sugar throughout most of PEAX.

The ModelAnalysis object stores the report closures (sometimes also called reporters) and already initialized reports in separate registries.
Each report and reporter is identified by a unique ID, which depends on its configuration parameters.
This removes the need for developers to check, if a report or reporter of the same config has already been created, instead all of this redundancy-avoidance logic is handled by the ModelAnalysis class.

## Implementation

If you want to implement a novel report, you should inhert from the baseReport class for the report class and implement all required functions, additionally you need to ensure that the ```submit_to``` function has been implemented correctly and that the creation of the parameter-based unique IDs works reliably across runs.

## Currently available Reports

A number of reports have already been implemented:

* **AccuracyReport**: Evaluates the prediction quality of the submitted Model on the datasets that have been passed to the AccuracyReport object.
* **BatchSizeReport**: This is a meta report class, which will not show up in the created summary files. It tries to identify the appropriate batch sizes for inference and training on the computer you are currently using.This information is not relevant for the performance on the target device and just used by other reports and rewrites to improve the speed and performance of the rewrite processes.
* **DatasetReport**: A wrapper for datasets, which will be used by other reports for training or evaluation. Can be created from Numpy arrays, TensorFlow dataset objects and more.
* **EarlyExitReport**: Identifies the available nodes within the graph, where an early exit branch can be inserted into the model, additionally, it extracts an appropriate blueprint from the submitted model, which will be used to create potential early exit branches at the identified locations. The options are evaluated in a lazy fashion, only training and evaluating early exit branches when requested. If a branch is evaluated, it will store its subgraph as Keras model as well as its cost (MAC ops/inference) and prediction quality (accuracy, loss).
* **HistogramReport**: Evaluates the behavior of IFMs over time (i.e. a sensor data stream). Specific evaluation for temporal early exit and histogram-based early termination rewrites.
* **HWCheckReport**: Utilizes (user-provided) performance models to estimate the latency, memory and storage footprint of the individual layer workloads on the target device. Requires the submission of a hardware description using the simple classes in the peax.hardware submodule.
* **TemporalReport**: Evaluates the behavior of Early Exits over time (i.e. a sensor data stream). Specific evaluation for temporal early exit and histogram-based early termination rewrites. Creates an additional EarlyExitReport object within the ModelAnalysis when submitted and evaluated.