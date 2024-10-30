# Analysis

The creation of the ModelAnalysis object from your neural network is the first step in the PEAX Model Rewrite Flow.
It extracts the most basic information from your model, which will be used for later stages that instrument and modify the model.

The ModelAnalysis is the main wrapper for this.
It contains the model as member and contains the basic information about your model, including:

- **the network graph**:
    - as layer-level graph
    - as block-level graph
- **the estimated input data modality** (i.e. image data, ...)
- **the estimated model task** (i.e. classification, regression, ...)
- **architectural information**
    - what kind of network is it? FFNN, RNN or CNN?
    - which inputs and outputs are contained within the model?
- **compute information**
    - the MAC operations per layer
    - the total inference MAC operations for the entire network
- **memory information**
    - the size of each IFM tensor
    - the total IFM footprint
- **storage information**
    - the storage footprint of the weight tensors per layer
    - the total storage footprint of all weights

The Architecture, Compute, Memory and Storage information is wrapped in dedicated classes that inhert from PartialAnalysis.
All Analysis classes are implementations of the abstract BaseAnalysis class.

In addition to the information about the analyzed model, the ModelAnalysis class also contains components to manage the rewrite and optimization process for the model.
This includes the information about the submitted reporters and already created reports as well as the registered rewrite flows.

You can create the basic ModelAnalysis from your Keras model by running:

```Python
from peax import analysis as aNN
from tensorflow import keras as keras

keras_model = keras.models.load_model("model.h5")

model_analysis = aNN.ModelAnalysis(
  model=keras_model,
  name="PEAX-Test"
)
```

To create the HTML-based human-readable report for your Model, including the analysis results, reports and rewirte efforts, you must run:

```Python
model_analysis.summary("./summary")
```

This will create a new folder structure based on the model and analysis name as well as on the ModelAnalysis creation time and create one or more files for ModelAnalysis, each report and rewriter.
Created Solutions will also be stored within this directory strucuture.
