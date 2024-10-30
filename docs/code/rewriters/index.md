# Rewriters

Rewriters are the main component when it comes to modifying the submitted model.
They utilize the information collected by the ModelAnalysis and additional reporters to perform their decisions and operations.

## Currently Implemented Rewriters

PEAX currently supports a few different rewrite flows:

* **Heterogeneous Platforms**: This flow converts the submitted model into an early exit neural network and distributes the subgraphs across the processing targets of the intended target device. The order, in which these targets should be utilized, needs to be described by the developer. The model architecture and the confidence-based at-runtime decision mechanism will be automatically configured by the rewriter.
* **Temporal Early Exit Neural Networks**: Inserts a single early exit branch at the optimal location into the model architecture and configures a temporal decision mechanism for its use. This requires the application to process temporally correlated datastreams with a feed-forward architecture.
* **Histogram-based Termination**: Inserts a branch that monitors an intermediate feature map through a pooling layer and configures an at-runtime decision mechanism that observes its change over time to enable early termination of the inference, if no relevant change happens on the input side. The difference of this and the Temporal Early Exiting is that the pooling-based monitoring branch does not have the ability to create predictions on its own and should be cheaper to execute (depending on the target hardware).
* **Right Sizing**: Byproduct of the early exit functionality, can prune deeper layer from the submitted model and creates a new classifier at the deepest remaining feature extraction/hidden layer.
* **Smart Convolution**: Replaces the most costly CONV2D layer with a depthwise-separable version, if possible, and retrains the modified subgraph of the model while keeping the remaining weights frozen for the majority of the training.
* **Static Model Slimming**: Creates multiple versions of the submitted model at different width scalings.
Keeps their training cost low by transfering weights from the already trained instances and the original model.

## Implementing a new Rewriter

To implement your own rewriter, you need to inherit from the base.Rewriter and base.Solution classes.
As for the Reports, both, your Rewriter class should also have the ability to create a unique identifier based on its parameters to enable the submission of multiple rewriters of the same type with different configurations.

A Solutions class needs to be implemented for each Rewriter, this class is supposed to wrap the outputs of the rewriters and add functionality like serialization, finetuning (if supported) and the ability to convert the solution to a new ModelAnalysis object, which is important to operate the RewriterQueue that can be defined.
The solutions that are created by CompilerInterfaces (which are specialized rewriters) do not support the conversion into ModelAnalysis objects as they consist of toolchain-specific representations of the models that are different from the default objects required by the underlying deep learning framework.