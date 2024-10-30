# Compiler Interfaces

Compiler Interfaces are specialized Rewriters that can only be put at the end of optimization/rewrite queues, as their solutions cannot be converted back to ModelAnalysis objects.
Compiler Interfaces convert the submitted model into the representation used by the interfaced deep learning compiler.
Currently, two interfaces are supported:

* **TensorFlow Lite (for Microcontrollers)**: This interface converts the submitted model using the TFLite converter, you can configure the quantization settings and other typical parameters, but the sensible values are set as defaults. The interface can also directly create the .c and .h files required to include the model into an embedded project (you still need to provide your own tflite library with the appropriate target-specific math kernels, often provided by device vendors.)
* **microTVM**: The microTVM interface relies on the TensorFlow Lite interface to quantize the model, it is configured to target the Ahead-of-Time executor with the C-library to enable developers to leverage the low overhead of this execution strategy, which is benefitial for embedded scenarios.