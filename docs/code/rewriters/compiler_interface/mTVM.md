# MicroTVM (and its ahead-of-time Runtime)

This compiler interface enables PEAX to directly compile models through the tvmc interface of the TVM Deep Learning compiler.

## What is microTVM?

microTVM is an extension of Apache TVM, an open-source machine learning compiler framework, designed to optimize and deploy machine learning models on microcontrollers and other resource-constrained devices.
It aims to provide an efficient and flexible way to run machine learning models on these small, low-power devices by generating highly optimized code specific to the microcontroller's architecture.

## Differences between microTVM and TensorFlow Lite for Microcontrollers

When comparing microTVM and TensorFlow Lite for Microcontrollers, it's essential to understand the distinct execution strategies and their implications on performance, flexibility, and resource utilization.

### Execution Strategy

* TensorFlow Lite for Microcontrollers:
  * Interpreter-Based Execution: TensorFlow Lite for Microcontrollers uses an interpreter to execute machine learning models. After converting a model into a TensorFlow Lite format, the interpreter on the microcontroller reads and executes the model step-by-step during runtime.
  * OpsResolver: This component maps the model’s operations to their implementations on the microcontroller, enabling the interpreter to execute each operation correctly.
  * Flexibility: The interpreter approach allows for dynamic execution and easier updates to models without recompiling the firmware. However, this can introduce some overhead due to the interpretation process.
* microTVM:
  * Ahead-of-Time (AOT) Compilation: microTVM uses an ahead-of-time compilation strategy. Models are compiled into highly optimized, low-level C code tailored to the target microcontroller architecture before deployment.
  * Optimized Code Generation: The compilation process includes various optimizations such as operator fusion, loop unrolling, and hardware-specific tweaks to maximize performance and minimize resource usage.
  * Efficiency: AOT compilation generally results in faster execution and lower memory overhead compared to interpreter-based execution, as the model is directly translated into executable code.

::: rewriters.compiler.micro_tvm