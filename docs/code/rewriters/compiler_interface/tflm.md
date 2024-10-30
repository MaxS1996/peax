# TensorFlow Lite (for Microcontrollers)

This compiler interface enables PEAX to directly interact with the TensorFlow Lite Converter.

## What is TensorFlow Lite for Microcontrollers

TensorFlow Lite for Microcontrollers is a specialized version of TensorFlow Lite, designed to run machine learning models on very small and low-power devices.
These devices are known as microcontrollers, which are tiny computers often found in household gadgets, wearable technology, and various other electronic products.

## Why Does It Exist?

Microcontrollers are used in many everyday products because they are inexpensive, small, and efficient.
However, their limited computing power makes it challenging to run complex machine learning models directly on them.
TensorFlow Lite for Microcontrollers was created to bridge this gap, enabling these tiny devices to perform machine learning tasks.
This allows manufacturers to build smarter products that can make decisions and predictions without relying on powerful, external servers.

## How It Works

The TensorFlow Lite workflow involves several key stages to prepare and deploy machine learning models on resource-constrained devices such as microcontrollers.

Steps wrapped by the Compiler Interface in PEAX:

* **Model Conversion**: The trained model needs to be converted into a format suitable for deployment on small devices. This is achieved using the TensorFlow Lite Converter. The converter optimizes the model by reducing its size and complexity. One of the key techniques used during this stage is quantization. Quantization involves converting the model's weights and activations from 32-bit floating-point numbers to more compact formats like 8-bit integers. This reduces the model's memory footprint and computational requirements, making it more efficient for execution on microcontrollers.
* **Model Deployment**: After conversion, the TensorFlow Lite model (.tflite file) is deployed onto the microcontroller. For most embedded projects, the tflite file must be converted into a binary representation that is then stored in a set of c and h files. These files need to be added to the project in the IDE of your intended target MCU.

Steps that are not solved by PEAX:

* **Model Deployment (part 2)**: To execute the model on the MCU, the TensorFlow Lite library and headers need to be added to your project. The library includes the TensorFlow Lite interpreter, which is responsible for reading the tflite file and executing the model.
* **Interpreter Initialization**: When the microcontroller starts, the interpreter initializes by loading the model into memory. It uses an OpsResolver to map the model's operations (like convolutional and dense layers) to their corresponding implementations available on the device.
* **Inference Execution**: During runtime, the interpreter takes in input data (such as sensor readings or images) and processes it through the model, step-by-step. For each operation defined in the model, the interpreter consults the OpsResolver to find and execute the appropriate implementation on the microcontroller. This allows the model to make predictions or decisions based on the input data.
* **Output Generation**: Finally, the interpreter produces the output from the model, which can be utilized by the microcontroller to perform specific actions, like triggering an alert, adjusting a device setting, or logging data for further analysis.

::: rewriters.compiler.tflite