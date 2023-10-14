# README

**This project is under Apache 2.0 License.**

Install `bazelisk`.

## App Inference Demo

<img src="Screenshots/Prediction_A.jpg" alt="prediction sample" width="50%">

## Train

XGBoostClassifier with Iris dataset:

```bash
cd ./xgboost_train
bazelisk run :train
```

## Convert (xgboost to ort)

- step 1: xgboost model -> sklearn & xgboost & onnx -> .onnx
- step 2: .onnx -> .ort

## Inference (on Android)

- OnnxRuntime

# Q&A

## Why do not deploy with XGBoost4j?

The Android architecture is not official supported in XGBoost4j.

## Why do not convert onnx to TensorFlow and deploy with TFLite?

Because it's impossible as TensorFlow do not implement the XGBoost backend. If you convert onnx to tf model, you would get this convertion error:

Convert code:

```python
import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load("xgbc_iris.onnx")

# Convert the ONNX model to TensorFlow format
tf_model = prepare(onnx_model)

# Save the TensorFlow model
tf_model.export_graph("xgbc_iris.tf")
```

Convert result (ERROR):

```plain
Error: "BackendIsNotSupposedToImplementIt: TreeEnsembleClassifier is not implemented."
```
