# README

**This project is under Apache 2.0 License.**

## App Inference Demo

<img src="Screenshots/Prediction_A.jpg" alt="prediction sample" width="50%">

## Train

XGBoostClassifier with Iris dataset

## Convert (xgboost to ort)

- step 1: xgboost model -> sklearn & xgboost & onnx -> .onnx
- step 2: .onnx -> .ort

## Inference (on Android)

OnnxRuntime

# Q&A

## Why do not deploy with XGBoost4j?

The Android architecture is not official supported in XGBoost4j.

## Why do not convert to TensorFlow and deploy with TFLite?

Because it's impossible as TensorFlow do not implement the XGBoost backend. If you transfer onnx to tf model, you would get this error:

```
# Error: "BackendIsNotSupposedToImplementIt: TreeEnsembleClassifier is not implemented."
```
