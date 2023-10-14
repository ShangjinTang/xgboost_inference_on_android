#!/usr/bin/env python
"""
one line to give the program's name and a brief description
Copyright 2023 yourname

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import argparse
import logging
import os
import sys

import numpy
import onnxruntime
from onnxmltools.convert.xgboost.operator_converters.XGBoost import \
    convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import \
    calculate_linear_classifier_output_shapes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

__author__ = "Shangjin Tang"
__email__ = "shangjin.tang@gmail.com"
__copyright__ = "Copyright 2023 Shangjin Tang"
__license__ = "APACHE 2.0"
__version__ = "1.0.0"

LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
LOGGING_DATEFMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    datefmt=LOGGING_DATEFMT,
)


def train_xgboost_model(X_train, y_train, n_estimators):
    # create model instance
    xgbc_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=1,
        objective="binary:logistic",
    )
    xgbc_model.fit(X_train, y_train)
    return xgbc_model


def convert_xgboost_to_onnx(model_xgboost):
    pipe = Pipeline([("xgb", model_xgboost)])

    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    model_onnx = convert_sklearn(
        pipe,
        "pipeline_xgboost",
        [("input", FloatTensorType([None, 4]))],
        target_opset={"": 12, "ai.onnx.ml": 4},
    )

    return model_onnx


def save_xgboost_model_to_file(model_xgboost, output_file):
    model_xgboost.save_model(output_file)


def save_onnx_model_to_file(model_onnx, output_file):
    with open(output_file, "wb") as f:
        f.write(model_onnx.SerializeToString())


def main(args):
    iris_data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris_data["data"], iris_data["target"], test_size=0.2, random_state=42
    )

    model_xgboost = train_xgboost_model(X_train, y_train, n_estimators=2)
    assert model_xgboost
    save_xgboost_model_to_file(model_xgboost, args.output_xgboost)

    model_onnx = convert_xgboost_to_onnx(model_xgboost)
    assert model_onnx
    save_onnx_model_to_file(model_onnx, args.output_onnx)

    # make predictions for
    y_pred_xgboost = model_xgboost.predict(X_test)
    # make predictions for
    sess = onnxruntime.InferenceSession(args.output_onnx)
    y_pred_onnx = sess.run(None, {"input": X_test.astype(numpy.float32)})[0]

    logging.info(f"Test Inputs:\n{X_test[:]}\n")
    logging.info(f"Test Outputs:\n{y_test[:]}\n")
    logging.info(f"Test predictions by XGBoost Model:\n{y_pred_xgboost[:]}\n")
    logging.info(f"Test predictions by OnnxRuntime Model:\n{y_pred_onnx[:]}\n")

    if (y_pred_xgboost == y_pred_onnx).all():
        logging.info("OK: onnx inference result is identical to xgboost\n")
    else:
        logging.error(
            "Error: onnx inference result is not same as xgboost, please check the code\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_xgboost",
        default="xgbc_iris.model",
        type=str,
    )
    parser.add_argument(
        "--output_onnx",
        default="xgbc_iris.onnx",
        type=str,
    )

    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
