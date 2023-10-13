#!/usr/bin/env python

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data["data"], data["target"], test_size=0.2, random_state=42
    )
    # create model instance
    xgbc_model = XGBClassifier(
        n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic"
    )
    # fit model
    xgbc_model.fit(X_train, y_train)
    # make predictions
    y_pred = xgbc_model.predict(X_test)

    xgbc_model.save_model("xgbc_iris.model")

    print(f"X_test[:5]:\n{X_test[:5]}\n")
    print(f"y_test[:5]:\n{y_test[:5]}\n")
    print(f"y_pred[:5]:\n{y_pred[:5]}\n")


if __name__ == "__main__":
    main()


# In[84]:

# # !pip3 install onnx
# # !pip3 install onnxconverter_common
# # !pip3 install onnxmltools
# # !pip3 install onnxruntime
# # !pip3 install pyquickhelper
# get_ipython().system("pip3 install docutils")
# get_ipython().system("pip3 install mlprodict")


# # In[ ]:


# # In[87]:


# import numpy
# import onnxruntime as rt
# from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
# from onnxmltools.convert.xgboost.operator_converters.XGBoost import \
#     convert_xgboost
# from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
# from skl2onnx.common.data_types import FloatTensorType
# from skl2onnx.common.shape_calculator import (
#     calculate_linear_classifier_output_shapes,
#     calculate_linear_regressor_output_shapes)
# from sklearn.datasets import load_diabetes, load_iris, make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from xgboost import DMatrix, XGBClassifier, XGBRegressor
# from xgboost import train as train_xgb

# xgbc = XGBClassifier()
# xgbc.load_model("xgbc_iris.model")

# pipe = Pipeline([("xgb", xgb_from_model)])

# pipe.fit(X_test, y_test)


# # In[89]:


# update_registered_converter(
#     XGBClassifier,
#     "XGBoostXGBClassifier",
#     calculate_linear_classifier_output_shapes,
#     convert_xgboost,
#     options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
# )

# model_onnx = convert_sklearn(
#     pipe,
#     "pipeline_xgboost",
#     [("input", FloatTensorType([None, 4]))],
#     target_opset={"": 12, "ai.onnx.ml": 4},
# )

# # And save.
# with open("xgbc_iris.onnx", "wb") as f:
#     f.write(model_onnx.SerializeToString())

# sess = rt.InferenceSession("xgbc_iris.onnx")
# pred_onx = sess.run(None, {"input": X_test[:5].astype(numpy.float32)})
# print("predict", pred_onx[0])
# print("predict_proba", pred_onx[1][:1])
