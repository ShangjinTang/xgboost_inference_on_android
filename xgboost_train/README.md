# xgboost_train

Just run the `.ipynb` file.

## How to use bzlmod to automatic the process

1. write a `train.py` or convert from `train.ipynb`:

   ```bash
   # note: output will auto add `.py` extension
   jupyter nbconvert train.ipynb --to script --output train
   ```

   Then edit the `train.py` file.

   Note: if you have run commands in `train.ipynb`, you need to remove the lines `get_ipython()...` in generated `train.py`. Otherwise will raise `NameError: name 'get_ipython' is not defined`.

2. write `requirements.in` and generate `requirements_lock.txt`:

   requirements.in is the dependencies (`import`s in you python scripts)

   ```plain
   numpy~=1.25.1
   onnxmltools~=1.11.2
   onnxruntime~=1.16.1
   scikit-learn~=1.3.0
   skl2onnx~=1.15.0
   xgboost~=2.0.0
   ```

   ```bash
   pip-compile requirements.in -o requirements_lock.txt
   ```

3. write bazel bzlmod configurations:

   - `MODULE.bazel` (dependencies)
   - `BUILD.bazel` (compile targets)

   ```
   py_binary(
    name = "train",
    srcs = ["train.py"],
    main = "train.py",
    deps = [
        requirement("skl2onnx"),
        requirement("scikit-learn"),
        requirement("xgboost"),
        requirement("numpy"),
        requirement("onnxruntime"),
        requirement("onnxmltools"),
    ],
   )
   ```

4. run the rule with bazelisk:

   - `.bazelrc` (enable bzlmod)
   - `.bazeliskrc` (specify the version)

   ```bash
   bazelisk run :train
   ```

   result:

   ```plain
    INFO: Build completed successfully, 5 total actions
    INFO: Running command line: bazel-bin/train

    2023-10-14 16:26:43 [INFO] train.py:115 - Test Inputs:
    [[6.1 2.8 4.7 1.2]
    [5.7 3.8 1.7 0.3]
    [7.7 2.6 6.9 2.3]
    [6.  2.9 4.5 1.5]
    [6.8 2.8 4.8 1.4]
    [5.4 3.4 1.5 0.4]
    [5.6 2.9 3.6 1.3]
    [6.9 3.1 5.1 2.3]
    [6.2 2.2 4.5 1.5]
    [5.8 2.7 3.9 1.2]
    [6.5 3.2 5.1 2. ]
    [4.8 3.  1.4 0.1]
    [5.5 3.5 1.3 0.2]
    [4.9 3.1 1.5 0.1]
    [5.1 3.8 1.5 0.3]
    [6.3 3.3 4.7 1.6]
    [6.5 3.  5.8 2.2]
    [5.6 2.5 3.9 1.1]
    [5.7 2.8 4.5 1.3]
    [6.4 2.8 5.6 2.2]
    [4.7 3.2 1.6 0.2]
    [6.1 3.  4.9 1.8]
    [5.  3.4 1.6 0.4]
    [6.4 2.8 5.6 2.1]
    [7.9 3.8 6.4 2. ]
    [6.7 3.  5.2 2.3]
    [6.7 2.5 5.8 1.8]
    [6.8 3.2 5.9 2.3]
    [4.8 3.  1.4 0.3]
    [4.8 3.1 1.6 0.2]]

    2023-10-14 16:26:43 [INFO] train.py:116 - Test Outputs:
    [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]

    2023-10-14 16:26:43 [INFO] train.py:117 - Test predictions by XGBoost Model:
    [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]

    2023-10-14 16:26:43 [INFO] train.py:118 - Test predictions by OnnxRuntime Model:
    [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]

    2023-10-14 16:26:43 [INFO] train.py:121 - OK: onnx inference result is identical to xgboost
   ```
