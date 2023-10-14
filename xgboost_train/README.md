# xgboost_train

## How to create bzlmod

1. write a `train.py` or convert from `train.ipynb`:

   ```bash
   # note: output will auto add `.py` extension
   jupyter nbconvert train.ipynb --to script --output train
   ```

   note: if you have run commands in `train.ipynb`, you need to remove them in generated `train.py`:

   ```bash
   get_ipython().system("pip3 install onnx")
   ```

   Otherwise will raise `NameError: name 'get_ipython' is not defined.`

2. write `requirements.in` and generate `requirements_lock.txt`:

   ```bash
   pip-compile requirements.in -o requirements_lock.txt
   ```

3. write bazel bzlmod deps and rules:

   see: `MODULE.bazel` & `BUILD.bazel`

   ```starlark
   py_binary(
       name = "train",
       # ...
   )
   ```

4. run the rule with bazelisk:

   - `.bazelrc` (enable bzlmod)
   - `.bazeliskrc` (specify the version)

   ```bash
   bazelisk run :train
   ```
