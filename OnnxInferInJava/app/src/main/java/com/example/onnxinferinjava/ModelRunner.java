package com.example.onnxinferinjava;

import android.util.Log;

import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OnnxTensor;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

public class ModelRunner {
    private static final String TAG = "ModelRunner";
    private static final String MODEL_FILE_NAME = "model.ort";
    private final int INPUT_SIZE;
    private final int OUTPUT_SIZE;
    private static final String OUTPUT_NAME = "output";
    private static final String INPUT_NAME = "input";

    private final OrtEnvironment ortEnvironment;
    private final OrtSession ortSession;

    public ModelRunner(InputStream modelInputStream, int input_size, int output_size) throws Exception {
        INPUT_SIZE = input_size;
        OUTPUT_SIZE = output_size;
        ortEnvironment = OrtEnvironment.getEnvironment();
        SessionOptions options = new SessionOptions();
        byte[] modelData = modelInputStream.readAllBytes();
        ortSession = ortEnvironment.createSession(modelData, options);
        Log.i(TAG, "Model loaded");
    }

    public long runInference(float[] inputData) throws Exception {
        // Get the name of the input node
        String inputName = ortSession.getInputNames().iterator().next();
        // Make a FloatBuffer of the inputs
        FloatBuffer floatBufferInputs = FloatBuffer.wrap(inputData);
        // Create input tensor with floatBufferInputs of shape (1, 1)
        OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBufferInputs, new long[]{1, INPUT_SIZE});
        // Run the model
        Map<String, OnnxTensor> inputMap = Collections.singletonMap(inputName, inputTensor);
        Result results = ortSession.run(inputMap);
        OnnxValue onnxValue = results.get(0);
        Object value = onnxValue.getValue();
        long[] output = (long[]) value;
        Log.i(TAG, "Inference result: " + output[0]);
        return output[0];
    }

    public void close() {
        // 释放 OrtSession 和 OrtEnvironment 对象
        try {
            ortSession.close();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        ortEnvironment.close();

        Log.i(TAG, "Model closed");
    }
}