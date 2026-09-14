"""Build a TFLite model from exported WakeNet weights. Runs inside the separate TensorFlow venv.

The network is rebuilt from raw TensorFlow ops rather than converted from ONNX, which avoids the
onnx-tf toolchain entirely. Output is checked against a numpy reference before the file is kept.

Usage: python tflite_convert.py weights.npz out.tflite
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

EPS = 1e-5  # torch.nn.LayerNorm default


def reference_forward(w, x):
    def dense(h, name):
        return h @ w[name + ".weight"].T + w[name + ".bias"]

    def layer_norm(h, name):
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        return (h - mean) / np.sqrt(var + EPS) * w[name + ".weight"] + w[name + ".bias"]

    h = x.reshape(x.shape[0], -1)
    h = np.maximum(layer_norm(dense(h, "layer1"), "layernorm1"), 0)
    for i in range(int(w["n_blocks"])):
        h = np.maximum(layer_norm(dense(h, f"blocks.{i}.fcn_layer"), f"blocks.{i}.layer_norm"), 0)
    return 1 / (1 + np.exp(-dense(h, "last_layer")))


def main(npz_path, out_path):
    w = dict(np.load(npz_path))
    n_frames = int(w["n_frames"])
    n_blocks = int(w["n_blocks"])
    const = {k: tf.constant(v.astype(np.float32)) for k, v in w.items() if v.ndim > 0}

    def dense(h, name):
        return tf.matmul(h, const[name + ".weight"], transpose_b=True) + const[name + ".bias"]

    def layer_norm(h, name):
        mean, var = tf.nn.moments(h, axes=[-1], keepdims=True)
        return (h - mean) / tf.sqrt(var + EPS) * const[name + ".weight"] + const[name + ".bias"]

    class Net(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, n_frames, 96], tf.float32, name="input")])
        def __call__(self, x):
            h = tf.reshape(x, [1, n_frames * 96])
            h = tf.nn.relu(layer_norm(dense(h, "layer1"), "layernorm1"))
            for i in range(n_blocks):
                h = tf.nn.relu(layer_norm(dense(h, f"blocks.{i}.fcn_layer"), f"blocks.{i}.layer_norm"))
            return tf.sigmoid(dense(h, "last_layer"))

    net = Net()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([net.__call__.get_concrete_function()], net)
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(8):
        x = rng.normal(0, 1, (1, n_frames, 96)).astype(np.float32)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        max_diff = max(max_diff, float(np.max(np.abs(interpreter.get_tensor(out["index"]) - reference_forward(w, x)))))
    if max_diff > 1e-4:
        raise SystemExit(f"TFLite output does not match reference (max difference {max_diff})")

    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(json.dumps({"max_abs_diff": max_diff, "input_shape": [int(s) for s in inp["shape"]], "bytes": len(tflite_model)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
