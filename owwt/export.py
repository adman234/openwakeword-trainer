"""Export trained models to ONNX (verified with onnxruntime) and TFLite (converted in a separate TensorFlow venv)."""
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .config import FEATURE_DIM, TFLITE_PYTHON


def export_onnx(model, path: Path) -> float:
    model = model.cpu().eval()
    dummy = torch.zeros(1, model.n_frames, FEATURE_DIM)
    kwargs = dict(input_names=["input"], output_names=["output"], opset_version=13)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(1, model.n_frames, FEATURE_DIM)
    with torch.no_grad():
        expected = model(x).numpy()
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.max(np.abs(got - expected)))
    if diff > 1e-4:
        raise RuntimeError(f"ONNX export does not match PyTorch (max difference {diff})")
    return diff


def save_weights(model, path: Path):
    weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path, n_frames=model.n_frames, n_blocks=len(model.blocks), **weights)


def export_tflite(npz_path: Path, tflite_path: Path) -> dict | None:
    if not TFLITE_PYTHON.exists():
        return None
    script = Path(__file__).with_name("tflite_convert.py")
    result = subprocess.run([str(TFLITE_PYTHON), str(script), str(npz_path), str(tflite_path)],
                            capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(f"TFLite conversion failed: {result.stderr.strip()[-1500:]}")
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1]) if lines else {}
