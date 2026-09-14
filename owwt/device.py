"""GPU detection with a real smoke test, so unsupported cards fall back to CPU instead of crashing."""
import shutil
import subprocess


def pick_torch_device() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        return "cpu", "No CUDA GPU visible to the container, using CPU"
    try:
        a = torch.randn(64, 64, device="cuda")
        (a @ a).sum().item()
        torch.nn.functional.layer_norm(a, (64,)).sum().item()
        return "cuda", torch.cuda.get_device_name(0)
    except Exception as exc:  # e.g. "no kernel image is available" on very old or very new cards
        return "cpu", f"GPU found but not usable by this PyTorch build ({str(exc).splitlines()[0]}), using CPU"


def gpu_summary() -> dict:
    """Cheap GPU info for the web UI via nvidia-smi (does not create a CUDA context)."""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "name": None, "detail": "nvidia-smi not found (no GPU passed through)"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        line = out.stdout.strip().splitlines()[0]
        name, mem_total, mem_used, util = [p.strip() for p in line.split(",")]
        return {"available": True, "name": name, "memory_total_mb": int(float(mem_total)),
                "memory_used_mb": int(float(mem_used)), "utilization": int(float(util)), "detail": ""}
    except Exception as exc:
        return {"available": False, "name": None, "detail": f"nvidia-smi failed: {exc}"}
