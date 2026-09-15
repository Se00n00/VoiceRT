"""Collect hardware/software environment for benchmark provenance."""
import platform
import sys
import subprocess
from datetime import datetime, timezone
from typing import Dict

import torch


def _try_import_version(pkg: str) -> str:
    try:
        m = __import__(pkg)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "not_installed"


def _gpu_info() -> Dict[str, str]:
    info: Dict[str, str] = {"gpu_name": "cpu", "gpu_memory_mb": "0", "cuda_version": "none"}
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_mb"] = f"{props.total_memory / 1024**2:.0f}"
        except Exception:
            pass
        info["cuda_version"] = getattr(torch.version, "cuda", "unknown") or "unknown"
        # Prefer nvidia-smi for total memory if available
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0 and r.stdout.strip():
                info["gpu_memory_mb"] = r.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
    else:
        info["cuda_version"] = getattr(torch.version, "cuda", "none") or "none"
    return info


def get_environment(model_name: str = "", dtype: str = "") -> Dict[str, str]:
    gpu = _gpu_info()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": _try_import_version("torch"),
        "triton_version": _try_import_version("triton"),
        "cuda_version": gpu["cuda_version"],
        "gpu_name": gpu["gpu_name"],
        "gpu_memory_mb": gpu["gpu_memory_mb"],
        "model_name": model_name,
        "dtype": dtype,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def add_common_fields(base: Dict, **extra) -> Dict:
    out = dict(base)
    out.update(extra)
    if "timestamp" not in out:
        from datetime import datetime, timezone
        out["timestamp"] = datetime.now(timezone.utc).isoformat()
    return out
