# setup/setup_torch_cuda.py
#!/usr/bin/env python3
"""
setup_torch_cuda.py

Installs a PyTorch build that is compatible with the detected NVIDIA GPU.

Why this exists:
- Some clusters misreport "CUDA Version" in `nvidia-smi` (e.g., "13.0").
- More importantly: older GPUs (e.g., GTX 1080 Ti, sm_61) need PyTorch wheels that
  still include kernels for that compute capability.

This script detects the GPU name and uses a safe mapping.

Compatible with Python 3.8+.

Usage:
  python setup/setup_torch_cuda.py
"""

import subprocess
import sys
from typing import Optional


def run(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()


def detect_gpu_name() -> Optional[str]:
    try:
        # Works even on multi-GPU nodes; returns one name per line
        out = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if not out:
            return None
        # Take the first GPU name (good enough for wheel selection)
        return out.splitlines()[0].strip()
    except Exception:
        return None


def pip_install(args):
    cmd = [sys.executable, "-m", "pip"] + args
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def pip_uninstall():
    # Clean slate to avoid mixing wheels
    pip_install(["uninstall", "-y", "torch", "torchvision", "torchaudio", "triton"])


def install_cpu():
    print("[INFO] Installing CPU-only PyTorch.")
    pip_install(["install", "--no-cache-dir", "torch", "torchvision", "torchaudio"])


def install_cu118_pascal_safe():
    """
    Pascal GPUs (GTX 10xx, sm_61) are not supported by the newest PyTorch wheels.
    A commonly compatible choice is CUDA 11.8 wheels with torch 2.0/2.1 era.

    We pin versions for stability and reproducibility.
    """
    print("[INFO] Pascal GPU detected -> installing torch 2.1.2 + cu118 (sm_61 compatible choice).")
    pip_install([
        "install", "--no-cache-dir",
        "torch==2.1.2+cu118",
        "torchvision==0.16.2+cu118",
        "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cu118",
    ])


def install_cu121_modern():
    """
    Modern default for many newer GPUs.
    """
    print("[INFO] Installing PyTorch from cu121 wheels.")
    pip_install([
        "install", "--no-cache-dir",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])


def verify():
    import torch
    print("[OK] torch version:", torch.__version__)
    print("[OK] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[OK] gpu:", torch.cuda.get_device_name(0))
        # quick kernel test
        x = torch.randn(512, 512, device="cuda")
        y = x @ x
        print("[OK] CUDA matmul ran, mean:", float(y.mean().item()))


def main():
    gpu_name = detect_gpu_name()
    print("[INFO] Detected GPU:", gpu_name if gpu_name else "None")

    # Always uninstall first to avoid mixing incompatible wheels
    try:
        pip_uninstall()
    except Exception:
        pass

    if gpu_name is None:
        install_cpu()
        verify()
        return

    # Pascal GPUs: GTX 10xx line includes 1080 Ti (sm_61)
    # We match by name because it’s robust on clusters.
    pascal_markers = ["GTX 10", "1080", "1070", "1060", "1050", "TITAN Xp", "TITAN X"]
    if any(m in gpu_name for m in pascal_markers):
        install_cu118_pascal_safe()
    else:
        # Default for newer GPUs
        install_cu121_modern()

    print("[INFO] Verifying installation...")
    verify()


if __name__ == "__main__":
    main()