"""Compute-device selection shared across the inference workers.

Prefers CUDA (NVIDIA), then MPS (Apple Silicon), then CPU. Centralized here so
detector, embedder, inference, and main all resolve to the same device and agree
on whether it's a GPU (for worker-count heuristics and cache eviction)."""

import contextlib
import threading

import torch

# MPS exposes a single Metal command queue that is not safe for concurrent
# submission from multiple threads. This app runs YOLO and CLIP batch workers on
# separate threads (and GPU_WORKERS>1 adds more), so simultaneous submissions
# abort the process with "failed assertion ... MTLCommandBufferStatusCommitted".
# One global lock serializes all GPU work on MPS; CUDA/CPU keep full parallelism.
_mps_lock = threading.Lock()


def pick_device() -> str:
    """Return the best available torch device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    # torch.backends.mps exists only on builds with the MPS backend compiled in.
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def is_gpu(device: str) -> bool:
    """True for accelerator devices (used to scale worker counts up from the CPU default)."""
    return device in ("cuda", "mps")


def empty_cache(device: str) -> None:
    """Release cached device memory back to the driver, when the device supports it."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def submit_lock(device: str):
    """Context manager serializing GPU submissions on MPS (its single Metal command
    queue is not thread-safe). A no-op on CUDA/CPU, which handle concurrent submission
    from multiple worker threads. Hold it around both model loading and inference."""
    if device == "mps":
        return _mps_lock
    return contextlib.nullcontext()
