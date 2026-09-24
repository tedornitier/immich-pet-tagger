"""Stub out heavy ML packages so tests run without the Docker environment."""
import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = None
    return mod


def _stub_torch() -> None:
    torch = _make_stub("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, Stream=None, empty_cache=lambda: None)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.mps = types.SimpleNamespace(empty_cache=lambda: None)
    torch.no_grad = lambda: __import__("contextlib").nullcontext()

    class _FakeTensor:
        def to(self, *a, **kw): return self
        def norm(self, *a, **kw): return self
        def __truediv__(self, o): return self
        def cpu(self): return self
        def numpy(self): return None

    torch.Tensor = _FakeTensor
    torch.stack = lambda tensors, **kw: _FakeTensor()
    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = types.ModuleType("torch.cuda")


def _stub_open_clip() -> None:
    oc = _make_stub("open_clip")
    oc.create_model_and_transforms = lambda *a, **kw: (None, lambda x: x, None)
    sys.modules["open_clip"] = oc


def _stub_ultralytics() -> None:
    ul = _make_stub("ultralytics")

    class _FakeYOLO:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return []

    ul.YOLO = _FakeYOLO
    sys.modules["ultralytics"] = ul


_stub_torch()
_stub_open_clip()
_stub_ultralytics()
