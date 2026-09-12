"""CUDA-graph capture with eager fallback.

CudaGraphRunner wraps a callable fn(*tensor_args) -> tensor(s). When CUDA
graphs are usable it captures one replayable graph (static input buffers
+ static outputs); otherwise it calls fn eagerly. Either way __call__ and
replay() return the outputs, so callers need no branching.
"""
import torch


def graphs_supported():
    """True when this host can plausibly capture a CUDA graph."""
    return bool(torch.cuda.is_available())


class CudaGraphRunner:
    """Capture-or-eager runner for a pure tensor function."""

    def __init__(self, fn, capture_on_cuda=True):
        if not callable(fn):
            raise TypeError(f"CudaGraphRunner expects a callable, got {type(fn)}")
        self.fn = fn
        self.capture_on_cuda = bool(capture_on_cuda)
        self._graph = None
        self._static_inputs = None
        self._static_outputs = None
        self._reason = "not captured yet"

    @property
    def is_captured(self):
        """True after a successful graph capture."""
        return self._graph is not None

    @property
    def mode(self):
        """'graph' when replaying a captured graph, else 'eager'."""
        return "graph" if self.is_captured else "eager"

    @property
    def fallback_reason(self):
        """Why eager mode is used (informational)."""
        return self._reason

    @staticmethod
    def _as_list(outputs):
        return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

    def _tensors_only(self, args):
        return all(torch.is_tensor(a) for a in args)

    def capture(self, *example_inputs):
        """Warm up fn once and capture a CUDA graph.

        Returns True on success. Any failure (no CUDA, non-tensor args,
        capture error) leaves the runner in eager mode and returns False.
        """
        if not self.capture_on_cuda or not graphs_supported():
            self._reason = "cuda unavailable; using eager"
            return False
        if not self._tensors_only(example_inputs):
            self._reason = "non-tensor inputs; using eager"
            return False
        try:
            # Warmup outside capture so lazy init (cudnn/allocator) settles.
            with torch.no_grad():
                warm = self.fn(*example_inputs)
            static_inputs = [t.clone() for t in example_inputs]
            graph = torch.cuda.CUDAGraph()
            static_outputs = None
            with torch.no_grad():
                with torch.cuda.graph(graph):
                    out = self.fn(*static_inputs)
                    static_outputs = self._as_list(out)
            self._graph = graph
            self._static_inputs = static_inputs
            self._static_outputs = static_outputs
            self._reason = "captured"
            return True
        except Exception as exc:  # capture-or-eager: never propagate here
            self._graph = None
            self._static_inputs = None
            self._static_outputs = None
            self._reason = f"capture failed ({exc}); using eager"
            return False

    def replay(self, *args):
        """Run the captured graph, or fn eagerly when not captured."""
        if self._graph is None:
            return self.fn(*args)
        try:
            if len(args) != len(self._static_inputs):
                return self.fn(*args)
            for dst, src in zip(self._static_inputs, args):
                if not torch.is_tensor(src) or src.shape != dst.shape or src.dtype != dst.dtype:
                    return self.fn(*args)
            with torch.no_grad():
                for dst, src in zip(self._static_inputs, args):
                    dst.copy_(src)
                self._graph.replay()
            outs = self._static_outputs
            if len(outs) == 1:
                return outs[0].clone() if torch.is_tensor(outs[0]) else outs[0]
            return [o.clone() if torch.is_tensor(o) else o for o in outs]
        except Exception:
            return self.fn(*args)

    def __call__(self, *args):
        return self.replay(*args)
