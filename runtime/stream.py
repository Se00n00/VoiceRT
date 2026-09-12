"""torch.cuda.Stream wrapper with CPU fallback (stream=None)."""
import torch


class CudaStream:
    """Thin wrapper around torch.cuda.Stream.

    On CPU-only hosts there is no stream object; every method becomes a
    no-op so pipeline code can use streams unconditionally.
    """

    def __init__(self, device=None, priority=0):
        self.device = torch.device(device) if device is not None else None
        self.priority = priority
        self._stream = None
        if torch.cuda.is_available():
            try:
                kwargs = {}
                if self.device is not None:
                    kwargs["device"] = self.device
                self._stream = torch.cuda.Stream(priority=priority, **kwargs)
            except Exception:
                self._stream = None

    @property
    def stream(self):
        """Underlying torch.cuda.Stream or None on CPU."""
        return self._stream

    @property
    def is_cuda(self):
        """True when wrapping a real CUDA stream."""
        return self._stream is not None

    def synchronize(self):
        """Block until work queued on this stream completes."""
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except Exception:
                pass

    def query(self):
        """True when all work on this stream has finished (always True on CPU)."""
        if self._stream is None:
            return True
        try:
            return bool(self._stream.query())
        except Exception:
            return False

    def wait_stream(self, other):
        """Make this stream wait for another CudaStream (no-op on CPU)."""
        if self._stream is None:
            return
        try:
            peer = other.stream if isinstance(other, CudaStream) else other
            if peer is not None:
                self._stream.wait_stream(peer)
        except Exception:
            pass

    def __enter__(self):
        if self._stream is not None:
            try:
                self._ctx = torch.cuda.stream(self._stream)
                self._ctx.__enter__()
            except Exception:
                self._ctx = None
        else:
            self._ctx = None
        return self

    def __exit__(self, exc_type, exc, tb):
        ctx, self._ctx = getattr(self, "_ctx", None), None
        if ctx is not None:
            try:
                ctx.__exit__(exc_type, exc, tb)
            except Exception:
                pass
        return False


def current_stream(device=None):
    """Current CUDA stream, or None on CPU-only hosts."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.current_stream(device)
    except Exception:
        return None


def synchronize_all():
    """Synchronize every CUDA device (no-op on CPU-only hosts)."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
