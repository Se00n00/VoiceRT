"""Micro-batcher: group queued requests by leg, then split into batches."""
import collections
import threading


def group_by_leg(items, key="leg"):
    """Group request dicts/objects by their leg label.

    `key` is a dict key (for dicts) or attribute name (for objects).
    Items without the key land in the "" group instead of raising.
    """
    groups = collections.defaultdict(list)
    for item in items:
        if isinstance(item, dict):
            leg = item.get(key, "")
        else:
            leg = getattr(item, key, "")
        groups[leg or ""].append(item)
    return dict(groups)


def split_batches(items, max_batch_size):
    """Split a list into consecutive chunks of at most max_batch_size."""
    max_batch_size = int(max_batch_size)
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")
    return [list(items[i:i + max_batch_size]) for i in range(0, len(items), max_batch_size)]


def batch_by_leg(items, max_batch_size=4, key="leg"):
    """Group by leg, then split each group: {leg: [[batch], ...]}."""
    return {
        leg: split_batches(group, max_batch_size)
        for leg, group in group_by_leg(items, key).items()
    }


class MicroBatcher:
    """Thread-safe buffer that accumulates requests and flushes grouped batches."""

    def __init__(self, max_batch_size=4, key="leg"):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.max_batch_size = int(max_batch_size)
        self.key = key
        self._lock = threading.Lock()
        self._buffer = []

    def add(self, item):
        """Enqueue one request; returns the current buffer depth."""
        with self._lock:
            self._buffer.append(item)
            return len(self._buffer)

    def extend(self, items):
        """Enqueue several requests; returns the current buffer depth."""
        with self._lock:
            self._buffer.extend(items)
            return len(self._buffer)

    def flush(self):
        """Drain the buffer into {leg: [[batch], ...]} and return it."""
        with self._lock:
            items = self._buffer
            self._buffer = []
        return batch_by_leg(items, self.max_batch_size, self.key)

    def peek_groups(self):
        """Group buffered items without draining (for observability)."""
        with self._lock:
            items = list(self._buffer)
        return group_by_leg(items, self.key)

    def clear(self):
        """Drop buffered requests; returns how many were dropped."""
        with self._lock:
            n = len(self._buffer)
            self._buffer = []
            return n

    def __len__(self):
        with self._lock:
            return len(self._buffer)
