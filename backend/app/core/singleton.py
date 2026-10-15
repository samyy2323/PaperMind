import functools
import threading


def once(fn):
    """Cache a loader's result per argument tuple, and make concurrent first calls wait for a single load.

    Model loaders are called from several threads (thread-mode pipeline, Celery thread pools); without the
    lock two threads would download/initialise the same weights at once and corrupt the shared HF cache.
    """
    cached = functools.lru_cache(maxsize=None)(fn)
    lock = threading.Lock()

    @functools.wraps(fn)
    def wrapper(*args):
        with lock:
            return cached(*args)

    wrapper.cache_clear = cached.cache_clear
    return wrapper
