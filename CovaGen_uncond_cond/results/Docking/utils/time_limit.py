"""
From: https://stackoverflow.com/a/601168/16407115

Usage:
```python
try:
    with time_limit(10):
        long_function_call()
except TimeoutException as e:
    print("Timed out!")
```
"""

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
