"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import contextlib
import functools
import os
from threading import Thread
import time

from .logging import logger


def run_in_thread(daemon=False):
    """Run a function/method in a thread."""
    def wrap(f):
        @functools.wraps(f)
        def threaded_f(*args, **kwargs):
            t = Thread(target=f, daemon=daemon, args=args, kwargs=kwargs)
            t.start()
            return t
        return threaded_f
    return wrap


def profiler(info, *, process_time=False):
    def wrap(f):
        @functools.wraps(f)
        def timed_f(*args, **kwargs):
            if process_time:
                timer = time.process_time
            else:
                timer = time.perf_counter

            t0 = timer()
            result = f(*args, **kwargs)
            dt_ms = 1000 * (timer() - t0)
            logger.info("Time spent on %s: %.3f ms", info, dt_ms)
            return result
        return timed_f
    return wrap


@contextlib.contextmanager
def changed_cwd(dirname):
    oldcwd = os.getcwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(oldcwd)
