# /home/shared/3_12_jupyter/bin/baselines/logger.py
import logging
import sys
from typing import Any, Iterable, Optional

_logger = logging.getLogger("baselines")
if not _logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    h.setFormatter(fmt)
    _logger.addHandler(h)
    _logger.setLevel(logging.INFO)

# Minimal API used by old baselines code
def configure(dir: Optional[str] = None, format_strs: Optional[Iterable[str]] = None, **kwargs):  # no-op
    pass

def set_level(level: int):
    _logger.setLevel(level)

def debug(msg: str, *args, **kwargs):
    _logger.debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    _logger.info(msg, *args, **kwargs)

def warn(msg: str, *args, **kwargs):
    _logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    _logger.error(msg, *args, **kwargs)

# Key-value logging stubs
_kv = {}
def logkv(key: str, val: Any):
    _kv[key] = val

def logkv_mean(key: str, val: Any):
    _kv[key] = val

def dumpkvs():
    if _kv:
        _logger.info("metrics: %s", _kv)
        _kv.clear()
