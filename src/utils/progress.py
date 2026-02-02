from __future__ import annotations

import sys
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    """
    Unified tqdm wrapper to reduce messy output across nested loops.
    """
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return _tqdm(*args, **kwargs)
