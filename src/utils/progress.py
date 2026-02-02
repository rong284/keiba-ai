from __future__ import annotations

import sys
from typing import Any

from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    """
    ネストした進捗表示でも崩れにくいようにまとめたラッパー。
    """
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return _tqdm(*args, **kwargs)


def log(message: str, **kwargs: Any) -> None:
    """
    tqdmの進捗バーを崩さずにログを出す。
    """
    _tqdm.write(message, **kwargs)
