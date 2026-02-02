from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import lightgbm as lgb


@dataclass(frozen=True)
class OutputDirs:
    out_dir: Path
    fig_dir: Path
    tab_dir: Path
    model_dir: Path

    @staticmethod
    def create(out_dir: Path) -> "OutputDirs":
        fig = out_dir / "figures"
        tab = out_dir / "tables"
        mdl = out_dir / "models"
        fig.mkdir(parents=True, exist_ok=True)
        tab.mkdir(parents=True, exist_ok=True)
        mdl.mkdir(parents=True, exist_ok=True)
        return OutputDirs(out_dir=out_dir, fig_dir=fig, tab_dir=tab, model_dir=mdl)


def save_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_table_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_lgb_model(path: Path, model: lgb.Booster):
    """
    LightGBM Booster のテキストモデル保存。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
