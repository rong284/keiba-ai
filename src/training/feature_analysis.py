from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import lightgbm as lgb

from src.training.artifacts import save_table_csv


def build_importance_table(model: lgb.Booster) -> pd.DataFrame:
    names = list(model.feature_name())
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    df = pd.DataFrame({
        "feature": names,
        "importance_gain": gain.astype(float),
        "importance_split": split.astype(int),
    })
    df["gain_rank"] = df["importance_gain"].rank(ascending=False, method="min").astype(int)
    df["split_rank"] = df["importance_split"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["importance_gain", "importance_split"], ascending=False).reset_index(drop=True)


def _top_value(s: pd.Series) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    vc = s.value_counts(dropna=True)
    if vc.empty:
        return None, None, None
    top_val = vc.index[0]
    top_cnt = int(vc.iloc[0])
    top_ratio = float(vc.iloc[0] / vc.sum()) if vc.sum() > 0 else None
    return str(top_val), top_ratio, top_cnt


def build_feature_summary(
    df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            rows.append({
                "feature": col,
                "dtype": None,
                "missing_rate": None,
                "nunique": None,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
                "top_value": None,
                "top_ratio": None,
                "top_count": None,
                "is_categorical": col in cat_cols,
                "present_in_data": False,
            })
            continue

        s = df[col]
        missing_rate = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        is_cat = col in cat_cols or dtype == "category" or dtype == "object"

        row = {
            "feature": col,
            "dtype": dtype,
            "missing_rate": missing_rate,
            "nunique": nunique,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "top_value": None,
            "top_ratio": None,
            "top_count": None,
            "is_categorical": is_cat,
            "present_in_data": True,
        }

        if is_cat:
            top_val, top_ratio, top_cnt = _top_value(s)
            row.update({
                "top_value": top_val,
                "top_ratio": top_ratio,
                "top_count": top_cnt,
            })
        else:
            s_num = pd.to_numeric(s, errors="coerce")
            row.update({
                "mean": float(s_num.mean()) if s_num.notna().any() else None,
                "std": float(s_num.std()) if s_num.notna().any() else None,
                "min": float(s_num.min()) if s_num.notna().any() else None,
                "max": float(s_num.max()) if s_num.notna().any() else None,
                "median": float(s_num.median()) if s_num.notna().any() else None,
            })

        rows.append(row)
    return pd.DataFrame(rows)


def save_feature_importance(
    out_dir: Path,
    tag: str,
    model: lgb.Booster,
    feature_summary: Optional[pd.DataFrame] = None,
    plot_top_n: int = 30,
):
    imp = build_importance_table(model)
    save_table_csv(out_dir / f"feature_importance_{tag}.csv", imp)

    if feature_summary is not None and not feature_summary.empty:
        merged = imp.merge(feature_summary, on="feature", how="left")
        save_table_csv(out_dir / f"feature_analysis_{tag}.csv", merged)

    _plot_top_importance(out_dir / f"feature_importance_{tag}_top{plot_top_n}.png", imp, top_n=plot_top_n)


def _plot_top_importance(out_path: Path, df: pd.DataFrame, top_n: int = 30):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    d = df.sort_values("importance_gain", ascending=False).head(top_n)
    if d.empty:
        return

    fig = plt.figure(figsize=(8.4, 7.2))
    ax = fig.add_subplot(111)
    ax.barh(d["feature"][::-1], d["importance_gain"][::-1], color="#1F7A8C")
    ax.set_xlabel("importance_gain")
    ax.set_title(f"Top {top_n} feature importance (gain)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
