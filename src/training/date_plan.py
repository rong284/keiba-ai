from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DatePlan:
    mode: str
    train_end: str
    test_start: Optional[str]
    test_end: Optional[str]
    eval_enabled: bool
    prod_train_end: Optional[str]

    def as_dict(self) -> Dict:
        return asdict(self)


def resolve_date_plan(cfg: Dict) -> DatePlan:
    """
    日付分割のプランを後方互換込みで解決する。
    - eval mode: train_end まで学習、test_start/test_end を holdout に使う
    - production mode: train_end を prod_train_end に切替、評価は任意
    """
    dp = cfg.get("date_plan", {})
    mode = str(dp.get("mode", "eval"))

    train_end = str(dp.get("train_end", cfg.get("train_end")))
    test_start = dp.get("test_start", cfg.get("test_start"))
    test_end = dp.get("test_end")
    prod_train_end = dp.get("prod_train_end")
    eval_enabled = bool(dp.get("eval_enabled", True))

    if mode == "production":
        # 本番モードでは prod_train_end があればそちらへ切り替える
        if prod_train_end:
            train_end = str(prod_train_end)
        if not eval_enabled:
            test_start = None
            test_end = None

    return DatePlan(
        mode=mode,
        train_end=train_end,
        test_start=str(test_start) if test_start else None,
        test_end=str(test_end) if test_end else None,
        eval_enabled=eval_enabled,
        prod_train_end=str(prod_train_end) if prod_train_end else None,
    )
