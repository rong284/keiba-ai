import numpy as np
import pandas as pd


def _rolling_mean_by_horse(h: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        h.groupby("horse_id")[col]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _dist_bin(series: pd.Series) -> pd.Series:
    # 距離を粗いビンにまとめる（距離帯の条件差を拾う）
    # bins: <=1400, 1401-1800, 1801-2200, 2201-2600, 2601+
    bins = [-np.inf, 1400, 1800, 2200, 2600, np.inf]
    labels = [0, 1, 2, 3, 4]
    return pd.cut(series, bins=bins, labels=labels).astype("Int64")


def add_race_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # レース日付由来の基礎情報を追加
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")

    # レース内の頭数
    out["n_horses"] = out.groupby("race_id")["horse_id"].transform("count").astype("Int64")
    # 距離帯（短中長の粗い分類）
    out["dist_bin"] = _dist_bin(out["course_len"])
    return out


def _rank_pct(rank: pd.Series, count: pd.Series) -> pd.Series:
    # 順位を0-1に正規化（頭数で補正）
    pct = (rank - 1) / (count - 1)
    return pct.where(count > 1, 0.5)


def _add_race_relative_one(out: pd.DataFrame, col: str, prefix: str, with_desc: bool) -> pd.DataFrame:
    # レース内での平均との差・Z・順位・パーセンタイルを作成
    grp = out.groupby("race_id")[col]
    mean = grp.transform("mean")
    std = grp.transform(lambda s: s.std(ddof=0))

    out[f"{prefix}_diff_mean"] = out[col] - mean
    out[f"{prefix}_z"] = (out[f"{prefix}_diff_mean"] / std.replace(0, np.nan)).fillna(0)

    rank_asc = grp.rank(method="average", ascending=True)
    count = grp.transform("count")
    out[f"{prefix}_rank_asc"] = rank_asc
    out[f"{prefix}_pct_asc"] = _rank_pct(rank_asc, count)

    if with_desc:
        rank_desc = grp.rank(method="average", ascending=False)
        out[f"{prefix}_rank_desc"] = rank_desc
        out[f"{prefix}_pct_desc"] = _rank_pct(rank_desc, count)
    return out


def add_race_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # レース内相対化（斤量・馬体重・増減・年齢）
    if "impost" in out.columns:
        # impost_diff_mean, impost_z, impost_rank_asc, impost_pct_asc
        out = _add_race_relative_one(out, "impost", "impost", with_desc=False)
    if "weight" in out.columns:
        # weight_diff_mean, weight_z, weight_rank_asc/desc, weight_pct_asc/desc
        out = _add_race_relative_one(out, "weight", "weight", with_desc=True)
    if "weight_diff" in out.columns:
        # weight_diff_diff_mean, weight_diff_z, weight_diff_rank_asc/desc, weight_diff_pct_asc/desc
        out = _add_race_relative_one(out, "weight_diff", "weight_diff", with_desc=True)
    if "age" in out.columns:
        # age_diff_mean, age_z, age_rank_asc/desc, age_pct_asc/desc
        out = _add_race_relative_one(out, "age", "age", with_desc=True)

    # 削除指定の特徴量を落とす
    drop_cols = [
        "impost_diff_mean",
        "impost_rank_asc",
        "weight_rank_asc",
        "weight_rank_desc",
        "weight_diff_mean",
        "weight_pct_asc",
        "weight_pct_desc",
        "weight_diff_diff_mean",
        "weight_diff_rank_asc",
        "weight_diff_pct_asc",
        "age_diff_mean",
        "age_rank_asc",
        "age_rank_desc",
        "age_pct_desc",
        "umaban_pct",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out


def _rolling_rate(s: pd.Series, window: int, threshold: int) -> pd.Series:
    # 指定順位以内の割合（直近window走）
    return s.rolling(window=window, min_periods=1).apply(
        lambda x: np.mean(x <= threshold), raw=True
    )


def _rolling_slope(arr: np.ndarray) -> float:
    # 時系列の単回帰傾き（良化/悪化のトレンド）
    n = len(arr)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    y = arr.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _runs_last_n_days(h: pd.DataFrame, days: int) -> pd.Series:
    # 直近N日以内の出走回数（当日を含めない）
    def per_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("date")
        s = pd.Series(1, index=g["date"])
        s_prev = s.shift(1)
        counts = s_prev.rolling(f"{days}D").sum()
        return pd.Series(counts.to_numpy(), index=g.index)
    return h.groupby("horse_id", group_keys=False).apply(per_group)


def add_horse_history_features(df_base: pd.DataFrame, df_horse: pd.DataFrame) -> pd.DataFrame:
    base = df_base.copy()
    h = df_horse.copy()

    # 日付型を揃える
    base["race_date"] = pd.to_datetime(base["race_date"], errors="coerce")
    h["date"] = pd.to_datetime(h["date"], errors="coerce")

    # 距離帯を馬履歴側にも追加
    if "course_len" in h.columns:
        h["dist_bin"] = _dist_bin(h["course_len"])

    # 過去のみで集計するため、当日成績をshiftで除外
    # ties on the same date should be stable for reproducibility
    h = h.sort_values(["horse_id", "date"], kind="mergesort")
    h["rank_prev"] = h.groupby("horse_id")["rank"].shift(1)
    h["rank_prev2"] = h.groupby("horse_id")["rank"].shift(2)
    h["prize_prev"] = h.groupby("horse_id")["prize"].shift(1)
    h["time_sec_prev"] = h.groupby("horse_id")["time_sec"].shift(1)
    h["up3f_prev"] = h.groupby("horse_id")["up3f"].shift(1)
    h["passing_first_prev"] = h.groupby("horse_id")["passing_first"].shift(1)
    h["passing_avg_prev"] = h.groupby("horse_id")["passing_avg"].shift(1)
    h["passing_last_prev"] = h.groupby("horse_id")["passing_last"].shift(1)

    # 直近平均着順
    for w in [3, 5]:
        # hr_rank_mean_3 / hr_rank_mean_5
        h[f"hr_rank_mean_{w}"] = _rolling_mean_by_horse(h, "rank_prev", window=w)
    # 通算（過去のみ）
    # hr_rank_mean_all
    h["hr_rank_mean_all"] = (
        h.groupby("horse_id")["rank_prev"]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 上位率・ベスト着順
    # hr_top3_rate_5
    h["hr_top3_rate_5"] = (
        h.groupby("horse_id")["rank_prev"]
        .rolling(window=5, min_periods=1)
        .apply(lambda x: np.mean(x <= 3), raw=True)
        .reset_index(level=0, drop=True)
    )
    # hr_rank_last, hr_rank_prev2
    h["hr_rank_last"] = h["rank_prev"]
    h["hr_rank_prev2"] = h["rank_prev2"]

    # 直近n走賞金（合計）と通算賞金（過去のみ）
    h["hr_prize_sum_5"] = (
        h.groupby("horse_id")["prize_prev"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    # hr_prize_sum_total
    h["hr_prize_sum_total"] = (
        h.groupby("horse_id")["prize_prev"]
        .cumsum()
    )

    # 直近の走破タイム・上り・通過・ペース
    for w in [3, 5]:
        # hr_time_sec_mean_3/5
        h[f"hr_time_sec_mean_{w}"] = (
            h.groupby("horse_id")["time_sec_prev"]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # hr_up3f_mean_3/5
        h[f"hr_up3f_mean_{w}"] = (
            h.groupby("horse_id")["up3f_prev"]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # hr_passing_avg_mean_3/5
        h[f"hr_passing_avg_mean_{w}"] = (
            h.groupby("horse_id")["passing_avg_prev"]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # hr_passing_first_mean_3/5
        h[f"hr_passing_first_mean_{w}"] = (
            h.groupby("horse_id")["passing_first_prev"]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # hr_passing_last_mean_3/5
        h[f"hr_passing_last_mean_{w}"] = (
            h.groupby("horse_id")["passing_last_prev"]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # hr_time_sec_last, hr_up3f_last, hr_passing_first_last, hr_passing_avg_last, hr_passing_last_last
    h["hr_time_sec_last"] = h["time_sec_prev"]
    h["hr_up3f_last"] = h["up3f_prev"]
    h["hr_passing_first_last"] = h["passing_first_prev"]
    h["hr_passing_avg_last"] = h["passing_avg_prev"]
    h["hr_passing_last_last"] = h["passing_last_prev"]

    # トレンド（傾き）と前走-2走前差
    h["hr_rank_slope_5"] = (
        h.groupby("horse_id")["rank_prev"]
        .rolling(window=5, min_periods=2)
        .apply(_rolling_slope, raw=True)
        .reset_index(level=0, drop=True)
    )

    # 条件別フォーム（芝/ダ、競馬場、距離帯、馬場状態）
    # condition-specific rolling mean (window=3)
    cond_map = {
        "race_type": "same_surface",
        "place": "same_place",
        "dist_bin": "same_dist_bin",
        "ground_state": "same_ground_state",
    }
    for cond_col, suffix in cond_map.items():
        if cond_col in h.columns:
            # hr_rank_mean_3_same_*（条件別の直近3走平均）
            h[f"hr_rank_mean_3_{suffix}"] = (
                h.groupby(["horse_id", cond_col])["rank_prev"]
                .rolling(window=3, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )

    # 休み明け系（出走間隔、休み明けフラグ、直近出走数）
    # rest / interval features
    h["prev_date"] = h.groupby("horse_id")["date"].shift(1)
    # days_since_last
    h["days_since_last"] = (h["date"] - h["prev_date"]).dt.days
    # runs_last_90d
    h["runs_last_90d"] = _runs_last_n_days(h, 90).astype("Int64")

    # 付与に使う列をまとめて結合
    feat_cols = [c for c in h.columns if c.startswith("hr_")] + [
        "days_since_last",
        "runs_last_90d",
    ]
    feat = h[["horse_id", "date"] + feat_cols].rename(columns={"date": "race_date"})

    # 既に同名の特徴がある場合は衝突を避ける（_x/_y を作らない）
    conflict = set(base.columns) & set(feat.columns) - {"horse_id", "race_date"}
    if conflict:
        base = base.drop(columns=sorted(conflict))

    out = base.merge(
        feat,
        on=["horse_id", "race_date"],
        how="left",
        validate="many_to_one",
    )
    return out


def add_race_hr_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # hr_rank_mean_3 が無ければ何もしない
    if "hr_rank_mean_3" not in out.columns:
        return out

    # レース内の相対順位・パーセンタイル・Z
    grp = out.groupby("race_id")["hr_rank_mean_3"]
    hr_rank = grp.rank(method="average", ascending=True)
    count = grp.transform("count")
    # hr_rank_mean_3_pct_asc
    out["hr_rank_mean_3_pct_asc"] = _rank_pct(hr_rank, count)

    std = grp.transform(lambda s: s.std(ddof=0))
    mean = grp.transform("mean")
    # hr_rank_mean_3_z
    out["hr_rank_mean_3_z"] = ((out["hr_rank_mean_3"] - mean) / std.replace(0, np.nan)).fillna(0)

    def race_stats(s: pd.Series) -> pd.Series:
        # レース単位の“荒れやすさ”指標
        vals = s.dropna().to_numpy()
        if vals.size == 0:
            return pd.Series(
                {"race_hr3_std": np.nan, "race_hr3_top1_minus_top2": np.nan,
                 "race_hr3_top1_minus_med": np.nan, "race_hr3_top3_mean": np.nan}
            )
        vals = np.sort(vals)
        top1 = vals[0]
        top2 = vals[1] if vals.size > 1 else np.nan
        med = np.median(vals)
        top3_mean = np.mean(vals[:3]) if vals.size >= 3 else np.nan
        return pd.Series(
            {
                "race_hr3_std": float(np.std(vals)),
                "race_hr3_top1_minus_top2": float(top2 - top1) if vals.size > 1 else np.nan,
                "race_hr3_top1_minus_med": float(med - top1),
                "race_hr3_top3_mean": float(top3_mean) if vals.size >= 3 else np.nan,
            }
        )

    stats = out.groupby("race_id")["hr_rank_mean_3"].apply(race_stats)
    stats = stats.reset_index()
    # merge 衝突回避（pandasの挙動差で同名列が混入する場合がある）
    overlap = set(stats.columns) & set(out.columns) - {"race_id"}
    if overlap:
        stats = stats.drop(columns=sorted(overlap))
    # pandas のバージョンによって level_1 が入ることがあるため除去
    if "level_1" in stats.columns:
        stats = stats.drop(columns=["level_1"])
    # まれに重複キーが出るケースを吸収（merge validate のため一意化）
    stats = stats.groupby("race_id", as_index=False).first()
    out = out.merge(stats, on="race_id", how="left", validate="many_to_one")
    return out


def add_people_history_features(df: pd.DataFrame) -> pd.DataFrame:
    # people history features are disabled (jockey/trainer/owner)
    return df.copy()

    base = df.copy()
    base["race_date"] = pd.to_datetime(base["race_date"], errors="coerce")

    for col, prefix in [("jockey_id", "jockey"), ("trainer_id", "trainer"), ("owner_id", "owner")]:
        if col not in base.columns:
            continue

        # 当日集計を含めないため、日付ごとに集約してから累積を1日シフト
        tmp = base[[col, "race_date", "rank"]].dropna(subset=[col, "race_date", "rank"]).copy()
        tmp["is_win"] = (tmp["rank"] == 1).astype(int)
        tmp["is_top3"] = (tmp["rank"] <= 3).astype(int)

        agg = (
            tmp.groupby([col, "race_date"])
            .agg(wins=("is_win", "sum"), top3=("is_top3", "sum"), runs=("rank", "count"))
            .reset_index()
        )
        agg = agg.sort_values([col, "race_date"])
        agg["wins_cum"] = agg.groupby(col)["wins"].cumsum().shift(1).fillna(0)
        agg["top3_cum"] = agg.groupby(col)["top3"].cumsum().shift(1).fillna(0)
        agg["runs_cum"] = agg.groupby(col)["runs"].cumsum().shift(1).fillna(0)

        # 平滑化で母数の少ないIDの暴れを抑える
        # Laplace smoothing to stabilize low-sample IDs
        agg[f"{prefix}_win_rate_smooth"] = (agg["wins_cum"] + 1) / (agg["runs_cum"] + 2)
        agg[f"{prefix}_top3_rate_smooth"] = (agg["top3_cum"] + 1) / (agg["runs_cum"] + 2)
        agg[f"{prefix}_runs_total"] = agg["runs_cum"]

        feat = agg[
            [col, "race_date", f"{prefix}_win_rate_smooth", f"{prefix}_top3_rate_smooth", f"{prefix}_runs_total"]
        ]
        base = base.merge(
            feat,
            on=[col, "race_date"],
            how="left",
            validate="many_to_one",
        )
    return base
