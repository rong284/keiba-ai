import pandas as pd
from src.utils.progress import tqdm
from src.data.features.feature_engineering import (
    add_horse_history_features,
    add_people_history_features,
    add_race_basic_features,
    add_race_hr_stats,
    add_race_relative_features,
)

def reorder_train_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 目的変数は最後固定
    target = ["rank"]

    # 先頭に置きたい（存在するものだけ使う）
    front = [
        # keys
        "race_id", "horse_id",

        # race info（前に出す）
        "race_date", "place", "race_type", "around", "course_len",
        "dist_bin", "weather", "ground_state", "race_class",
        "race_season", "n_horses",

        # horse day-of-race info
        "wakuban", "umaban",
        "sex", "age", "impost", "weight", "weight_diff",

        # people IDs
        "jockey_id", "trainer_id", "owner_id",

        # market（任意）
        "popularity", "tansho_odds",
    ]
    front = [c for c in front if c in df.columns]

    # hr_ は増えても全部まとめて末尾（targetの直前）へ
    hr_cols = sorted([c for c in df.columns if c.startswith("hr_")])

    used = set(front) | set(hr_cols) | set(target)
    rest = [c for c in df.columns if c not in used]

    ordered = front + rest + hr_cols + [c for c in target if c in df.columns]
    return df.reindex(columns=ordered)



def build_train_table(df_result: pd.DataFrame, df_race_info: pd.DataFrame, df_horse: pd.DataFrame) -> pd.DataFrame:
    """
    主テーブル df_result を軸に、
    df_race_info を race_id で結合して race_date を付与し、
    df_horse の過去戦績特徴量や休み明け特徴量をリーク無しで付与する。

    Returns:
        学習用テーブル（df_resultベース + race_info + hr_rank_mean_3）
    """
    steps = tqdm(total=8, desc="build train table", leave=False)

    # 1) レース情報結合（race_date 付与）
    steps.set_description("join race info")
    base = df_result.merge(
        df_race_info.rename(columns={"date": "race_date"}),
        on="race_id",
        how="left",
        validate="many_to_one"
    )
    base["race_date"] = pd.to_datetime(base["race_date"])
    steps.update(1)

    # 2) レース基本特徴量（条件スイッチ用）
    steps.set_description("race basic features")
    train = add_race_basic_features(base)
    steps.update(1)

    # 3) レース内相対特徴量（レース内の軽重・大小を表現）
    steps.set_description("race relative features")
    train = add_race_relative_features(train)
    steps.update(1)

    # 4) 過去戦績・休み明け特徴量（リーク無し）
    steps.set_description("horse history features")
    train = add_horse_history_features(train, df_horse)
    steps.update(1)

    # 5) レース内 hr 系統計（抜けてる馬や荒れやすさ）
    steps.set_description("race hr stats")
    train = add_race_hr_stats(train)
    steps.update(1)

    # 6) 騎手/調教師/馬主の過去成績（同日除外でリーク防止）
    steps.set_description("people history features")
    train = add_people_history_features(train)
    steps.update(1)

    # 7) 並び安定化
    steps.set_description("sort rows")
    train = train.sort_values(["race_id", "umaban"]).reset_index(drop=True)
    steps.update(1)

    # 8) 列順を整える（分析しやすく）
    steps.set_description("reorder columns")
    train = reorder_train_columns(train)
    steps.update(1)
    steps.close()

    return train
