import pandas as pd
from src.data.features.horse_history import add_last3_mean_rank

def reorder_train_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 目的変数は最後固定
    target = ["rank"]

    # 先頭に置きたい（存在するものだけ使う）
    front = [
        # keys
        "race_id", "horse_id",

        # race info（前に出す）
        "race_date", "place", "race_type", "around", "course_len",
        "weather", "ground_state", "race_class",

        # horse day-of-race info
        "wakuban", "umaban", "sex", "age", "impost", "weight", "weight_diff",

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
    df_horse の過去戦績特徴量（直近3走平均着順）をリーク無しで付与する。

    Returns:
        学習用テーブル（df_resultベース + race_info + hr_rank_mean_3）
    """
    # 1) レース情報結合（race_date 付与）
    base = df_result.merge(
        df_race_info.rename(columns={"date": "race_date"}),
        on="race_id",
        how="left",
        validate="many_to_one"
    )
    base["race_date"] = pd.to_datetime(base["race_date"])

    # 2) 過去戦績特徴量（直近3走平均着順）を付与
    train = add_last3_mean_rank(base, df_horse)

    # 3) 並び安定化
    train = train.sort_values(["race_id", "umaban"]).reset_index(drop=True)

    # 4) 列順を整える（分析しやすく）
    train = reorder_train_columns(train)

    return train
