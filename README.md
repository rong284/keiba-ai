# keiba-ai

日本の競馬データを対象に、時系列リークを避けて学習・評価・回収率シミュレーションまで行うためのKeiba AIプロジェクト。

このリポジトリでやりたいこと:
- 生のレース/馬/結果データからクリーンな学習テーブルを作成
- 二値（勝ち/複勝）・順位（LambdaRank）モデルを時系列CVで学習
- ホールドアウト年で評価し、払戻表を使った回収率シミュレーションを実行

---

## 1. ディレクトリ構成（3レイヤ）

大きく「データ収集」「学習」「評価」に分けています。

```
src/
  data_collection/   # データ収集・前処理・特徴量付与
    scraping/        # スクレイピング（HTML保存）
    loaders/         # CSV読み込み + 前処理
    preprocess/      # raw dfの基本クリーニング
    features/        # 履歴・相対特徴量
    pipelines/       # 学習テーブル生成
    common/          # マッピング/定数

  training/          # 学習・CV・モデル・指標
    models/          # LightGBM binary / rank
    tuning/          # Optuna チューニング
    data.py          # data_collectionの読み込みまとめ
    split.py         # 時系列分割
    features.py      # 学習で使う特徴量の確定
    metrics.py       # 評価指標
    reporting.py     # 図・レポート生成

  evaluation/        # 評価・バックテスト
    backtest_rules.py

notebooks/           # 検証用（特徴量/品質チェック中心）
```

---

## 2. セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m playwright install --with-deps chromium
```

---

## 3. データ収集（スクレイピング → CSV変換）

1) レースデータのスクレイピング
```bash
python src/data_collection/scraping/scrape_race_data.py --year 2024 --concurrency 1 --skip
```

2) HTMLをCSVに変換
```bash
python src/data_collection/scraping/parse_race_html_to_csv.py --year 2024
python src/data_collection/scraping/parse_race_info_html_to_csv.py --year 2024
python src/data_collection/scraping/parse_horse_html_to_csv.py --year 2024
python src/data_collection/scraping/parse_return_html_to_csv.py --year 2025
```

> 学習テーブルの生成は `src/training/data.py` 経由で自動的に行います。

---

## 4. Optunaチューニング

```bash
python -m src.training.tuning.tune_binary_optuna \
  --config src/configs/train_binary.json \
  --out_dir outputs/03_train/binary/tables \
  --target win --n_trials 50

python -m src.training.tuning.tune_binary_optuna \
  --config src/configs/train_binary.json \
  --out_dir outputs/03_train/binary/tables \
  --target place --n_trials 50

python -m src.training.tuning.tune_rank_optuna \
  --config src/configs/train_rank.json \
  --out_dir outputs/03_train/rank/tables \
  --n_trials 50
```

---

## 5. 学習（Binary / Rank）

### 5-1. 二値（win/place）学習
```bash
python -m src.training.run_train_binary \
  --config_path src/configs/train_binary.json \
  --out_dir outputs/03_train/binary
```

### 5-2. 順位（LambdaRank）学習
```bash
python -m src.training.run_train_rank \
  --config_path src/configs/train_rank.json \
  --out_dir outputs/03_train/rank
```

---

## 6. 評価 / 回収率シミュレーション

```bash
python -m src.evaluation.backtest_rules \
  --config src/configs/train_binary.json \
  --out_dir outputs/04_backtest_rules \
  --return_path data/rawdf/return/return_2025.csv
```

予測済みCSVから単純な回収率検証だけを行う場合:
```bash
python -m src.evaluation.simulate_returns \
  --return_path data/rawdf/return/return_2025.csv \
  --out_dir outputs/04_returns/2025
```
---

## 7. 設定のポイント（date_plan）

`src/configs/train_*.json` の `date_plan` で評価モードを制御します。

```json
"date_plan": {
  "mode": "eval",
  "train_end": "2024-12-31",
  "test_start": "2025-01-01",
  "test_end": "2025-12-31",
  "prod_train_end": "2025-12-31",
  "eval_enabled": true
}
```

- `eval`: train_end まで学習 → test期間で評価
- `production`: prod_train_end まで学習（評価は任意）

---

## 8. 出力先（主な成果物）

- `outputs/03_train/*/tables` : CV・ホールドアウト・予測テーブル
- `outputs/03_train/*/models` : LightGBMモデル
- `outputs/03_train/*/figures`: 学習・評価図
- `outputs/04_backtest_rules` : 回収率シミュレーション結果
- `outputs/04_returns/*`      : 予測CSVベースの簡易回収率

---

## 9. ノートブック

`notebooks/` は検証用（特徴量の中身確認や品質チェック用）だけを残しています。
学習や評価の本処理はすべて `src/` 配下のCLIから実行してください。

---

## 10. データ入力（想定）

主な入力:
- `data/rawdf/result/result_*.csv`
- `data/rawdf/horse/*.csv`
- `data/rawdf/race_info/*.csv`
- `data/rawdf/return/return_2025.csv`（回収率計算用）

---

## 11. 特徴量（全体像）

特徴量は `training/features.py` で定義したルールに従い、
実際に使われた列は学習出力の `feature_spec.json` に保存されます。

```
outputs/03_train/*/tables/feature_spec.json
```

注意:
- 実際の `feature_cols` は入力データにより決まり、存在しない列は自動スキップされます。
- `use_market=false` の場合、市場特徴量（オッズ/人気）は除外されます。

### 基本列（存在する場合）
- `race_date`, `place`, `race_type`, `around`, `course_len`, `dist_bin`, `weather`,
  `ground_state`, `race_class`, `race_season`, `n_horses`
- `wakuban`, `umaban`, `sex`, `age`, `impost`, `weight`, `weight_diff`
- `jockey_id`, `trainer_id`, `owner_id`
- `popularity`, `tansho_odds`（市場特徴、任意）

### レース内相対特徴（`add_race_relative_features`）
- `impost_z`, `impost_pct_asc`
- `weight_diff_mean`, `weight_z`, `weight_pct_asc`, `weight_pct_desc`
- `weight_diff_z`, `weight_diff_rank_desc`
- `age_z`, `age_pct_asc`

### 馬の履歴特徴（接頭辞 `hr_`）
- 順位平均: `hr_rank_mean_3`, `hr_rank_mean_5`, `hr_rank_mean_all`
- 複勝率: `hr_top3_rate_5`
- 直近順位: `hr_rank_last`, `hr_rank_prev2`
- 賞金合計: `hr_prize_sum_5`, `hr_prize_sum_total`
- タイム/ペース平均: `hr_time_sec_mean_3`, `hr_time_sec_mean_5`
- 上がり3F平均: `hr_up3f_mean_3`, `hr_up3f_mean_5`
- 通過平均:
  - `hr_passing_avg_mean_3`, `hr_passing_avg_mean_5`
  - `hr_passing_first_mean_3`, `hr_passing_first_mean_5`
  - `hr_passing_last_mean_3`, `hr_passing_last_mean_5`
- 直近レース値:
  - `hr_time_sec_last`, `hr_up3f_last`
  - `hr_passing_first_last`, `hr_passing_avg_last`, `hr_passing_last_last`
- トレンド: `hr_rank_slope_5`
- 条件別フォーム（履歴テーブルに列がある場合）:
  - `hr_rank_mean_3_same_surface`
  - `hr_rank_mean_3_same_place`
  - `hr_rank_mean_3_same_dist_bin`
  - `hr_rank_mean_3_same_ground_state`

### 休養・間隔特徴
- `days_since_last`
- `runs_last_90d`

### レースレベルの履歴統計（`hr_rank_mean_3`が存在する場合）
- `hr_rank_mean_3_pct_asc`
- `hr_rank_mean_3_z`
- `race_hr3_std`
- `race_hr3_top1_minus_top2`
- `race_hr3_top1_minus_med`
- `race_hr3_top3_mean`

---

## 12. よく使うフロー（まとめ）

1. scrape/parseでrawデータを用意
2. `run_train_binary` / `run_train_rank` で学習
3. `backtest_rules` で回収率評価
