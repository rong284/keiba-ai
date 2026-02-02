# keiba-ai

Keiba AI project for time-series-safe training, evaluation, and return simulation on Japanese horse racing data.

This repository focuses on:
- Building a clean training table from raw race/horse/result data
- Training binary (win/place) and rank (LambdaRank) models without leakage
- Selecting best parameters and best iteration via CV
- Evaluating on a holdout year (default: 2025)
- Simulating return with payout tables for practical operation design

---

## Project goals
- Predict race outcomes with time-aware features
- Keep strict separation between training and evaluation periods
- Make evaluation reproducible and explainable
- Support practical betting simulation (ROI) on holdout data
- Be flexible enough to move from evaluation mode to production mode

---

## Data and directory structure

Main inputs:
- `data/rawdf/result/result_*.csv`
- `data/rawdf/horse/*.csv`
- `data/rawdf/race_info/*.csv`
- `data/rawdf/return/return_2025.csv` (payout table for ROI simulation)

Main outputs:
- `outputs/03_train/*/tables` (CV, holdout, predictions)
- `outputs/03_train/*/models` (LightGBM model files)
- `outputs/03_train/*/figures` (plots)
- `outputs/04_returns/2025` (return simulation CSV/plots)

---

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m playwright install --with-deps chromium
```

---

## Data pipeline (high-level)

1) Scrape race data
```bash
python src/data/scraping/scrape_race_data.py --year 2024 --concurrency 1 --skip
```

2) Parse HTML to CSV/TSV
```bash
python src/data/scraping/parse_race_html_to_tsv.py --year 2024
python src/data/scraping/parse_return_html_to_csv.py --year 2025
```

3) Build training table (internally)
- `src/data/pipelines/build_train_table.py`
- This merges race info, results, and horse history features safely (no leakage)

---

## Feature overview (what the model sees)

The training table is built per (race_id, horse_id) row. Major feature groups:
- Race info: place, race_type, around, course_len, dist_bin, weather, ground_state, race_class
- Race meta: race_date, race_season, n_horses
- Horse day-of-race: wakuban, umaban, sex, age, impost, weight, weight_diff
- People IDs: jockey_id, trainer_id, owner_id
- Market (optional): popularity, tansho_odds (controlled by config `use_market`)
- History features (prefix: `hr_*`): horse performance stats, race-level stats, and people history stats

Feature selection is controlled by:
- `src/train/features.py`
- The feature list is saved as `feature_spec.json` in each training run

### Full feature list (by definition)

Note:
- The exact `feature_cols` is determined by the input data and saved in
  `outputs/03_train/*/tables/feature_spec*.json`.
- Columns missing in the raw data are automatically skipped.
- Market features are dropped when `use_market=false`.

Base columns (if present):
- `race_date`, `place`, `race_type`, `around`, `course_len`, `dist_bin`, `weather`,
  `ground_state`, `race_class`, `race_season`, `n_horses`
- `wakuban`, `umaban`, `sex`, `age`, `impost`, `weight`, `weight_diff`
- `jockey_id`, `trainer_id`, `owner_id`
- `popularity`, `tansho_odds` (market features, optional)

Race-relative features (from `add_race_relative_features`):
- `impost_z`, `impost_pct_asc`
- `weight_diff_mean`, `weight_z`, `weight_pct_asc`, `weight_pct_desc`
- `weight_diff_z`, `weight_diff_rank_desc`
- `age_z`, `age_pct_asc`

Horse history features (prefix `hr_`):
- Rank means: `hr_rank_mean_3`, `hr_rank_mean_5`, `hr_rank_mean_all`
- Top rate: `hr_top3_rate_5`
- Last ranks: `hr_rank_last`, `hr_rank_prev2`
- Prize sums: `hr_prize_sum_5`, `hr_prize_sum_total`
- Time/pace means: `hr_time_sec_mean_3`, `hr_time_sec_mean_5`
- Up3f means: `hr_up3f_mean_3`, `hr_up3f_mean_5`
- Passing means:
  - `hr_passing_avg_mean_3`, `hr_passing_avg_mean_5`
  - `hr_passing_first_mean_3`, `hr_passing_first_mean_5`
  - `hr_passing_last_mean_3`, `hr_passing_last_mean_5`
- Last race values:
  - `hr_time_sec_last`, `hr_up3f_last`
  - `hr_passing_first_last`, `hr_passing_avg_last`, `hr_passing_last_last`
- Trend: `hr_rank_slope_5`
- Condition-specific form (if columns exist in history table):
  - `hr_rank_mean_3_same_surface`
  - `hr_rank_mean_3_same_place`
  - `hr_rank_mean_3_same_dist_bin`
  - `hr_rank_mean_3_same_ground_state`

Rest/interval features:
- `days_since_last`
- `runs_last_90d`

Race-level hr stats (if `hr_rank_mean_3` exists):
- `hr_rank_mean_3_pct_asc`
- `hr_rank_mean_3_z`
- `race_hr3_std`
- `race_hr3_top1_minus_top2`
- `race_hr3_top1_minus_med`
- `race_hr3_top3_mean`

---

## Training logic (binary and rank)

### Common design
- Time-series CV (2021-2024 by default)
- `best_iter` is determined from CV (median over folds)
- Final model is trained using `best_iter` on all training data
- Holdout evaluation is done on 2025 (if evaluation mode is enabled)

### Binary model (win/place)
- Targets:
  - `win` -> y_top1
  - `place` -> y_top3
- Training: LightGBM binary objective
- Metrics: hit@1, hit@3, hit@5, top1_pos_rate

### Rank model (LambdaRank)
- Objective: `lambdarank`
- Metrics: mrr, ndcg@3, ndcg@5, hit@k
- Uses race-level ordering and group consistency

---

## Date splitting logic

Date splits are controlled by `date_plan` in the config.

Example (evaluation mode):
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

Modes:
- `eval`: train on <= train_end, evaluate on test_start to test_end
- `production`: train on prod_train_end, optionally skip evaluation

This allows the workflow:
1) Evaluate on 2025
2) If good, switch to production mode and train on 2025 as well

---

## Hyperparameter tuning (Optuna)

Binary and rank use Optuna to find better LightGBM parameters:

```bash
python -m src.train.tune_binary_optuna --config src/configs/train_binary.json --out_dir outputs/03_train/binary/tables --target win --n_trials 50
python -m src.train.tune_binary_optuna --config src/configs/train_binary.json --out_dir outputs/03_train/binary/tables --target place --n_trials 50
python -m src.train.tune_rank_optuna  --config src/configs/train_rank.json  --out_dir outputs/03_train/rank/tables  --n_trials 50
```

Outputs:
- `optuna_best_params_*.json`
- `optuna_trials_*.csv`

---

## Training & evaluation

Binary:
```bash
python -m src.train.run_train_binary --config_path src/configs/train_binary.json --out_dir outputs/03_train/binary
```

Rank:
```bash
python -m src.train.run_train_rank --config_path src/configs/train_rank.json --out_dir outputs/03_train/rank
```

Key outputs:
- CV tables and summaries in `outputs/03_train/*/tables`
- Models in `outputs/03_train/*/models`
- Figures in `outputs/03_train/*/figures`
- Reports in `outputs/03_train/*/report_*.md`
- Predictions for 2025 evaluation (for ROI simulation)

---

## Return simulation (ROI)

We simulate betting returns on 2025 using payout table:
- `data/rawdf/return/return_2025.csv`

Run:
```bash
python -m src.train.simulate_returns \
  --return_path data/rawdf/return/return_2025.csv \
  --out_dir outputs/04_returns/2025 \
  --bet_type 単勝
```

Options:
- `--bet_type 単勝` or `--bet_type 複勝`
- `--thresholds 0.05,0.1,0.15,0.2`
- `--ensemble_weights win:1,place:1,rank:1`

Outputs:
- `outputs/04_returns/2025/return_simulation.csv`
- `outputs/04_returns/2025/return_roi_vs_threshold.png`
- `outputs/04_returns/2025/return_report.md`

---

## What is currently supported
- Win/Place return simulation based on horse selection
- Top1 strategy (one bet per race)
- Threshold strategy (bet all horses above threshold)
- Simple ensemble by normalized average of win/place/rank

---

## Return simulation design for combo bets (planned)

Goal: support bet types beyond win/place using the same prediction outputs.

Common inputs:
- Predictions per horse: `score` (win/place/rank or ensemble)
- Return table row: `bet_type`, `combination`, `payout`
- Combination parsing:
  - `-` separated for unordered combos (e.g., "1-5")
  - `→` separated for ordered combos (e.g., "1→5")

Planned bet types and mapping:
- 馬連 (umaren): choose top-N horses, evaluate unordered pairs
- ワイド (wide): choose top-N horses, evaluate unordered pairs, payout if any pair hits
- 馬単 (umatan): choose ordered top-N (e.g., top1→top2, top1→top3)
- 三連複 (sanrenpuku): choose top-N horses, evaluate unordered triples
- 三連単 (sanrentan): choose ordered triples (e.g., top1→top2→top3)

Strategy controls (per bet type):
- `topN`: use top N by score per race
- `threshold`: use horses above score threshold
- `max_tickets`: cap number of combinations per race
- `stake`: default 100 yen, configurable

Outputs to compare:
- ROI
- hit rate
- avg tickets per race
- max drawdown (optional)

---

## ROI evaluation templates (operation candidates)

Use these as templates for experimenting on 2025:

1) Top1 single (単勝/複勝)
- Pick the top1 horse per race
- Metric: ROI and hit rate

2) Threshold sweep (単勝/複勝)
- Bet all horses with score >= threshold
- Sweep thresholds and pick stable region

3) Top-K single (単勝/複勝)
- Bet top K horses per race (K=2,3)
- Limit tickets per race

4) Ensemble top1
- Normalize win/place/rank per race and take average
- Bet top1 from the ensemble

5) Ensemble + threshold
- Same ensemble, but threshold-based betting
- Focus on ROI stability vs. ticket count

6) Rank-based combo (planned)
- Use rank score for ordering and generate combos
- Compare umaren/wide vs. umatan/sanrentan

Suggested reporting for each:
- ROI, total return, total bet
- Number of bets, bets per race
- Monthly ROI trend (optional)

---

## Next steps / future direction

Short-term:
- Add combo-bet simulations: umaren, wide, sanrenpuku, sanrentan
- Add strategy templates: EV-based, top-k, odds-adjusted filters
- Compare model ensembles with weight sweep and visualization

Mid-term:
- Build a report dashboard (HTML or notebook) for quick review
- Add walk-forward validation and yearly stability analysis
- Add better calibration evaluation if needed for thresholding

Long-term:
- Online inference pipeline for daily operation
- Automated backtest + report generation
- Model monitoring and drift checks

---

## Notes
- All time splits are strictly forward-looking (no leakage)
- Feature spec used in training is saved per run
- ROI simulations are designed to be extensible to more bet types
