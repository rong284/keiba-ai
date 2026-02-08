import argparse
import subprocess
from pathlib import Path


def project_root() -> Path:
    # パス例: ~/work/keiba-ai/src/data/scraping/run_bulk_scraping.py -> ~/work/keiba-ai
    return Path(__file__).resolve().parents[3]


def run_cmd(args: list[str]) -> None:
    result = subprocess.run(args)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(args)}")


def main() -> None:
    root = project_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--year-start", type=int, default=2015)
    ap.add_argument("--year-end", type=int, default=2020)
    ap.add_argument("--peds-year-start", type=int, default=2015)
    ap.add_argument("--peds-year-end", type=int, default=2025)
    ap.add_argument("--peds-from-gcs", action="store_true", default=False)
    ap.add_argument("--gcs-horse-prefix-base", default="gs://keiba-ai-data-1988/keiba-ai/data/html/horse")
    ap.add_argument("--sync-gcs", action="store_true", default=False)
    ap.add_argument("--race", action="store_true", default=True)
    ap.add_argument("--no-race", dest="race", action="store_false")
    ap.add_argument("--parse-race", action="store_true", default=True)
    ap.add_argument("--no-parse-race", dest="parse_race", action="store_false")
    ap.add_argument("--horse", action="store_true", default=True)
    ap.add_argument("--no-horse", dest="horse", action="store_false")
    ap.add_argument("--peds", action="store_true", default=True)
    ap.add_argument("--no-peds", dest="peds", action="store_false")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--sleep-min", type=float, default=2.5)
    ap.add_argument("--sleep-max", type=float, default=3.5)
    ap.add_argument("--results-glob", default=None)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--headed", dest="headless", action="store_false")
    args = ap.parse_args()

    race_years = range(args.year_start, args.year_end + 1)
    peds_years = range(args.peds_year_start, args.peds_year_end + 1)

    if args.race:
        for y in race_years:
            run_cmd(
                [
                    "python",
                    str(root / "src/data_collection/scraping/scrape_race_data.py"),
                    "--year",
                    str(y),
                    "--concurrency",
                    str(args.concurrency),
                    "--sleep-min",
                    str(args.sleep_min),
                    "--sleep-max",
                    str(args.sleep_max),
                    "--headless" if args.headless else "--headed",
                ]
            )
            if args.parse_race:
                run_cmd(
                    [
                        "python",
                        str(root / "src/data_collection/scraping/parse_race_html_to_csv.py"),
                        "--year",
                        str(y),
                    ]
                )

    if args.horse:
        for y in race_years:
            results_path = root / f"data/rawdf/result/result_{y}.csv"
            if args.results_glob:
                results_arg = ["--results-glob", args.results_glob]
            else:
                if not results_path.exists():
                    print(f"[warn] result file missing: {results_path} (skip horse {y})")
                    continue
                results_arg = ["--results-path", str(results_path)]

            out_dir = root / f"data/html/horse/{y}"
            run_cmd(
                [
                    "python",
                    str(root / "src/data_collection/scraping/scrape_horse_result.py"),
                    *results_arg,
                    "--out-dir",
                    str(out_dir),
                    "--concurrency",
                    str(args.concurrency),
                    "--sleep-min",
                    str(args.sleep_min),
                    "--sleep-max",
                    str(args.sleep_max),
                    "--headless" if args.headless else "--headed",
                ]
            )

    if args.peds:
        if args.peds_from_gcs:
            out_dir = root / "data/html/peds"
            run_cmd(
                [
                    "python",
                    str(root / "src/data_collection/scraping/scrape_horse_peds.py"),
                    "--horse-ids-gcs-prefix",
                    f"{args.gcs_horse_prefix_base}/",
                    "--out-dir",
                    str(out_dir),
                    "--concurrency",
                    str(args.concurrency),
                    "--sleep-min",
                    str(args.sleep_min),
                    "--sleep-max",
                    str(args.sleep_max),
                    "--headless" if args.headless else "--headed",
                ]
            )
        else:
            for y in peds_years:
                results_path = root / f"data/rawdf/result/result_{y}.csv"
                if args.results_glob:
                    results_arg = ["--results-glob", args.results_glob]
                else:
                    if results_path.exists():
                        results_arg = ["--results-path", str(results_path)]
                    else:
                        print(f"[warn] result file missing: {results_path} (skip peds {y})")
                        continue

                out_dir = root / f"data/html/peds/{y}"
                run_cmd(
                    [
                        "python",
                        str(root / "src/data_collection/scraping/scrape_horse_peds.py"),
                        *results_arg,
                        "--out-dir",
                        str(out_dir),
                        "--concurrency",
                        str(args.concurrency),
                        "--sleep-min",
                        str(args.sleep_min),
                        "--sleep-max",
                        str(args.sleep_max),
                        "--headless" if args.headless else "--headed",
                    ]
                )

    if args.sync_gcs:
        if args.race:
            for y in race_years:
                run_cmd(
                    [
                        "python",
                        str(root / "src/data_collection/scraping/sync_html_to_gcs.py"),
                        "--kind",
                        "race",
                        "--year",
                        str(y),
                    ]
                )
        if args.horse:
            run_cmd(
                [
                    "python",
                    str(root / "src/data_collection/scraping/sync_html_to_gcs.py"),
                    "--kind",
                    "horse",
                ]
            )
        if args.peds:
            run_cmd(
                [
                    "python",
                    str(root / "src/data_collection/scraping/sync_html_to_gcs.py"),
                    "--kind",
                    "peds",
                ]
            )


if __name__ == "__main__":
    main()
