import argparse
import asyncio
import gzip
import random
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm
from playwright.async_api import async_playwright, Page

BASE_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
]


def project_root() -> Path:
    # パス例: ~/work/keiba-ai/src/data/scraping/scrape_horse_peds.py -> ~/work/keiba-ai
    return Path(__file__).resolve().parents[3]


def resolve_result_files(results_path: str, results_glob: Optional[str]) -> list[Path]:
    """
    results_path: ファイル/ディレクトリ/グロブいずれでも受け付ける
    results_glob: 指定があればこちらを優先
    """
    if results_glob:
        return sorted(Path().glob(results_glob))
    path = Path(results_path)
    if path.is_dir():
        return sorted(path.glob("result*.csv"))
    if any(ch in results_path for ch in ["*", "?", "["]):
        return sorted(Path().glob(results_path))
    if path.is_file():
        return [path]
    return []


def load_horse_ids(files: list[Path], sep: str, limit: int | None) -> list[str]:
    ids: set[str] = set()
    for f in files:
        df = pd.read_csv(f, sep=sep, encoding="utf-8-sig")
        if "horse_id" not in df.columns:
            raise ValueError(f"horse_id column not found: {f}")
        for hid in df["horse_id"].dropna().astype(str).tolist():
            ids.add(hid)
    result = sorted(ids)
    if limit is not None:
        return result[:limit]
    return result


def load_horse_ids_from_dir(html_dir: Path, limit: int | None) -> list[str]:
    ids: set[str] = set()
    if not html_dir.exists():
        raise FileNotFoundError(f"horse html dir not found: {html_dir}")
    for p in html_dir.glob("*.html.gz"):
        name = p.name
        if name.endswith(".html.gz"):
            ids.add(name[:-8])
    result = sorted(ids)
    if limit is not None:
        return result[:limit]
    return result


def load_horse_ids_from_gcs(gcs_prefix: str, limit: int | None) -> list[str]:
    if not gcs_prefix.endswith("/"):
        gcs_prefix = gcs_prefix + "/"
    result = subprocess.run(
        ["gsutil", "ls", gcs_prefix],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"gsutil ls failed: {msg}")

    ids: set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.endswith("/"):
            continue
        name = line.split("/")[-1]
        if name.endswith(".html.gz"):
            ids.add(name[:-8])
        elif name.endswith(".html"):
            ids.add(name[:-5])
        elif name.endswith(".bin"):
            ids.add(name[:-4])
    result_ids = sorted(ids)
    if limit is not None:
        return result_ids[:limit]
    return result_ids


def out_path(out_dir: Path, horse_id: str) -> Path:
    return out_dir / f"{horse_id}.html.gz"


def save_gz(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(html.encode("utf-8"))


def load_existing_ids(out_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not out_dir.exists():
        return ids
    for p in out_dir.glob("*.html.gz"):
        name = p.name
        if name.endswith(".html.gz"):
            ids.add(name[:-8])
    return ids


def load_existing_ids_gcs(gcs_prefix: str) -> set[str]:
    """
    GCS上の既存HTMLを列挙してID集合を返す
    """
    if not gcs_prefix.endswith("/"):
        gcs_prefix = gcs_prefix + "/"
    result = subprocess.run(
        ["gsutil", "ls", gcs_prefix],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"gsutil ls failed: {msg}")

    ids: set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.endswith("/"):
            continue
        name = line.split("/")[-1]
        if name.endswith(".html.gz"):
            ids.add(name[:-8])
        elif name.endswith(".html"):
            ids.add(name[:-5])
        elif name.endswith(".bin"):
            ids.add(name[:-4])
    return ids


async def fetch_html(page: Page, url: str, timeout_ms: int = 45_000) -> str:
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    await page.wait_for_selector("body", timeout=30_000)
    return await page.content()


async def worker(
    name: str,
    queue: asyncio.Queue,
    out_dir: Path,
    sem: asyncio.Semaphore,
    max_retry: int,
    sleep_min: float,
    sleep_max: float,
    headless: bool,
    failed: list[str],
    saved: list[str],
):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
        )
        page = await context.new_page()

        while True:
            horse_id = await queue.get()
            if horse_id is None:
                queue.task_done()
                break

            path = out_path(out_dir, horse_id)
            url = BASE_URL.format(horse_id=horse_id)

            async with sem:
                ok = False
                for attempt in range(1, max_retry + 1):
                    try:
                        html = await fetch_html(page, url)
                        save_gz(path, html)
                        saved.append(horse_id)
                        ok = True
                        break
                    except Exception as e:
                        backoff = min(60.0, (2 ** attempt)) + random.uniform(0.0, 1.5)
                        print(f"[{name}] retry {attempt}/{max_retry} horse_id={horse_id} wait={backoff:.1f}s err={e}")
                        await asyncio.sleep(backoff)

                if not ok:
                    failed.append(horse_id)

                await asyncio.sleep(random.uniform(sleep_min, sleep_max))

            queue.task_done()

        await context.close()
        await browser.close()


async def run(
    horse_ids: Iterable[str],
    out_dir: Path,
    skip: bool,
    concurrency: int,
    max_retry: int,
    sleep_min: float,
    sleep_max: float,
    headless: bool,
    log_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    q: asyncio.Queue = asyncio.Queue()
    total = 0
    skipped = 0
    existing_ids = load_existing_ids(out_dir) if skip else set()
    for hid in horse_ids:
        hid = str(hid)
        if skip and hid in existing_ids:
            skipped += 1
            continue
        await q.put(hid)
        total += 1

    for _ in range(concurrency):
        await q.put(None)

    sem = asyncio.Semaphore(concurrency)
    failed: list[str] = []
    saved: list[str] = []

    tasks = [
        asyncio.create_task(
            worker(
                name=f"W{i}",
                queue=q,
                out_dir=out_dir,
                sem=sem,
                max_retry=max_retry,
                sleep_min=sleep_min,
                sleep_max=sleep_max,
                headless=headless,
                failed=failed,
                saved=saved,
            )
        )
        for i in range(concurrency)
    ]

    with tqdm(total=total, desc="horse_peds scrape", leave=True) as pbar:
        done_prev = 0
        while any(not t.done() for t in tasks):
            await asyncio.sleep(1.0)
            done_now = len(saved) + len(failed)
            if done_now > done_prev:
                pbar.update(done_now - done_prev)
                done_prev = done_now

        await q.join()

    if skip:
        print(f"skipped(existing)={skipped}")
    (log_dir / "peds_saved.txt").write_text("\n".join(saved), encoding="utf-8")
    (log_dir / "peds_failed.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"saved={len(saved)} failed={len(failed)} (logs: {log_dir})")


def main():
    root = project_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-path", default=str(root / "data/rawdf/result"))
    ap.add_argument("--results-glob", default=None)
    ap.add_argument("--out-dir", default=str(root / "data/html/peds"))
    ap.add_argument("--log-dir", default=str(root / "data/logs"))
    ap.add_argument("--sep", default=",")
    ap.add_argument("--skip", action="store_true", default=True)
    ap.add_argument("--no-skip", dest="skip", action="store_false")
    ap.add_argument("--horse-ids-dir", default=None)
    ap.add_argument("--horse-ids-gcs-prefix", default=None)
    ap.add_argument("--gcs-prefix", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1)  # まずは1推奨
    ap.add_argument("--max-retry", type=int, default=3)
    ap.add_argument("--sleep-min", type=float, default=2.5)
    ap.add_argument("--sleep-max", type=float, default=3.5)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--headed", dest="headless", action="store_false")
    args = ap.parse_args()

    if args.horse_ids_dir and args.horse_ids_gcs_prefix:
        raise ValueError("--horse-ids-dir and --horse-ids-gcs-prefix are mutually exclusive.")

    if args.horse_ids_dir:
        horse_ids = load_horse_ids_from_dir(Path(args.horse_ids_dir), limit=args.limit)
    elif args.horse_ids_gcs_prefix:
        horse_ids = load_horse_ids_from_gcs(args.horse_ids_gcs_prefix, limit=args.limit)
    else:
        files = resolve_result_files(args.results_path, args.results_glob)
        if not files:
            raise FileNotFoundError(f"No result csv files found. path={args.results_path} glob={args.results_glob}")
        horse_ids = load_horse_ids(files, sep=args.sep, limit=args.limit)

    gcs_existing: set[str] = set()
    if args.skip and args.gcs_prefix:
        gcs_existing = load_existing_ids_gcs(args.gcs_prefix)
        if gcs_existing:
            horse_ids = [hid for hid in horse_ids if str(hid) not in gcs_existing]
            print(f"skip by gcs existing={len(gcs_existing)} remaining={len(horse_ids)}")

    asyncio.run(
        run(
            horse_ids=horse_ids,
            out_dir=Path(args.out_dir),
            skip=args.skip,
            concurrency=args.concurrency,
            max_retry=args.max_retry,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
            headless=args.headless,
            log_dir=Path(args.log_dir),
        )
    )


if __name__ == "__main__":
    main()
