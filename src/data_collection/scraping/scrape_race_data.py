import argparse
import asyncio
import gzip
import random
import re
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
from playwright.async_api import async_playwright, Page

BASE_CAL_URL = "https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
BASE_RACE_LIST_URL = "https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date}"
BASE_RACE_DB_URL = "https://db.netkeiba.com/race/{race_id}"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
]


def project_root() -> Path:
    # パス例: ~/work/keiba-ai/src/data/scraping/scrape_race_data.py -> ~/work/keiba-ai
    return Path(__file__).resolve().parents[3]


def save_gz(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(content.encode("utf-8"))


def out_path(out_dir: Path, race_id: str) -> Path:
    return out_dir / f"{race_id}.html.gz"


async def fetch_content(page: Page, url: str, timeout_ms: int = 45_000) -> str:
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    await page.wait_for_selector("body", timeout=30_000)
    return await page.content()


async def fetch_with_retry(
    page: Page,
    url: str,
    max_retry: int,
    base_sleep: float,
) -> str | None:
    for attempt in range(1, max_retry + 1):
        try:
            return await fetch_content(page, url)
        except Exception as e:
            # 指数バックオフ + ジッター
            backoff = min(60.0, (base_sleep * (2 ** attempt))) + random.uniform(0.0, 1.5)
            print(f"[retry {attempt}/{max_retry}] {url} wait={backoff:.1f}s err={e}")
            await asyncio.sleep(backoff)
    return None


async def collect_kaisai_dates(page: Page, year: int, sleep_min: float, sleep_max: float) -> list[str]:
    """カレンダー（月） -> kaisai_date(YYYYMMDD) を収集"""
    dates: set[str] = set()

    for month in tqdm(range(1, 13), desc=f"calendar {year}", leave=False):
        url = BASE_CAL_URL.format(year=year, month=month)
        html = await fetch_with_retry(page, url, max_retry=3, base_sleep=1.5)
        if html is None:
            continue

        # kaisai_date=YYYYMMDD を抜く
        found = re.findall(r"kaisai_date=(\d{8})", html)
        for d in found:
            dates.add(d)

        await asyncio.sleep(random.uniform(sleep_min, sleep_max))

    return sorted(dates)


async def collect_race_ids_for_date(page: Page, kaisai_date: str, sleep_min: float, sleep_max: float) -> list[str]:
    """開催日一覧(kaisai_date) -> race_id(12桁) をDOMから収集"""
    url = BASE_RACE_LIST_URL.format(kaisai_date=kaisai_date)

    # goto のレスポンスを取ってステータスも見る
    resp = await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    status = resp.status if resp is not None else None
    if status is not None and status >= 400:
        print(f"[race_list bad status] {kaisai_date} status={status}")
        return []

    # JS描画待ち：レース行が出るまで待つ
    try:
        await page.wait_for_selector(".RaceList_DataItem a", timeout=25_000)
    except Exception:
        # ここに来るのは「描画されない」か「別ページ（制限/同意）」の可能性
        # デバッグしたい時は中身を少し出す
        snippet = (await page.content())[:500]
        print(f"[race_list selector timeout] {kaisai_date} url={url} snippet={snippet}")
        return []

    anchors = await page.query_selector_all(".RaceList_DataItem a")
    hrefs = []
    for a in anchors:
        h = await a.get_attribute("href")
        if h:
            hrefs.append(h)

    # href から race_id=12桁 を抜く
    ids = []
    for h in hrefs:
        m = re.search(r"race_id=(\d{12})", h)
        if m:
            ids.append(m.group(1))

    # 念のため重複排除
    ids = sorted(set(ids))

    await asyncio.sleep(random.uniform(sleep_min, sleep_max))
    return ids



async def scrape_race_html_worker(
    name: str,
    queue: asyncio.Queue,
    browser,
    out_dir: Path,
    skip: bool,
    max_retry: int,
    sleep_min: float,
    sleep_max: float,
    saved: list[str],
    failed: list[str],
    restart_every: int,
    saved_log: Path,
    failed_log: Path,
):
    async def new_page():
        ctx = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
        )
        page = await ctx.new_page()
        return ctx, page

    ctx, page = await new_page()
    processed = 0

    while True:
        race_id = await queue.get()
        if race_id is None:
            queue.task_done()
            break

        path = out_path(out_dir, race_id)
        if skip and path.is_file():
            queue.task_done()
            processed += 1
            if restart_every > 0 and processed % restart_every == 0:
                await ctx.close()
                ctx, page = await new_page()
            continue

        url = BASE_RACE_DB_URL.format(race_id=race_id)
        html = await fetch_with_retry(page, url, max_retry=max_retry, base_sleep=2.0)

        if html is None:
            failed.append(race_id)
            failed_log.write_text("\n".join(failed), encoding="utf-8")
        else:
            save_gz(path, html)
            saved.append(race_id)
            saved_log.write_text("\n".join(saved), encoding="utf-8")

        await asyncio.sleep(random.uniform(sleep_min, sleep_max))
        queue.task_done()
        processed += 1
        if restart_every > 0 and processed % restart_every == 0:
            await ctx.close()
            ctx, page = await new_page()

    await ctx.close()


async def run(
    year: int,
    out_dir: Path,
    log_dir: Path,
    skip: bool,
    concurrency: int,
    sleep_min: float,
    sleep_max: float,
    headless: bool,
    restart_every: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


    saved: list[str] = []
    failed: list[str] = []
    date_failed: list[str] = []
    saved_log = log_dir / f"race_{year}_saved.txt"
    failed_log = log_dir / f"race_{year}_failed.txt"
    saved_log.write_text("", encoding="utf-8")
    failed_log.write_text("", encoding="utf-8")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)

        # 収集フェーズ用コンテキスト（軽め）
        ctx_collect = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
        )
        page_collect = await ctx_collect.new_page()

        # 1) 開催日一覧
        kaisai_dates = await collect_kaisai_dates(page_collect, year, sleep_min, sleep_max)
        if not kaisai_dates:
            print(f"[warn] no kaisai_dates collected for year={year}")

        # 2) レースID一覧
        race_ids: set[str] = set()
        for d in tqdm(kaisai_dates, desc=f"race_id list {year}", leave=False):
            ids = await collect_race_ids_for_date(page_collect, d, sleep_min, sleep_max)
            if not ids:
                date_failed.append(d)
            for rid in ids:
                race_ids.add(rid)

        await ctx_collect.close()

        # 3) レースHTML取得フェーズ（ワーカー別コンテキストにしてUA分散）
        q: asyncio.Queue = asyncio.Queue()
        race_id_list = sorted(race_ids)
        for rid in race_id_list:
            await q.put(rid)
        for _ in range(concurrency):
            await q.put(None)

        tasks = []
        for i in range(concurrency):
            task = asyncio.create_task(
                scrape_race_html_worker(
                    name=f"W{i}",
                    queue=q,
                    browser=browser,
                    out_dir=out_dir,
                    skip=skip,
                    max_retry=3,
                    sleep_min=sleep_min,
                    sleep_max=sleep_max,
                    saved=saved,
                    failed=failed,
                    restart_every=restart_every,
                    saved_log=saved_log,
                    failed_log=failed_log,
                )
            )
            tasks.append(task)

        with tqdm(total=len(race_id_list), desc=f"race html {year}", leave=True) as pbar:
            done_prev = 0
            while any(not t.done() for t in tasks):
                await asyncio.sleep(1.0)
                done_now = len(saved) + len(failed)
                if done_now > done_prev:
                    pbar.update(done_now - done_prev)
                    done_prev = done_now
            await q.join()

        await browser.close()

    # ログ出力
    saved_log.write_text("\n".join(saved), encoding="utf-8")
    failed_log.write_text("\n".join(failed), encoding="utf-8")
    (log_dir / f"race_{year}_date_failed.txt").write_text("\n".join(date_failed), encoding="utf-8")

    print(f"[done] year={year} race_ids={len(race_id_list)} saved={len(saved)} failed={len(failed)}")
    if date_failed:
        print(f"[warn] date_failed={len(date_failed)} (see log)")


def main():
    root = project_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log-dir", default=str(root / "data/logs"))
    ap.add_argument("--skip", action="store_true", default=True)
    ap.add_argument("--no-skip", dest="skip", action="store_false")
    ap.add_argument("--concurrency", type=int, default=1)  # まずは1推奨
    ap.add_argument("--sleep-min", type=float, default=1.0)  # カレンダー/ID収集もあるので少し短め
    ap.add_argument("--sleep-max", type=float, default=2.0)
    ap.add_argument("--restart-every", type=int, default=0)  # 0=無効, N件ごとにブラウザ再起動
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--headed", dest="headless", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (root / "data/html/race" / str(args.year))

    asyncio.run(
        run(
            year=args.year,
            out_dir=out_dir,
            log_dir=Path(args.log_dir),
            skip=args.skip,
            concurrency=args.concurrency,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
            headless=args.headless,
            restart_every=args.restart_every,
        )
    )


if __name__ == "__main__":
    main()
