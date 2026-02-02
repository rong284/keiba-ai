# 旧ファイル名: parse_race_info_html_to_tsv.py
import argparse
import gzip
import os
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from bs4.dammit import UnicodeDammit
from tqdm import tqdm


def project_root() -> Path:
    # パス例: ~/work/keiba-ai/src/data/scraping/parse_race_info_html_to_tsv.py -> ~/work/keiba-ai
    return Path(__file__).resolve().parents[3]


def normalize_race_id(file: Path) -> str:
    """
    ファイル名から12桁race_idを抽出して返す
    例: 202403020501.html.gz -> 202403020501
    """
    m = re.search(r"(\d{12})", file.name)
    if not m:
        raise ValueError(f"race_id not found in filename: {file.name}")
    return m.group(1)


def load_html_bytes(path: Path) -> bytes:
    """
    .html / .bin / .gz などに対応してHTML bytesを返す
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def strip_weird_tags(html_bytes: bytes) -> bytes:
    # タグごと消す（開閉どっちも）
    html_bytes = re.sub(rb"</?diary_snap_cut>", b"", html_bytes)
    return html_bytes


def decode_html_bytes(html_bytes: bytes) -> str:
    """
    文字化けしにくいようにエンコーディングを推測してdecodeする
    """
    dammit = UnicodeDammit(html_bytes)
    if dammit.unicode_markup is not None:
        return dammit.unicode_markup
    return html_bytes.decode("utf-8", errors="replace")


def parse_one_race_info(path: Path) -> dict:
    """
    raceページの data_intro からレース概要情報を抽出して1行分のdictを返す
    """
    race_id = normalize_race_id(path)

    html_bytes = strip_weird_tags(load_html_bytes(path))
    text = decode_html_bytes(html_bytes)
    soup = BeautifulSoup(text, "lxml")

    intro = soup.find("div", class_="data_intro")
    if intro is None:
        title = soup.title.get_text(strip=True) if soup.title else "(no title)"
        raise ValueError(f"data_intro not found (title={title})")

    h1 = intro.find("h1")
    if h1 is None:
        raise ValueError("data_intro h1 not found")

    p_tags = intro.find_all("p")
    p1_text = p_tags[0].get_text(" ", strip=True) if len(p_tags) >= 1 else ""
    p2_text = p_tags[1].get_text(" ", strip=True) if len(p_tags) >= 2 else ""

    info1_list = re.findall(r"[\w:]+", p1_text.replace(" ", ""))
    info2_list = re.findall(r"\w+", p2_text)

    return {
        "race_id": race_id,
        "title": h1.get_text(strip=True),
        "info1": info1_list,
        "info2": info2_list,
    }


def get_race_info(year: int, html_dir: Path, out_path: Path) -> pd.DataFrame:
    files = sorted(html_dir.glob("*.html.gz"))
    if not files:
        # 旧形式(.bin/.html)も拾う
        files = sorted(list(html_dir.glob("*.bin")) + list(html_dir.glob("*.html")))

    rows = []
    failed = []

    for f in tqdm(files, desc=f"{year}年レース情報取得", leave=True):
        try:
            rows.append(parse_one_race_info(f))
        except Exception as e:
            failed.append((f.name, str(e)))

    if not rows:
        raise RuntimeError(f"No race infos parsed. failed={len(failed)}")

    df = pd.DataFrame(rows)

    # 列名の空白除去
    df.columns = [str(c).replace(" ", "") for c in df.columns]
    df = df[["race_id", "title", "info1", "info2"]]

    # 保存（CSV）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=",", index=False, encoding="utf-8-sig")

    # 失敗ログ
    if failed:
        log_path = out_path.parent / f"parse_failed_race_info_{year}.txt"
        log_path.write_text("\n".join([f"{n}\t{msg}" for n, msg in failed]), encoding="utf-8")
        print(f"[warn] failed parses: {len(failed)} (see {log_path})")

    print(f"[done] parsed={len(rows)} rows={len(df)} saved={out_path}")
    return df


def main():
    root = project_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--html-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    html_dir = Path(args.html_dir) if args.html_dir else (root / "data/html/race" / str(args.year))
    out_path = Path(args.out) if args.out else (root / "data/rawdf/race_info" / f"race_info_{args.year}.csv")

    # tqdm（進捗バー）: どれだけ処理が進んだかを表示するやつ
    get_race_info(args.year, html_dir=html_dir, out_path=out_path)


if __name__ == "__main__":
    main()
