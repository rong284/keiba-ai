import argparse
import gzip
import io
import os
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


def project_root() -> Path:
    # ~/work/keiba-ai/src/data/scraping/parse_race_html_to_tsv.py -> ~/work/keiba-ai
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
    else:
        return path.read_bytes()


def strip_weird_tags(html_bytes: bytes) -> bytes:
    # netkeibaで稀に混じる謎タグ対策（元コード踏襲）
    return (
        html_bytes
        .replace(b"<diary_snap_cut>", b"")
        .replace(b"</diary_snap_cut>", b"")
    )


def extract_table_html(soup: BeautifulSoup) -> str:
    """
    race_table_01 nk_tb_common を優先して取り出す
    """
    table = soup.find("table", class_="race_table_01 nk_tb_common")
    if table is None:
        table = soup.find("table", class_="race_table_01")
    if table is None:
        raise ValueError("race_table_01 not found")
    return str(table)


def extract_ids_from_table(table: BeautifulSoup, pattern: str, digits: int) -> list[str]:
    """
    テーブル内のa hrefからIDを抽出
    pattern例: r"^/horse/" , digits例:10
    """
    a_list = table.find_all("a", href=re.compile(pattern))
    ids = []
    for a in a_list:
        href = a.get("href", "")
        m = re.search(rf"(\d{{{digits}}})", href)
        if m:
            ids.append(m.group(1))
    return ids


def extract_owner_ids(table: BeautifulSoup) -> list[str]:
    a_list = table.find_all("a", href=re.compile(r"^/owner/"))
    ids = []
    for a in a_list:
        href = a.get("href", "")
        m = re.search(r"(\d{6})", href)
        if m:
            ids.append(m.group(1))
        else:
            # 英数字6桁のケース（たまにある）
            m2 = re.search(r"/owner/([A-Za-z0-9]{6})/", href)
            if m2:
                ids.append(m2.group(1))
    return ids


def safe_assign(df: pd.DataFrame, col: str, values: list[str], fill: str = "") -> None:
    """
    valuesの長さがdf行数と一致しない場合でも落ちないようにする
    """
    n = len(df)
    if len(values) == n:
        df[col] = values
    else:
        # 合わない時は一旦空で埋めて、入る分だけ詰める
        tmp = [fill] * n
        for i, v in enumerate(values[:n]):
            tmp[i] = v
        df[col] = tmp


def parse_one_race(path: Path) -> pd.DataFrame:
    """
    レースHTML 1ファイル -> 結果テーブルDataFrame（行=馬）を返す
    """
    race_id = normalize_race_id(path)

    html_bytes = strip_weird_tags(load_html_bytes(path))
    soup = BeautifulSoup(html_bytes, "lxml")

    table_html = extract_table_html(soup)
    df = pd.read_html(io.StringIO(table_html))[0]

    # ID追加（テーブルをBeautifulSoup化してリンク抽出）
    table_soup = BeautifulSoup(table_html, "lxml")

    horse_ids = extract_ids_from_table(table_soup, r"^/horse/", 10)
    jockey_ids = extract_ids_from_table(table_soup, r"^/jockey/", 5)
    trainer_ids = extract_ids_from_table(table_soup, r"^/trainer/", 5)
    owner_ids = extract_owner_ids(table_soup)

    safe_assign(df, "horse_id", horse_ids)
    safe_assign(df, "jockey_id", jockey_ids)
    safe_assign(df, "trainer_id", trainer_ids)
    safe_assign(df, "owner_id", owner_ids)

    # race_id を index or column に（後段で扱いやすいよう両方でもOK）
    df.insert(0, "race_id", race_id)
    return df


def get_race_table(year: int, html_dir: Path, out_path: Path) -> pd.DataFrame:
    files = sorted(html_dir.glob("*.html.gz"))  # 今回はこれが基本
    if not files:
        # 旧形式(.bin/.html)も拾う
        files = sorted(list(html_dir.glob("*.bin")) + list(html_dir.glob("*.html")))

    dfs = []
    failed = []

    for f in tqdm(files, desc=f"{year}年レーステーブル取得", leave=True):
        try:
            df = parse_one_race(f)
            dfs.append(df)
        except Exception as e:
            failed.append((f.name, str(e)))

    if not dfs:
        raise RuntimeError(f"No race tables parsed. failed={len(failed)}")

    concat_df = pd.concat(dfs, ignore_index=True)

    # 列名の空白除去
    concat_df.columns = [c.replace(" ", "") for c in concat_df.columns]

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_df.to_csv(out_path, sep="\t", index=False)

    # 失敗ログ
    if failed:
        log_path = out_path.parent / f"parse_failed_{year}.txt"
        log_path.write_text("\n".join([f"{n}\t{msg}" for n, msg in failed]), encoding="utf-8")
        print(f"[warn] failed parses: {len(failed)} (see {log_path})")

    print(f"[done] parsed={len(dfs)} rows={len(concat_df)} saved={out_path}")
    return concat_df


def main():
    root = project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--html-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    html_dir = Path(args.html_dir) if args.html_dir else (root / "data/html/race" / str(args.year))
    out_path = Path(args.out) if args.out else (root / "data/rawdf/result" / f"result_{args.year}.tsv")

    get_race_table(args.year, html_dir=html_dir, out_path=out_path)


if __name__ == "__main__":
    main()
