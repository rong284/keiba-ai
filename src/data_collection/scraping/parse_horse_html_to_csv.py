# -*- coding: utf-8 -*-
"""
parse_horse_html_to_csv.py

data/html/horse/2010101667.html.gz のような horse 成績HTML(gz) を一括で読み取り、
テーブルを抽出してCSVにまとめて保存する。

- .html.gz を基本に、旧形式(.bin/.html)も拾う
- 文字化けしにくい decode（UnicodeDammit）
- テーブル部分だけから race_id / jockey_id / trainer_id を抽出して列追加
"""

import argparse
import gzip
import io
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, FeatureNotFound
from bs4.dammit import UnicodeDammit
from tqdm import tqdm


def project_root() -> Path:
    # パス例: ~/work/keiba-ai/src/data/scraping/parse_horse_html_to_csv.py -> ~/work/keiba-ai
    return Path(__file__).resolve().parents[3]


def normalize_horse_id(file: Path) -> str:
    """
    ファイル名から horse_id を抽出（例: 2010101667.html.gz -> 2010101667）
    """
    m = re.search(r"(\d{10})", file.name)
    if not m:
        raise ValueError(f"horse_id not found in filename: {file.name}")
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
    # 必要なら変なタグ除去（race 側に合わせた実装）
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


def make_soup(text: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(text, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(text, "html.parser")


def read_html_table(table_html: str) -> pd.DataFrame:
    last_err: Exception | None = None
    for flavor in [None, "bs4", "html5lib"]:
        try:
            if flavor is None:
                return pd.read_html(io.StringIO(table_html))[0]
            return pd.read_html(io.StringIO(table_html), flavor=flavor)[0]
        except Exception as e:
            last_err = e
    assert last_err is not None
    raise last_err


def extract_ids_from_table(table: BeautifulSoup, pattern: str, digits: int) -> list[str]:
    """
    テーブル内のa hrefからIDを抽出
    pattern例: r"^/race/" , digits例:12
    """
    a_list = table.find_all("a", href=re.compile(pattern))
    ids: list[str] = []
    for a in a_list:
        href = a.get("href", "")
        m = re.search(rf"(\d{{{digits}}})", href)
        if m:
            ids.append(m.group(1))
    return ids


def extract_ids_by_row(table: BeautifulSoup, pattern: str, digits: int) -> list[str]:
    """
    Extract IDs per data row to keep alignment with DataFrame rows.
    """
    ids: list[str] = []
    for row in table.select("tr"):
        if not row.find("td"):
            continue
        a = row.find("a", href=re.compile(pattern))
        if not a:
            ids.append("")
            continue
        href = a.get("href", "")
        m = re.search(rf"(\d{{{digits}}})", href)
        ids.append(m.group(1) if m else "")
    return ids


def safe_assign(df: pd.DataFrame, col: str, values: list[str], fill: str = "") -> None:
    """
    valuesの長さがdf行数と一致しない場合でも落ちないようにする
    """
    n = len(df)
    if len(values) == n:
        df[col] = values
        return

    tmp = [fill] * n
    for i, v in enumerate(values[:n]):
        tmp[i] = v
    df[col] = tmp


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and keep ID columns at the end (horse_id stays first).
    """
    df.columns = [str(c).replace(" ", "") for c in df.columns]
    id_cols = ["horse_id", "race_id", "jockey_id", "trainer_id"]
    if "horse_id" in df.columns:
        base_cols = [c for c in df.columns if c not in id_cols]
        tail = [c for c in id_cols if c in df.columns and c != "horse_id"]
        df = df[["horse_id"] + base_cols + tail]
    return df


def pick_horse_results_table(soup: BeautifulSoup) -> BeautifulSoup:
    """
    馬成績ページの「成績テーブル」をできるだけ頑健に拾う。
    netkeiba 側のclass変更に備えて複数候補を試す。
    """
    # よくある候補（環境によって微妙に違う）
    candidates = [
        "table.db_h_race_results.nk_tb_common",
        "table.db_h_race_results",
        "table.horse_table_01",
        "table.race_table_01.nk_tb_common",
        "table.race_table_01",
    ]
    for sel in candidates:
        t = soup.select_one(sel)
        if t is not None:
            return t

    # 最後の手段：class名に db_h_race_results を含むテーブル
    t2 = soup.find("table", class_=re.compile(r"db_h_race_results"))
    if t2 is not None:
        return t2

    title = soup.title.get_text(strip=True) if soup.title else "(no title)"
    raise ValueError(f"horse results table not found (title={title})")


def parse_one_horse(path: Path) -> pd.DataFrame:
    horse_id = normalize_horse_id(path)

    html_bytes = strip_weird_tags(load_html_bytes(path))
    text = decode_html_bytes(html_bytes)
    soup = make_soup(text)

    table_tag = pick_horse_results_table(soup)
    table_html = str(table_tag)

    # テーブルをDataFrame化
    df = read_html_table(table_html)

    # ID抽出は「テーブル部分だけ」で行う（ノイズに強い）
    table_soup = make_soup(table_html)
    race_ids = extract_ids_by_row(table_soup, r"^/race/", 12)
    jockey_ids = extract_ids_by_row(table_soup, r"^/jockey/", 5)
    trainer_ids = extract_ids_by_row(table_soup, r"^/trainer/", 5)
    if len(race_ids) != len(df):
        race_ids = extract_ids_from_table(table_soup, r"^/race/", 12)
    if len(jockey_ids) != len(df):
        jockey_ids = extract_ids_from_table(table_soup, r"^/jockey/", 5)
    if len(trainer_ids) != len(df):
        trainer_ids = extract_ids_from_table(table_soup, r"^/trainer/", 5)

    # 追加列
    safe_assign(df, "race_id", race_ids)
    safe_assign(df, "jockey_id", jockey_ids)
    safe_assign(df, "trainer_id", trainer_ids)

    # horse_id は全行同一なので先頭へ
    df.insert(0, "horse_id", horse_id)

    # 列名の空白除去
    return normalize_columns(df)


def resolve_html_files(html_dir: Path, html_glob: str | None) -> list[Path]:
    if html_glob:
        return sorted(Path().glob(html_glob))
    files = sorted(html_dir.glob("*.html.gz"))
    if files:
        return files
    return sorted(list(html_dir.glob("*.bin")) + list(html_dir.glob("*.html")))


def split_and_save_csv(df: pd.DataFrame, out_path: Path, split_size: int | None) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not split_size:
        df.to_csv(out_path, sep=",", index=False, encoding="utf-8-sig")
        return [out_path]

    total = len(df)
    parts = (total + split_size - 1) // split_size
    width = max(2, len(str(parts)))
    saved: list[Path] = []
    for i in range(parts):
        start = i * split_size
        end = min(start + split_size, total)
        chunk = df.iloc[start:end]
        part_path = make_part_path(out_path, i + 1, width)
        chunk.to_csv(part_path, sep=",", index=False, encoding="utf-8-sig")
        saved.append(part_path)
    return saved


def split_and_save_by_id(df: pd.DataFrame, out_path: Path, split_by_id: int) -> list[Path]:
    if "horse_id" not in df.columns:
        raise ValueError("horse_id column is required for split-by-id")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids_in_order: list[str] = []
    seen: set[str] = set()
    for hid in df["horse_id"].astype(str).tolist():
        if hid in seen:
            continue
        seen.add(hid)
        ids_in_order.append(hid)

    total_ids = len(ids_in_order)
    parts = (total_ids + split_by_id - 1) // split_by_id
    width = max(2, len(str(parts)))
    saved: list[Path] = []
    for i in range(parts):
        start = i * split_by_id
        end = min(start + split_by_id, total_ids)
        chunk_ids = set(ids_in_order[start:end])
        chunk = df[df["horse_id"].astype(str).isin(chunk_ids)]
        part_path = make_part_path(out_path, i + 1, width)
        chunk.to_csv(part_path, sep=",", index=False, encoding="utf-8-sig")
        saved.append(part_path)
    return saved


def make_part_path(out_path: Path, part_index: int, width: int = 2) -> Path:
    stem = out_path.stem
    suffix = out_path.suffix or ".csv"
    return out_path.with_name(f"{stem}_part{str(part_index).zfill(width)}{suffix}")


def get_horse_table(
    html_dir: Path,
    out_path: Path,
    html_glob: str | None,
    limit: int | None,
    split_size: int | None,
    split_by_id: int | None,
) -> pd.DataFrame:
    files = resolve_html_files(html_dir, html_glob)
    if limit is not None:
        files = files[:limit]

    dfs: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []

    if split_by_id and split_size:
        raise ValueError("use only one of --split-size or --split-by-id")

    stream_by_id = split_by_id is not None
    current_ids = 0
    part_index = 1
    part_width = 2
    header_written = False
    saved_paths: list[Path] = []
    total_rows = 0

    for f in tqdm(files, desc="horse results parse", leave=True):
        try:
            df = parse_one_horse(f)
            if stream_by_id:
                if current_ids >= (split_by_id or 0):
                    part_index += 1
                    current_ids = 0
                    header_written = False

                part_path = make_part_path(out_path, part_index, part_width)
                if not saved_paths or saved_paths[-1] != part_path:
                    saved_paths.append(part_path)

                if not header_written:
                    df.to_csv(part_path, sep=",", index=False, encoding="utf-8-sig", mode="w", header=True)
                    header_written = True
                else:
                    df.to_csv(part_path, sep=",", index=False, encoding="utf-8", mode="a", header=False)

                current_ids += 1
                total_rows += len(df)
            else:
                dfs.append(df)
        except Exception as e:
            failed.append((f.name, str(e)))

    if failed:
        log_path = out_path.parent / "parse_failed_horse.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join([f"{n}\t{msg}" for n, msg in failed]), encoding="utf-8")
        print(f"[warn] failed parses: {len(failed)} (see {log_path})")

    if stream_by_id:
        if total_rows == 0:
            first = f"{failed[0][0]}: {failed[0][1]}" if failed else "no files"
            raise RuntimeError(f"No horse tables parsed. failed={len(failed)} first_error={first}")
        concat_df = pd.DataFrame()
    else:
        if not dfs:
            first = f"{failed[0][0]}: {failed[0][1]}" if failed else "no files"
            raise RuntimeError(f"No horse tables parsed. failed={len(failed)} first_error={first}")

        concat_df = pd.concat(dfs, ignore_index=True)
        saved_paths = split_and_save_csv(concat_df, out_path, split_size)

    if len(saved_paths) == 1:
        saved_msg = str(saved_paths[0])
    else:
        saved_msg = f"{saved_paths[0]} .. {saved_paths[-1]} (parts={len(saved_paths)})"
    rows = total_rows if stream_by_id else len(concat_df)
    parsed_files = len(dfs) if not stream_by_id else len(files) - len(failed)
    print(f"[done] parsed_files={parsed_files} rows={rows} saved={saved_msg}")
    return concat_df


def main():
    root = project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-dir", default=None, help="default: <root>/data/html/horse")
    ap.add_argument("--html-glob", default=None, help="glob override (e.g. data/html/horse/*.html.gz)")
    ap.add_argument("--out", default=None, help="default: <root>/data/rawdf/horse/horse_results.csv")
    ap.add_argument("--limit", type=int, default=None, help="limit files for debug")
    ap.add_argument("--split-size", type=int, default=None, help="split rows per file (e.g. 1000)")
    ap.add_argument("--split-by-id", type=int, default=None, help="split by unique horse_id count per file")
    args = ap.parse_args()

    html_dir = Path(args.html_dir) if args.html_dir else (root / "data/html/horse")
    out_path = Path(args.out) if args.out else (root / "data/rawdf/horse" / "horse_results.csv")

    get_horse_table(
        html_dir=html_dir,
        out_path=out_path,
        html_glob=args.html_glob,
        limit=args.limit,
        split_size=args.split_size,
        split_by_id=args.split_by_id,
    )


if __name__ == "__main__":
    main()
