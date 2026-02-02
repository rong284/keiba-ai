import argparse
import gzip
import io
import os
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from bs4.dammit import UnicodeDammit
from tqdm import tqdm


def project_root() -> Path:
    # パス例: ~/work/keiba-ai/src/data/scraping/parse_race_html_to_tsv.py -> ~/work/keiba-ai
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    netkeibaの結果テーブルの列名を正規化する（文字化け対策）
    26列前提：race_id + 21(HTML) + 4(ID) みたいな構成を想定
    """
    # 追加したID列が末尾にある前提
    id_cols = ["horse_id", "jockey_id", "trainer_id", "owner_id"]
    if all(c in df.columns for c in id_cols):
        base_cols = [c for c in df.columns if c not in id_cols]
        # race_id も base 側
        # HTML由来列名が文字化けしても「位置」で直す
        expected_html = [
            "着順","枠番","馬番","馬名","性齢","斤量","騎手","タイム","着差","通過","上り",
            "単勝","人気","馬体重","調教師","馬主","賞金(万円)"
        ]
        # base_cols は race_id + HTML列 なので race_id を除いたぶんを当てる
        if len(base_cols) >= 1 + len(expected_html):
            new_base = ["race_id"] + expected_html[: len(base_cols)-1]
            # 余った列がある場合は元名を残す
            if len(new_base) < len(base_cols):
                new_base += base_cols[len(new_base):]
            rename_map = dict(zip(base_cols, new_base))
            df = df.rename(columns=rename_map)

        # ID列も確実に末尾へ
        df = df[[c for c in df.columns if c not in id_cols] + id_cols]

    # 空白除去
    df.columns = [str(c).replace(" ", "") for c in df.columns]
    return df


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
    # 最後の手段
    return html_bytes.decode("utf-8", errors="replace")



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
    race_id = normalize_race_id(path)

    html_bytes = strip_weird_tags(load_html_bytes(path))

    # meta等を見てencoding推定してからSoupへ
    text = decode_html_bytes(html_bytes)
    soup = BeautifulSoup(text, "lxml")

    # テーブル抽出（class順序・追加classに強い）
    table_tag = soup.select_one("table.race_table_01.nk_tb_common") or soup.select_one("table.race_table_01")
    if table_tag is None:
        title = soup.title.get_text(strip=True) if soup.title else "(no title)"
        raise ValueError(f"race_table_01 not found (title={title})")

    table_html = str(table_tag)
    df = pd.read_html(io.StringIO(table_html))[0]

    # ID抽出はテーブル部分だけから行う（ノイズに強い）
    table_soup = BeautifulSoup(table_html, "lxml")

    horse_ids = extract_ids_from_table(table_soup, r"^/horse/", 10)
    jockey_ids = extract_ids_from_table(table_soup, r"^/jockey/", 5)
    trainer_ids = extract_ids_from_table(table_soup, r"^/trainer/", 5)
    owner_ids = extract_owner_ids(table_soup)

    safe_assign(df, "horse_id", horse_ids)
    safe_assign(df, "jockey_id", jockey_ids)
    safe_assign(df, "trainer_id", trainer_ids)
    safe_assign(df, "owner_id", owner_ids)

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
    concat_df.to_csv(out_path, sep=",", index=False, encoding="utf-8-sig")

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
    out_path = Path(args.out) if args.out else (root / "data/rawdf/result" / f"result_{args.year}.csv")

    get_race_table(args.year, html_dir=html_dir, out_path=out_path)


if __name__ == "__main__":
    main()
