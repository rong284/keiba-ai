import argparse
import gzip
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from bs4.dammit import UnicodeDammit
from tqdm import tqdm


def project_root() -> Path:
    # パス例: ~/work/keiba-ai/src/data/scraping/parse_return_html_to_csv.py -> ~/work/keiba-ai
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


def split_cell_text(tag) -> list[str]:
    if tag is None:
        return [""]
    text = tag.get_text("\n", strip=True)
    if not text:
        return [""]
    return [t for t in text.split("\n") if t != ""]


def parse_one_return(path: Path) -> pd.DataFrame:
    race_id = normalize_race_id(path)

    html_bytes = strip_weird_tags(load_html_bytes(path))
    text = decode_html_bytes(html_bytes)
    soup = BeautifulSoup(text, "lxml")

    pay_block = soup.find("dl", class_="pay_block")
    if pay_block is None:
        title = soup.title.get_text(strip=True) if soup.title else "(no title)"
        raise ValueError(f"pay_block not found (title={title})")

    tables = pay_block.find_all("table", class_="pay_table_01")
    if not tables:
        raise ValueError("pay_table_01 not found")

    rows: list[dict] = []
    for table in tables:
        for tr in table.find_all("tr"):
            th = tr.find("th")
            if th is None:
                continue
            bet_type = th.get_text(strip=True)
            tds = tr.find_all("td")
            if not tds:
                continue

            combos = split_cell_text(tds[0]) if len(tds) >= 1 else [""]
            payouts = split_cell_text(tds[1]) if len(tds) >= 2 else [""]
            pops = split_cell_text(tds[2]) if len(tds) >= 3 else [""]

            n = max(len(combos), len(payouts), len(pops), 1)
            combos += [""] * (n - len(combos))
            payouts += [""] * (n - len(payouts))
            pops += [""] * (n - len(pops))

            for i in range(n):
                rows.append(
                    {
                        "race_id": race_id,
                        "bet_type": bet_type,
                        "combination": combos[i],
                        "payout": payouts[i],
                        "popularity": pops[i],
                    }
                )

    if not rows:
        raise ValueError("no payout rows parsed")

    df = pd.DataFrame(rows)
    df = df[["race_id", "bet_type", "combination", "payout", "popularity"]]
    return df


def get_return_table(year: int, html_dir: Path, out_path: Path) -> pd.DataFrame:
    files = sorted(html_dir.glob("*.html.gz"))
    if not files:
        # 旧形式(.bin/.html)も拾う
        files = sorted(list(html_dir.glob("*.bin")) + list(html_dir.glob("*.html")))

    dfs = []
    failed = []

    for f in tqdm(files, desc=f"{year}年払い戻し取得", leave=True):
        try:
            df = parse_one_return(f)
            dfs.append(df)
        except Exception as e:
            failed.append((f.name, str(e)))

    if not dfs:
        raise RuntimeError(f"No return tables parsed. failed={len(failed)}")

    concat_df = pd.concat(dfs, ignore_index=True)
    concat_df.columns = [str(c).replace(" ", "") for c in concat_df.columns]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_df.to_csv(out_path, sep=",", index=False, encoding="utf-8-sig")

    if failed:
        log_path = out_path.parent / f"parse_failed_return_{year}.txt"
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
    out_path = Path(args.out) if args.out else (root / "data/rawdf/return" / f"return_{args.year}.csv")

    get_return_table(args.year, html_dir=html_dir, out_path=out_path)


if __name__ == "__main__":
    main()
