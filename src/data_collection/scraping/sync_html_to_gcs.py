import argparse
import subprocess
from pathlib import Path


def project_root() -> Path:
    # パス例: ~/work/keiba-ai/src/data/scraping/sync_html_to_gcs.py -> ~/work/keiba-ai
    return Path(__file__).resolve().parents[3]


def run_cmd(args: list[str]) -> str:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(args)}\n{msg}")
    return result.stdout


def normalize_gcs_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else prefix + "/"


def list_remote_basenames(gcs_prefix: str) -> set[str]:
    result = subprocess.run(
        ["gsutil", "ls", gcs_prefix],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip()
        # 初回同期でprefix配下にまだオブジェクトがない場合は空として扱う
        if "One or more URLs matched no objects." in msg:
            return set()
        raise RuntimeError(f"command failed: gsutil ls {gcs_prefix}\n{msg}")

    out = result.stdout
    names: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line or line.endswith("/"):
            continue
        name = line.split("/")[-1]
        if name.endswith(".html.gz") or name.endswith(".html") or name.endswith(".bin"):
            names.add(name)
    return names


def list_local_files(local_dir: Path) -> list[Path]:
    if not local_dir.exists():
        return []
    files = []
    files.extend(sorted(local_dir.glob("*.html.gz")))
    files.extend(sorted(local_dir.glob("*.html")))
    files.extend(sorted(local_dir.glob("*.bin")))
    # 重複排除（.html.gz と .html などが混在する場合）
    uniq = []
    seen = set()
    for f in files:
        if f.name in seen:
            continue
        seen.add(f.name)
        uniq.append(f)
    return uniq


def chunked(seq: list[Path], size: int) -> list[list[Path]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def main() -> None:
    root = project_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["race", "horse", "peds"], default="race")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--gcs-prefix", default=None)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    if args.local_dir:
        local_dir = Path(args.local_dir)
    else:
        if args.kind == "race":
            if args.year is None:
                raise ValueError("--year is required when kind=race and --local-dir is not set.")
            local_dir = root / "data/html/race" / str(args.year)
        elif args.kind == "horse":
            local_dir = root / "data/html/horse"
        else:
            local_dir = root / "data/html/peds"

    if args.gcs_prefix:
        gcs_prefix = normalize_gcs_prefix(args.gcs_prefix)
    else:
        base = f"gs://keiba-ai-data-1988/keiba-ai/data/html/{args.kind}/"
        if args.kind == "race":
            if args.year is None:
                raise ValueError("--year is required when kind=race and --gcs-prefix is not set.")
            gcs_prefix = normalize_gcs_prefix(base + f"{args.year}/")
        else:
            gcs_prefix = normalize_gcs_prefix(base)

    local_files = list_local_files(local_dir)
    if not local_files:
        raise FileNotFoundError(f"no local html files found in {local_dir}")

    remote_names = list_remote_basenames(gcs_prefix)
    to_upload = [f for f in local_files if f.name not in remote_names]

    print(f"local_files={len(local_files)} remote_files={len(remote_names)} missing={len(to_upload)}")
    if args.dry_run or not to_upload:
        if to_upload:
            print("missing examples:")
            for f in to_upload[:10]:
                print(f"  {f.name}")
        return

    for batch in chunked(to_upload, args.batch_size):
        cmd = ["gsutil", "-m", "cp", "-n"] + [str(p) for p in batch] + [gcs_prefix]
        run_cmd(cmd)

    print(f"[done] uploaded={len(to_upload)} to {gcs_prefix}")


if __name__ == "__main__":
    main()
