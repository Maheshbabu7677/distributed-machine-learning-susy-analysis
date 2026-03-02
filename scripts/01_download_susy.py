
"""
01_download_susy.py
Downloads SUSY.csv.gz from UCI into data/raw/

Run:
  python3 scripts/01_download_susy.py
"""

from pathlib import Path
import urllib.request

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"
OUT_DIR = Path("data/raw")
OUT_FILE = OUT_DIR / "SUSY.csv.gz"


def download(url: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and out_file.stat().st_size > 100 * 1024 * 1024:
       
        print(f"Already downloaded (skipping): {out_file} ({out_file.stat().st_size/1024/1024:.1f} MB)")
        return

    tmp = out_file.with_suffix(out_file.suffix + ".part")
    print(f"Downloading:\n  {url}\nTo:\n  {out_file}")

    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = r.length
        downloaded = 0
        chunk_size = 1024 * 1024  

        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = (downloaded / total) * 100
                print(f"\rDownloaded {downloaded/(1024**2):.1f} MB ({pct:.1f}%)", end="")
            else:
                print(f"\rDownloaded {downloaded/(1024**2):.1f} MB", end="")

    print("\nDownload complete.")
    tmp.replace(out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    download(URL, OUT_FILE)