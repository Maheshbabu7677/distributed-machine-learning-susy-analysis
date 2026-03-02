

from pathlib import Path

RAW_DIR = Path("data/raw")
CSV = RAW_DIR / "SUSY.csv"
GZ = RAW_DIR / "SUSY.csv.gz"


def bytes_to_gb(n: int) -> float:
    return n / (1024**3)


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"Missing {CSV}. If you only have .gz, unzip it with: gunzip -k {GZ.name}")

    size_gb = bytes_to_gb(CSV.stat().st_size)
    print(f"Found: {CSV}  size={size_gb:.2f} GB")

    with CSV.open("r", encoding="utf-8") as f:
        first = f.readline().strip()

    cols = first.split(",")
    print(f"First row columns: {len(cols)} (expected 19)")
    print(f"Label (col0): {cols[0]}")
    print("OK")


if __name__ == "__main__":
    main()
