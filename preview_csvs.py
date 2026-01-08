import gzip
from pathlib import Path

def read_first_lines(filepath: Path, n: int = 5) -> list[str]:
    """Read first n lines from a csv or csv.gz file."""
    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return [f.readline().rstrip('\n') for _ in range(n)]
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [f.readline().rstrip('\n') for _ in range(n)]

def main():
    data_dir = Path(__file__).parent / "tardis_binance_btc"
    
    # Find all csv and csv.gz files
    csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.csv.gz"))
    
    for csv_path in sorted(csv_files):
        print(f"\n{'='*60}")
        print(f"Path: {csv_path}")
        print(f"{'='*60}")
        lines = read_first_lines(csv_path, 5)
        for i, line in enumerate(lines, 1):
            print(f"{i}: {line}")

if __name__ == "__main__":
    main()

