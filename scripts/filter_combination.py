import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "backtest" / "portfolio_combinations.csv"
OUTPUT_DIR = BASE_DIR / "data" / "backtest"
OUTPUT_PATH = OUTPUT_DIR / "filtered_combinations.csv"
OUTPUT_PATH_DISPLAY = f"{BASE_DIR.name}/{OUTPUT_PATH.relative_to(BASE_DIR)}"

# A combination is kept only when every one of these weights is non-zero
WEIGHT_COLUMNS = (
    "equity", "largecap", "midcap", "smallcap", "debt", "liquidcase",
    "ltgiltbees", "realty", "gold", "goldbees", "goldietf",
)

def read_combinations():
    with open(INPUT_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if not columns or any(column not in columns for column in WEIGHT_COLUMNS):
            raise KeyError()
        return columns, list(reader)

def matches(row):
    return all(float(row[column]) != 0 for column in WEIGHT_COLUMNS)

def filter_combinations(columns, rows):
    selected = [row for row in rows if matches(row)]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected)
    return len(selected)

def main():
    try:
        columns, rows = read_combinations()
        if not rows:
            raise ValueError()
        selected = filter_combinations(columns, rows)
    except Exception as error:
        print(f"ERROR : {type(error).__name__}")
        return

    print(f"Filtered {selected}/{len(rows)} combinations and saved to {OUTPUT_PATH_DISPLAY}")

if __name__ == "__main__":
    main()
