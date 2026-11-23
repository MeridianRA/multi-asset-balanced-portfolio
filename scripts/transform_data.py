import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

JSON_DIR = BASE_DIR / "data" / "json"
OUTPUT_DIR = BASE_DIR / "data" / "csv_transformed"
OUTPUT_PATH_DISPLAY = f"{BASE_DIR.name}/{OUTPUT_DIR.relative_to(BASE_DIR)}"

CATEGORIES = ("index", "etf")

LAYOUTS = (
    ("HistoricalDate", "%d %b %Y", "CLOSE"),
    ("TRIDate", "%d %b %Y", "Total Returns Index"),
    ("mTIMESTAMP", "%d-%b-%Y", "CH_CLOSING_PRICE"),
)

def prepare_output_dir():
    if OUTPUT_DIR.exists():
        for child in OUTPUT_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        print(f"\nOutput directory exists : {OUTPUT_PATH_DISPLAY}")
        print("Cleared output directory")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated output directory : {OUTPUT_PATH_DISPLAY}")

def read_inputs():
    return [
        json_file
        for category in CATEGORIES
        for json_file in sorted((JSON_DIR / category).glob("*.json"))
    ]

def transform_file(json_file):
    data = json.loads(json_file.read_text(encoding="utf-8"))
    records = data["data"] if isinstance(data, dict) else data
    if not isinstance(records, list) or not records:
        raise ValueError()

    date_column, date_format, value_column = next(
        layout for layout in LAYOUTS if layout[0] in records[0] and layout[2] in records[0]
    )
    rows = [
        {
            "date": datetime.strptime(record[date_column], date_format).date().isoformat(),
            "value": float(record[value_column]),
        }
        for record in records
    ]
    rows.sort(key=lambda r: r["date"])

    with open(OUTPUT_DIR / f"{json_file.stem}.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "value"])
        writer.writeheader()
        writer.writerows(rows)

def process_file(json_file, label):
    try:
        transform_file(json_file)
    except (json.JSONDecodeError, StopIteration, ValueError, KeyError, TypeError, OSError) as error:
        print(f"{label}->  ERROR : {type(error).__name__}")
        return False
    print(f"{label}->  SUCCESS")
    return True

def main():
    try:
        json_files = read_inputs()
        if not json_files:
            raise ValueError()
        prepare_output_dir()
    except Exception as error:
        print(f"ERROR : {type(error).__name__}")
        return

    total = len(json_files)
    width = max(len(f.stem) for f in json_files) + 2
    print(f"\nTransforming data for {total} input files ...\n")

    success = 0
    for json_file in json_files:
        label = f"{json_file.stem:<{width}}"
        success += process_file(json_file, label)

    print(f"\nTransformed data for {success}/{total} input files")
    print(f"Saved to : {OUTPUT_PATH_DISPLAY}")

if __name__ == "__main__":
    main()
