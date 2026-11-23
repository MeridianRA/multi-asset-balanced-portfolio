import json
import time
import shutil
from datetime import date, timedelta
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent.parent

SYMBOL_FILE = "etf-symbol.txt"
SYMBOL_PATH = BASE_DIR / "data" / "symbols" / SYMBOL_FILE
OUTPUT_DIR = BASE_DIR / "data" / "json" / "etf"
OUTPUT_PATH_DISPLAY = f"{BASE_DIR.name}/{OUTPUT_DIR.relative_to(BASE_DIR)}"

START_DATE = date(2021, 7, 1)
END_DATE = date(2026, 11, 22)
CHUNK_MONTHS = 3

REQUEST_SLEEP = 0.5

BASE_URL = "https://www.nseindia.in"
API_URL = f"{BASE_URL}/api/historicalOR/generateSecurityWiseHistoricalData"

HEADERS = {
    "accept": "*/*",
    "accept-language": "en-GB,en;q=0.7",
    "referer": f"{BASE_URL}/report-detail/eq_security",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/156.0.0.0 Safari/537.36"
    ),
}

def read_symbols():
    return [
        s.strip()
        for s in SYMBOL_PATH.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]

def build_chunks():
    chunks = []
    start = START_DATE
    while start <= END_DATE:
        month_index = start.year * 12 + start.month - 1 + CHUNK_MONTHS
        next_start = date(month_index // 12, month_index % 12 + 1, 1)
        chunks.append((start, min(next_start - timedelta(days=1), END_DATE)))
        start = next_start
    return chunks

CHUNKS = build_chunks()

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

def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(BASE_URL, timeout=10).raise_for_status()
    time.sleep(1)
    return session

def fetch_chunk(session, symbol, start, end):
    params = {
        "from": start.strftime("%d-%m-%Y"),
        "to": end.strftime("%d-%m-%Y"),
        "symbol": symbol,
        "type": "priceVolume",
        "series": "ALL",
    }
    resp = session.get(API_URL, params=params, timeout=15)
    resp.raise_for_status()
    records = resp.json()["data"]
    if not isinstance(records, list):
        raise ValueError()
    return records

def fetch_data(session, symbol):
    rows = []
    for index, (start, end) in enumerate(CHUNKS):
        if index:
            time.sleep(REQUEST_SLEEP)
        rows.extend(fetch_chunk(session, symbol, start, end))
    if not rows:
        raise ValueError()
    return rows

def process_symbol(session, symbol, label):
    try:
        rows = fetch_data(session, symbol)
        (OUTPUT_DIR / f"{symbol}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (requests.RequestException, ValueError, KeyError, TypeError, OSError) as error:
        print(f"{label}->  ERROR : {type(error).__name__}")
        return False
    print(f"{label}->  SUCCESS")
    return True

def main():
    try:
        symbols = read_symbols()
        if not symbols:
            raise ValueError()
        session = create_session()
        prepare_output_dir()
    except Exception as error:
        print(f"ERROR : {type(error).__name__}")
        return

    total = len(symbols)
    width = max(len(s) for s in symbols) + 2
    print(f"\nDownloading data for {total} ETFs ...\n")

    success = 0
    for symbol in symbols:
        label = f"{symbol:<{width}}"
        success += process_symbol(session, symbol, label)
        time.sleep(REQUEST_SLEEP)

    print(f"\nDownloaded data for {success}/{total} ETFs")
    print(f"Saved to : {OUTPUT_PATH_DISPLAY}")

if __name__ == "__main__":
    main()
