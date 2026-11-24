import csv
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "csv_transformed"
OUTPUT_DIR = BASE_DIR / "data" / "backtest"
OUTPUT_FILE = "portfolio_combinations.csv"
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_FILE
OUTPUT_PATH_DISPLAY = f"{BASE_DIR.name}/{OUTPUT_PATH.relative_to(BASE_DIR)}"

# Portfolio sleeve -> input file. The order defines the columns of the weight matrix
SECURITIES = (
    ("largecap", "Nifty 100"),
    ("midcap", "NIFTY MIDCAP 150"),
    ("smallcap", "NIFTY SMLCAP 250"),
    ("liquidcase", "NIFTY 1D RATE INDEX"),
    ("ltgiltbees", "Nifty GS 8 13Yr"),
    ("realty", "Nifty REITs & Realty"),
    ("goldbees", "GOLDBEES"),
    ("goldietf", "GOLDIETF"),
)
RISK_FREE = "liquidcase"

# Weights are in percent. Equity/debt securities move in SPLIT_STEP, realty and gold in CLASS_STEP.
# Gold takes whatever is left and is always split equally between GOLDBEES and GOLDIETF
CLASS_STEP = 1
SPLIT_STEP = 10
TOTAL_WEIGHT = 100

TRADING_DAYS = 252
ROLLING_WINDOWS = {"1y": 1, "3y": 3, "5y": 5}
CHUNK_SIZE = 2000

COLUMNS = (
    "combination_id", "equity", "largecap", "midcap", "smallcap", "debt", "liquidcase",
    "ltgiltbees", "realty", "gold", "goldbees", "goldietf",
    "cagr", "volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown", "correlation",
    "rolling_returns_1y", "rolling_returns_3y", "rolling_returns_5y",
    "positive_months", "positive_months_pct",
)

def read_series(name):
    with open(INPUT_DIR / f"{name}.csv", encoding="utf-8", newline="") as f:
        rows = [(row["date"], float(row["value"])) for row in csv.DictReader(f)]
    if not rows:
        raise ValueError()
    return dict(rows)

def read_prices():
    series = [read_series(name) for _, name in SECURITIES]

    # Use only the period covered by every instrument, then forward fill holiday mismatches
    start = max(min(s) for s in series)
    end = min(max(s) for s in series)
    dates = sorted({d for s in series for d in s if start <= d <= end})
    if len(dates) < 2:
        raise ValueError()

    prices = np.empty((len(dates), len(series)))
    for col, s in enumerate(series):
        last = None
        for row, date in enumerate(dates):
            last = s.get(date, last)
            prices[row, col] = last
    if np.isnan(prices).any() or (prices <= 0).any():
        raise ValueError()
    return dates, prices

def build_weights():
    # Columns follow SECURITIES; every row sums to TOTAL_WEIGHT
    rows = []
    units = TOTAL_WEIGHT // SPLIT_STEP
    for large in range(units + 1):
        for mid in range(units + 1 - large):
            for small in range(units + 1 - large - mid):
                for liquid in range(units + 1 - large - mid - small):
                    for gilt in range(units + 1 - large - mid - small - liquid):
                        used = (large + mid + small + liquid + gilt) * SPLIT_STEP
                        for realty in range(0, TOTAL_WEIGHT - used + 1, CLASS_STEP):
                            gold = TOTAL_WEIGHT - used - realty
                            rows.append((
                                large * SPLIT_STEP, mid * SPLIT_STEP, small * SPLIT_STEP,
                                liquid * SPLIT_STEP, gilt * SPLIT_STEP, realty,
                                gold / 2, gold / 2,
                            ))
    if not rows:
        raise ValueError()
    return np.array(rows, dtype=float)

def month_groups(dates):
    groups = []
    for i, date in enumerate(dates):
        if groups and dates[groups[-1][0]][:7] == date[:7]:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [np.array(g) for g in groups]

def portfolio_values(weights, prices):
    # Buy and hold: weights apply on day one and then drift with the prices
    return (weights / TOTAL_WEIGHT) @ (prices / prices[0]).T

def rolling_mean(values, years):
    window = years * TRADING_DAYS
    if values.shape[1] <= window:
        return np.full(len(values), np.nan)
    growth = values[:, window:] / values[:, :-window]
    return (growth ** (1 / years) - 1).mean(axis=1)

def compute_metrics(weights, values, dates, groups_monthly, security_corr, risk_free):
    returns = values[:, 1:] / values[:, :-1] - 1
    excess = returns - risk_free
    days = (datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])).days

    with np.errstate(divide="ignore", invalid="ignore"):
        cagr = values[:, -1] ** (365.25 / days) - 1
        volatility = returns.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
        sharpe = excess.mean(axis=1) / excess.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
        downside = np.sqrt((np.minimum(excess, 0) ** 2).mean(axis=1))
        sortino = excess.mean(axis=1) / downside * np.sqrt(TRADING_DAYS)
        drawdown = (values / np.maximum.accumulate(values, axis=1) - 1).min(axis=1)

        # Weight-adjusted internal correlation: average pairwise correlation between the held
        # securities, weighted by w_i * w_j. A single-security portfolio is fully correlated (1)
        fractions = weights / TOTAL_WEIGHT
        self_weight = (fractions ** 2).sum(axis=1)
        pair_weight = 1 - self_weight
        pair_corr = ((fractions @ security_corr) * fractions).sum(axis=1) - self_weight
        correlation = np.where(pair_weight > 1e-12, pair_corr / pair_weight, 1.0)

    month_end = np.array([g[-1] for g in groups_monthly])
    month_values = np.concatenate([values[:, :1], values[:, month_end]], axis=1)
    monthly_returns = month_values[:, 1:] / month_values[:, :-1] - 1
    positive_months = (monthly_returns > 0).sum(axis=1)

    metrics = {
        "cagr": cagr * 100,
        "volatility": volatility * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": drawdown * 100,
        "correlation": correlation,
        "positive_months": positive_months,
        "positive_months_pct": positive_months / monthly_returns.shape[1] * 100,
    }
    for key, years in ROLLING_WINDOWS.items():
        metrics[f"rolling_returns_{key}"] = rolling_mean(values, years) * 100
    return metrics

def format_number(value):
    if isinstance(value, (int, np.integer)):
        return str(value)
    if np.isnan(value):
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")

def write_combinations(weights, dates, prices, groups_monthly):
    keys = [key for key, _ in SECURITIES]
    daily = prices[1:] / prices[:-1] - 1
    risk_free = daily[:, keys.index(RISK_FREE)]
    security_corr = np.corrcoef(daily, rowvar=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for start in range(0, len(weights), CHUNK_SIZE):
            chunk = weights[start:start + CHUNK_SIZE]
            values = portfolio_values(chunk, prices)
            metrics = compute_metrics(chunk, values, dates, groups_monthly, security_corr, risk_free)
            for i, w in enumerate(chunk):
                large, mid, small, liquid, gilt, realty, goldbees, goldietf = w
                weight_cells = (
                    large + mid + small, large, mid, small, liquid + gilt, liquid, gilt,
                    realty, goldbees + goldietf, goldbees, goldietf,
                )
                writer.writerow(
                    [start + i + 1]
                    + [format_number(x) for x in weight_cells]
                    + [format_number(metrics[c][i]) for c in COLUMNS[12:]]
                )

def main():
    try:
        dates, prices = read_prices()
        weights = build_weights()
        write_combinations(weights, dates, prices, month_groups(dates))
    except Exception as error:
        print(f"ERROR : {type(error).__name__}")
        return

    print(f"Built {len(weights)} combinations from {dates[0]} to {dates[-1]} and saved to {OUTPUT_PATH_DISPLAY}")

if __name__ == "__main__":
    main()
