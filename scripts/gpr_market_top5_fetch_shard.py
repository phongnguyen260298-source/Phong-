from __future__ import annotations

import os
import random
import time
from pathlib import Path

import pandas as pd
from vnstock_data import Quote

SHARD = int(os.getenv("SHARD", "0"))
N_SHARDS = int(os.getenv("N_SHARDS", "4"))
MIN_REQUEST_INTERVAL = float(os.getenv("REQUEST_INTERVAL", "1.6"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
OUT = Path("data/gpr_market_top5_v4/shards")
OUT.mkdir(parents=True, exist_ok=True)

base = pd.read_csv("data/research_cohorts/vci578_preexposure.csv")
base["symbol"] = base["symbol"].astype(str).str.upper().str.strip()
all_symbols = base["symbol"].dropna().drop_duplicates().tolist()
symbols = [s for i, s in enumerate(all_symbols) if i % N_SHARDS == SHARD]

START = "2020-01-01"
END = "2025-12-31"
_next_request_at = 0.0


def pace_request() -> None:
    global _next_request_at
    now = time.monotonic()
    if now < _next_request_at:
        time.sleep(_next_request_at - now)
    _next_request_at = time.monotonic() + MIN_REQUEST_INTERVAL + random.uniform(0.0, 0.08)


def normalize_history(sym: str, df: pd.DataFrame) -> pd.DataFrame:
    tc = next((c for c in ["time", "date", "trading_date"] if c in df.columns), None)
    cc = next((c for c in ["close", "Close", "close_price"] if c in df.columns), None)
    if tc is None or cc is None:
        raise ValueError(f"Unexpected schema={list(df.columns)}")
    z = df[[tc, cc]].rename(columns={tc: "date", cc: "close"}).copy()
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.normalize()
    z["close"] = pd.to_numeric(z["close"], errors="coerce")
    z = z.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    z["symbol"] = sym
    return z[["symbol", "date", "close"]]


def fetch_one(sym: str):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            pace_request()
            q = Quote(source="VCI", symbol=sym)
            df = q.history(start=START, end=END, interval="1D")
            if df is None or df.empty:
                return sym, None, "EMPTY"
            z = normalize_history(sym, df)
            if z.empty:
                return sym, None, "EMPTY_AFTER_NORMALIZE"
            return sym, z, None
        except BaseException as exc:
            last = exc
            msg = str(exc).lower()
            is_rate = any(k in msg for k in ["limit", "rate", "429", "too many request"])
            if is_rate:
                wait = min(90.0, 12.0 * (2 ** attempt)) + random.uniform(0.5, 3.0)
            else:
                wait = min(30.0, 3.0 * (attempt + 1)) + random.uniform(0.2, 1.0)
            print(f"shard {SHARD} retry {attempt + 1}/{MAX_RETRIES} {sym}: {type(exc).__name__}: {exc}; sleep={wait:.1f}s", flush=True)
            time.sleep(wait)
    return sym, None, f"{type(last).__name__}:{last}"


frames = []
errors = []
coverage = []
started = time.monotonic()

for i, sym in enumerate(symbols, 1):
    sym, z, err = fetch_one(sym)
    if z is not None:
        frames.append(z)
        coverage.append({"symbol": sym, "rows": len(z), "min_date": z["date"].min(), "max_date": z["date"].max()})
    else:
        errors.append({"symbol": sym, "error": err})
    elapsed = max(time.monotonic() - started, 1e-9)
    rpm = i / elapsed * 60.0
    print(f"shard {SHARD} {i}/{len(symbols)} {sym} rows={0 if z is None else len(z)} ok={z is not None} avg_rpm={rpm:.1f}", flush=True)

if frames:
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    prices.to_csv(OUT / f"price_shard_{SHARD}.csv.gz", index=False, compression="gzip")

pd.DataFrame(errors, columns=["symbol", "error"]).to_csv(OUT / f"errors_shard_{SHARD}.csv", index=False)
pd.DataFrame(coverage, columns=["symbol", "rows", "min_date", "max_date"]).to_csv(OUT / f"coverage_shard_{SHARD}.csv", index=False)

elapsed = time.monotonic() - started
print({"shard": SHARD, "cohort_total": len(all_symbols), "target": len(symbols), "ok": len(frames), "errors": len(errors), "elapsed_sec": round(elapsed, 1), "avg_requests_per_min": round(len(symbols) / max(elapsed, 1e-9) * 60.0, 1)}, flush=True)
