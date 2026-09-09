from __future__ import annotations

import argparse
import json
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from vnstock_data import Fundamental, Market, Reference

LOOKBACK_START = "2013-01-01"
TARGET_END = "2025-12-31"
START_YEAR = 2013
END_YEAR = 2025


class SlidingRateLimiter:
    """Process-wide limiter for top-level VNStock calls.

    Bronze is currently 180 requests/minute. We intentionally target below that
    because one high-level vnstock_data call can internally trigger more than
    one provider request. The limiter is shared by all symbol workers.
    """

    def __init__(self, max_calls: int = 100, window_seconds: float = 60.0):
        self.max_calls = max(1, int(max_calls))
        self.window = float(window_seconds)
        self.calls = deque()
        self.lock = threading.Lock()
        self.blocked_until = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                if now < self.blocked_until:
                    wait_for = self.blocked_until - now
                else:
                    cutoff = now - self.window
                    while self.calls and self.calls[0] <= cutoff:
                        self.calls.popleft()
                    if len(self.calls) < self.max_calls:
                        self.calls.append(now)
                        return
                    wait_for = max(0.05, self.calls[0] + self.window - now)
            time.sleep(min(wait_for, 2.0))

    def cooldown(self, seconds: float = 65.0):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)


LIMITER: SlidingRateLimiter | None = None


def parse_year(v):
    m = re.search(r"(19|20)\d{2}", str(v))
    return int(m.group(0)) if m else None


def filter_finance(df):
    if df is None or df.empty or "period" not in df.columns:
        return df
    years = df["period"].map(parse_year)
    return df[years.notna() & years.between(START_YEAR, END_YEAR)].copy()


def _looks_rate_limited(exc) -> bool:
    s = str(exc).lower()
    return "rate limit" in s or "180/180" in s or "requests/phút" in s or "requests/min" in s


def retry(fn, label, attempts=4):
    last = None
    for k in range(attempts):
        try:
            if LIMITER is not None:
                LIMITER.wait()
            return fn()
        except SystemExit as exc:
            # vnstock_data may terminate the current call when its server-side
            # minute quota is reached. Convert that into a recoverable retry.
            last = RuntimeError(f"VNStock SystemExit during {label}: {exc}")
            if LIMITER is not None:
                LIMITER.cooldown(65)
            print(f"[retry {k+1}/{attempts}] {label}: SystemExit: {exc}", flush=True)
        except Exception as exc:
            last = exc
            if _looks_rate_limited(exc) and LIMITER is not None:
                LIMITER.cooldown(65)
            print(f"[retry {k+1}/{attempts}] {label}: {type(exc).__name__}: {exc}", flush=True)
        if k + 1 < attempts:
            time.sleep(min(12.0, 1.5 * (2 ** k)))
    raise last


def save_df(df, path, source):
    if df is None or df.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    x = df.copy()
    x["retrieval_source"] = source
    x["retrieval_utc"] = pd.Timestamp.utcnow().isoformat()
    x.to_parquet(path, index=False, compression="zstd")
    return len(x)


def coverage(df):
    if df is None or df.empty:
        return None, None, 0
    if "period" in df.columns:
        vals = df["period"].dropna().astype(str).drop_duplicates().tolist()
        vals = sorted(vals, key=lambda s: (parse_year(s) or -1, s))
        return (vals[0], vals[-1], len(vals)) if vals else (None, None, 0)
    for c in ("time", "date", "trading_date"):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce").dropna()
            if len(s):
                return s.min().strftime("%Y-%m-%d"), s.max().strftime("%Y-%m-%d"), int(s.dt.normalize().nunique())
    return None, None, 0


def fundamental_report(eq, report):
    fn = getattr(eq, report)
    argsets = [
        dict(period="quarter", lang="vi", drop_empty=False, format="long"),
        dict(period="quarter", drop_empty=False, format="long"),
        dict(period="quarter", format="long"),
    ]
    last = None
    for kwargs in argsets:
        try:
            return fn(**kwargs)
        except TypeError as exc:
            last = exc
    raise last


def manifest_row(symbol, dataset, rows, source, df):
    f, l, n = coverage(df)
    return {
        "symbol": symbol,
        "dataset": dataset,
        "rows": rows,
        "status": "OK" if rows else "EMPTY",
        "source": source,
        "first_period": f,
        "last_period": l,
        "period_count": n,
    }


def extract_symbol(symbol: str, root: Path):
    manifest = []
    errors = []

    try:
        eq = Fundamental().equity(symbol)
    except Exception as exc:
        return [], [{"symbol": symbol, "dataset": "fundamental_init", "error_type": type(exc).__name__, "error": str(exc)}]

    for report in ("balance_sheet", "income_statement", "cash_flow", "ratio"):
        dataset = f"fundamental_{report}_quarter"
        try:
            df = retry(lambda r=report: fundamental_report(eq, r), f"{symbol}:{report}")
            df = filter_finance(df)
            rows = save_df(df, root / "fundamental_quarterly" / report / f"{symbol}.parquet", "Fundamental")
            manifest.append(manifest_row(symbol, dataset, rows, "Fundamental", df))
        except Exception as exc:
            errors.append({"symbol": symbol, "dataset": dataset, "error_type": type(exc).__name__, "error": str(exc)})

    # Current vnstock_data Market v3 API: Market().equity(symbol).ohlcv(...).
    # Market().quote(symbols_list) is only for current multi-symbol quote snapshots.
    try:
        px = retry(
            lambda: Market().equity(symbol).ohlcv(start=LOOKBACK_START, end=TARGET_END, interval="1D"),
            f"{symbol}:price",
            attempts=3,
        )
        if px is not None and not px.empty:
            tc = next((c for c in ("time", "date", "trading_date") if c in px.columns), None)
            if tc:
                d = pd.to_datetime(px[tc], errors="coerce")
                px = px[d.between(pd.Timestamp(LOOKBACK_START), pd.Timestamp(TARGET_END))].copy()
        rows = save_df(px, root / "price_daily" / f"{symbol}.parquet", "Market.equity.ohlcv")
        manifest.append(manifest_row(symbol, "price_daily", rows, "Market.equity.ohlcv", px))
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "price_daily", "error_type": type(exc).__name__, "error": str(exc)})

    try:
        info = retry(lambda: Reference().company(symbol).info(), f"{symbol}:company_info", attempts=3)
        rows = save_df(info, root / "company_info" / f"{symbol}.parquet", "Reference")
        manifest.append({"symbol": symbol, "dataset": "company_info", "rows": rows, "status": "OK" if rows else "EMPTY",
                         "source": "Reference", "first_period": None, "last_period": None, "period_count": 0})
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "company_info", "error_type": type(exc).__name__, "error": str(exc)})

    return manifest, errors


def main():
    global LIMITER

    p = argparse.ArgumentParser()
    p.add_argument("--cohort-file", required=True)
    p.add_argument("--chunk-index", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--rpm", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    LIMITER = SlidingRateLimiter(args.rpm, 60.0)

    cohort = pd.read_csv(args.cohort_file)
    symbols = cohort["symbol"].dropna().astype(str).str.upper().drop_duplicates().sort_values().tolist()
    start = args.chunk_index * args.chunk_size
    selected = symbols[start:start + args.chunk_size]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    manifest, errors = [], []
    t0 = time.time()
    print(f"chunk={args.chunk_index} symbols={len(selected)} workers={args.workers} target_rpm={args.rpm}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract_symbol, s, root): s for s in selected}
        done = 0
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                mf, er = fut.result()
                manifest.extend(mf)
                errors.extend(er)
            except BaseException as exc:
                # Preserve progress if a provider raises SystemExit inside a worker.
                errors.append({"symbol": s, "dataset": "worker", "error_type": type(exc).__name__, "error": str(exc)})
            done += 1
            print(f"[{done}/{len(selected)}] {s}", flush=True)

    pd.DataFrame(manifest).to_csv(root / "manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(root / "errors.csv", index=False, encoding="utf-8-sig")
    summary = {
        "version": "v4.1-fast",
        "chunk_index": args.chunk_index,
        "chunk_size": args.chunk_size,
        "workers": args.workers,
        "target_top_level_rpm": args.rpm,
        "symbols_n": len(selected),
        "symbols": selected,
        "manifest_rows": len(manifest),
        "errors": len(errors),
        "elapsed_seconds": round(time.time() - t0, 2),
        "lookback_start": LOOKBACK_START,
        "target_end": TARGET_END,
    }
    (root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
