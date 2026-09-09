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
from vnstock_data import Finance

START_YEAR = 2013
END_YEAR = 2025
REPORTS = ("balance_sheet", "income_statement", "cash_flow", "ratio")


class SlidingRateLimiter:
    def __init__(self, max_calls=90, window_seconds=60.0):
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

    def cooldown(self, seconds=65.0):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + float(seconds))


LIMITER = None


def parse_year(v):
    m = re.search(r"(19|20)\d{2}", str(v))
    return int(m.group(0)) if m else None


def filter_years(df):
    if df is None or df.empty or "period" not in df.columns:
        return df
    years = df["period"].map(parse_year)
    return df[years.notna() & years.between(START_YEAR, END_YEAR)].copy()


def looks_rate_limited(exc):
    s = str(exc).lower()
    return "rate limit" in s or "requests/min" in s or "requests/phút" in s or "180/180" in s


def retry(fn, label, attempts=4):
    last = None
    for k in range(attempts):
        try:
            if LIMITER is not None:
                LIMITER.wait()
            return fn()
        except SystemExit as exc:
            last = RuntimeError(f"VNStock SystemExit during {label}: {exc}")
            if LIMITER is not None:
                LIMITER.cooldown()
            print(f"[retry {k+1}/{attempts}] {label}: SystemExit: {exc}", flush=True)
        except Exception as exc:
            last = exc
            if looks_rate_limited(exc) and LIMITER is not None:
                LIMITER.cooldown()
            print(f"[retry {k+1}/{attempts}] {label}: {type(exc).__name__}: {exc}", flush=True)
        if k + 1 < attempts:
            time.sleep(min(12.0, 1.5 * (2 ** k)))
    raise last


def call_report(finance, report, period="quarter"):
    fn = getattr(finance, report)
    argsets = [
        dict(period=period, lang="vi", format="long", drop_empty=False),
        dict(period=period, lang="vi", format="long"),
        dict(period=period, lang="vi"),
        dict(period=period),
    ]
    last = None
    for kwargs in argsets:
        try:
            return fn(**kwargs)
        except TypeError as exc:
            last = exc
    raise last


def fetch_with_priority(symbol, report, period):
    attempts = []
    for source in ("VCI", "MAS"):
        try:
            fin = Finance(source=source, symbol=symbol)
            df = retry(lambda f=fin: call_report(f, report, period), f"{symbol}:{source}:{report}:{period}")
            df = filter_years(df)
            if df is not None and not df.empty:
                return df, source, attempts
            attempts.append({"source": source, "status": "EMPTY"})
        except Exception as exc:
            attempts.append({"source": source, "status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)})
    return pd.DataFrame(), None, attempts


def coverage(df):
    if df is None or df.empty or "period" not in df.columns:
        return None, None, 0
    vals = df["period"].dropna().astype(str).drop_duplicates().tolist()
    vals = sorted(vals, key=lambda s: (parse_year(s) or -1, s))
    if not vals:
        return None, None, 0
    return vals[0], vals[-1], len(vals)


def item_count(df):
    if df is None or df.empty:
        return 0
    for c in ("item_id", "id", "code"):
        if c in df.columns:
            return int(df[c].dropna().astype(str).nunique())
    return 0


def save_df(df, path, source):
    if df is None or df.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    x = df.copy()
    x["retrieval_source"] = source
    x["retrieval_utc"] = pd.Timestamp.utcnow().isoformat()
    x.to_parquet(path, index=False, compression="zstd")
    return len(x)


def extract_symbol(symbol, root, period):
    manifest, errors = [], []
    for report in REPORTS:
        try:
            df, source, attempts = fetch_with_priority(symbol, report, period)
            rows = save_df(df, root / f"fundamental_{period}" / report / f"{symbol}.parquet", source or "NONE")
            first_p, last_p, periods = coverage(df)
            manifest.append({
                "symbol": symbol,
                "report": report,
                "period_type": period,
                "rows": rows,
                "status": "OK" if rows else "EMPTY",
                "source_selected": source,
                "source_priority": "VCI>MAS",
                "first_period": first_p,
                "last_period": last_p,
                "period_count": periods,
                "item_count": item_count(df),
                "attempts_json": json.dumps(attempts, ensure_ascii=False),
            })
            if rows == 0:
                errors.append({"symbol": symbol, "report": report, "period_type": period, "error_type": "NO_DATA", "error": json.dumps(attempts, ensure_ascii=False)})
        except Exception as exc:
            errors.append({"symbol": symbol, "report": report, "period_type": period, "error_type": type(exc).__name__, "error": str(exc)})
    return manifest, errors


def main():
    global LIMITER
    p = argparse.ArgumentParser()
    p.add_argument("--cohort-file", required=True)
    p.add_argument("--chunk-index", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--rpm", type=int, default=90)
    p.add_argument("--period", choices=["quarter", "year"], default="quarter")
    p.add_argument("--symbols", default="", help="Optional comma-separated explicit symbols for pilot mode")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    LIMITER = SlidingRateLimiter(args.rpm)
    cohort = pd.read_csv(args.cohort_file)
    all_symbols = cohort["symbol"].dropna().astype(str).str.upper().drop_duplicates().sort_values().tolist()
    if args.symbols.strip():
        wanted = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        symbols = [s for s in wanted if s in set(all_symbols)]
    else:
        start = args.chunk_index * args.chunk_size
        symbols = all_symbols[start:start + args.chunk_size]

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest, errors = [], []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract_symbol, s, root, args.period): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                m, e = fut.result()
                manifest.extend(m)
                errors.extend(e)
            except BaseException as exc:
                errors.append({"symbol": s, "report": "worker", "period_type": args.period, "error_type": type(exc).__name__, "error": str(exc)})
            done += 1
            print(f"[{done}/{len(symbols)}] {s}", flush=True)

    mdf = pd.DataFrame(manifest)
    edf = pd.DataFrame(errors)
    mdf.to_csv(root / "manifest.csv", index=False, encoding="utf-8-sig")
    edf.to_csv(root / "errors.csv", index=False, encoding="utf-8-sig")

    selected_vci = int((mdf.get("source_selected") == "VCI").sum()) if len(mdf) else 0
    selected_mas = int((mdf.get("source_selected") == "MAS").sum()) if len(mdf) else 0
    summary = {
        "version": "v4.2-vci-priority",
        "period": args.period,
        "symbols_n": len(symbols),
        "reports_expected": len(symbols) * len(REPORTS),
        "manifest_rows": len(mdf),
        "errors": len(edf),
        "vci_selected": selected_vci,
        "mas_fallback_selected": selected_mas,
        "elapsed_seconds": round(time.time() - t0, 2),
        "source_policy": "VCI primary; MAS fallback only on VCI error/empty",
    }
    (root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
