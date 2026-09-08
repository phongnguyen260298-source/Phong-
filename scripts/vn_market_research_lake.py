from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from vnstock_data import Finance, Listing, Macro, Quote, Reference, Trading

TARGET_START = "2014-01-01"
TARGET_END = "2025-12-31"
LOOKBACK_START = "2013-01-01"  # needed for lags / 2014Q1 averages and YoY growth
EXCHANGES = {"HOSE", "HSX", "HNX", "UPCOM"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(x: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def save_df(df: pd.DataFrame | None, path: Path) -> int:
    if df is None or df.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["retrieval_utc"] = utcnow()
    df.to_parquet(path, index=False, compression="zstd")
    return len(df)


def retry(fn: Callable[[], pd.DataFrame], label: str, attempts: int = 4, base_sleep: float = 1.5):
    last = None
    for k in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            print(f"[retry {k}/{attempts}] {label}: {type(exc).__name__}: {exc}", flush=True)
            if k < attempts:
                time.sleep(min(12.0, base_sleep * (2 ** (k - 1))))
    raise last


def normalize_symbol_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol"])
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    symbol_col = next((c for c in x.columns if c.lower() in {"symbol", "ticker", "code"}), None)
    if not symbol_col:
        return pd.DataFrame(columns=["symbol"])
    x = x.rename(columns={symbol_col: "symbol"})
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x = x[x["symbol"].str.match(r"^[A-Z0-9]{3,10}$", na=False)].copy()
    x[f"seen_{source.lower()}"] = True
    return x


def build_universe(out: Path) -> pd.DataFrame:
    frames = []

    # VCI: best current exchange + ICB metadata.
    try:
        vci = Listing(source="VCI").symbols_by_exchange()
        vci = normalize_symbol_frame(vci, "VCI")
        if not vci.empty:
            frames.append(vci)
    except Exception as exc:
        print(f"[universe] VCI symbols_by_exchange failed: {exc}", flush=True)

    # VND: when available it carries status, listed_date and delisted_date.
    try:
        vnd = Listing(source="VND").all_symbols()
        vnd = normalize_symbol_frame(vnd, "VND")
        if not vnd.empty:
            frames.append(vnd)
    except Exception as exc:
        print(f"[universe] VND all_symbols unavailable: {exc}", flush=True)

    # KBS fallback/recall source.
    try:
        kbs = Listing(source="KBS").symbols_by_exchange(get_all=True)
        kbs = normalize_symbol_frame(kbs, "KBS")
        if not kbs.empty:
            frames.append(kbs)
    except Exception as exc:
        print(f"[universe] KBS symbols_by_exchange unavailable: {exc}", flush=True)

    if not frames:
        raise RuntimeError("Unable to build symbol universe from VNStock listing sources")

    # Outer-merge all source metadata by symbol, suffixing collisions.
    u = frames[0]
    for idx, nxt in enumerate(frames[1:], start=2):
        common = [c for c in u.columns if c in nxt.columns and c != "symbol"]
        nxt = nxt.rename(columns={c: f"{c}_src{idx}" for c in common})
        u = u.merge(nxt, on="symbol", how="outer")

    # Consolidate exchange/type/date fields if they exist under source suffixes.
    def coalesce_cols(names: list[str]):
        candidates = [c for c in u.columns if c.lower() in names or any(c.lower().startswith(n + "_") for n in names)]
        if not candidates:
            return pd.Series([None] * len(u), index=u.index)
        s = u[candidates[0]]
        for c in candidates[1:]:
            s = s.where(s.notna() & s.astype(str).ne(""), u[c])
        return s

    u["exchange_best"] = coalesce_cols(["exchange"])
    u["type_best"] = coalesce_cols(["type", "asset_type"])
    u["listed_date_best"] = coalesce_cols(["listed_date"])
    u["delisted_date_best"] = coalesce_cols(["delisted_date"])
    u["status_best"] = coalesce_cols(["status"])

    # Prefer equities and Vietnamese equity exchanges when the metadata exists.
    ex = u["exchange_best"].astype(str).str.upper()
    typ = u["type_best"].astype(str).str.upper()
    exchange_ok = ex.isin(EXCHANGES) | ex.isin({"NAN", "NONE", ""})
    type_ok = typ.isin({"STOCK", "EQUITY", "COMMON", "NAN", "NONE", ""})
    u = u[exchange_ok & type_ok].copy()

    # Remove firms whose known listing interval cannot overlap 2014-2025.
    listed = pd.to_datetime(u["listed_date_best"], errors="coerce")
    delisted = pd.to_datetime(u["delisted_date_best"], errors="coerce")
    u = u[(listed.isna() | (listed <= pd.Timestamp(TARGET_END))) &
          (delisted.isna() | (delisted >= pd.Timestamp(TARGET_START)))].copy()

    u = u.sort_values("symbol").drop_duplicates("symbol", keep="first").reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    u.to_csv(out / "universe.csv", index=False, encoding="utf-8-sig")
    u.to_parquet(out / "universe.parquet", index=False)
    return u


def finance_call(symbol: str, report: str, period: str = "quarter") -> pd.DataFrame:
    f = Finance(source="VCI", symbol=symbol)
    fn = getattr(f, report)
    argsets = [
        dict(period=period, lang="vi", drop_empty=False, format="long"),
        dict(period=period, drop_empty=False, format="long"),
        dict(period=period, format="long"),
    ]
    last = None
    for kwargs in argsets:
        try:
            return fn(**kwargs)
        except TypeError as exc:
            last = exc
    raise last


def extract_core(symbol: str, root: Path, manifest: list[dict], errors: list[dict], delay: float):
    # Preserve ALL semantic IDs. Variable selection happens later.
    for report in ("balance_sheet", "income_statement", "cash_flow", "ratio"):
        label = f"{symbol}:{report}:quarter"
        try:
            df = retry(lambda r=report: finance_call(symbol, r, "quarter"), label)
            path = root / "finance_quarterly" / report / f"{symbol}.parquet"
            n = save_df(df, path)
            manifest.append({"symbol": symbol, "dataset": report, "periodicity": "quarter", "rows": n, "status": "OK" if n else "EMPTY"})
        except Exception as exc:
            errors.append({"symbol": symbol, "dataset": report, "error_type": type(exc).__name__, "error": str(exc)})
        time.sleep(delay)

    # Daily adjusted OHLCV: market outcomes, liquidity, volatility, event studies.
    try:
        q = Quote(source="VCI")
        px = retry(lambda: q.history(symbol=symbol, start=LOOKBACK_START, end=TARGET_END, interval="1D"), f"{symbol}:price")
        n = save_df(px, root / "price_daily" / f"{symbol}.parquet")
        manifest.append({"symbol": symbol, "dataset": "price_daily", "periodicity": "daily", "rows": n, "status": "OK" if n else "EMPTY"})
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "price_daily", "error_type": type(exc).__name__, "error": str(exc)})
    time.sleep(delay)

    # Company info is a retrieval-date snapshot: useful as metadata, not historical exposure unless verified.
    try:
        ref = Reference()
        info = retry(lambda: ref.company(symbol).info(), f"{symbol}:company_info")
        n = save_df(info, root / "company_info" / f"{symbol}.parquet")
        manifest.append({"symbol": symbol, "dataset": "company_info", "periodicity": "snapshot", "rows": n, "status": "OK" if n else "EMPTY"})
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "company_info", "error_type": type(exc).__name__, "error": str(exc)})


def extract_notes(symbol: str, root: Path, manifest: list[dict], errors: list[dict], delay: float):
    # Annual notes are the main source for predetermined exposure measures:
    # export/import, FX, geographic segments, debt currency, materials, software/R&D, etc.
    try:
        df = retry(lambda: finance_call(symbol, "note", "year"), f"{symbol}:note:year")
        n = save_df(df, root / "notes_annual" / f"{symbol}.parquet")
        manifest.append({"symbol": symbol, "dataset": "note", "periodicity": "year", "rows": n, "status": "OK" if n else "EMPTY"})
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "note_year", "error_type": type(exc).__name__, "error": str(exc)})
    time.sleep(delay)

    try:
        ref = Reference()
        tasks = {
            "events": lambda: ref.company(symbol).events(),
            "shareholders_summary": lambda: ref.company(symbol).shareholders(mode="summary"),
            "subsidiaries": lambda: ref.company(symbol).subsidiaries(filter_by="all"),
        }
        for name, fn in tasks.items():
            try:
                df = retry(fn, f"{symbol}:{name}", attempts=3)
                n = save_df(df, root / "reference" / name / f"{symbol}.parquet")
                manifest.append({"symbol": symbol, "dataset": name, "periodicity": "mixed_or_snapshot", "rows": n, "status": "OK" if n else "EMPTY"})
            except Exception as exc:
                errors.append({"symbol": symbol, "dataset": name, "error_type": type(exc).__name__, "error": str(exc)})
            time.sleep(delay)
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "reference_init", "error_type": type(exc).__name__, "error": str(exc)})

    # Filing index only; do not download thousands of PDFs yet.
    try:
        from vnstock_data import Fundamental
        fun = Fundamental()
        filing = retry(lambda: fun.equity(symbol).filing(), f"{symbol}:filing", attempts=3)
        n = save_df(filing, root / "filings" / f"{symbol}.parquet")
        manifest.append({"symbol": symbol, "dataset": "filing", "periodicity": "event", "rows": n, "status": "OK" if n else "EMPTY"})
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "filing", "error_type": type(exc).__name__, "error": str(exc)})


def extract_trading(symbol: str, root: Path, manifest: list[dict], errors: list[dict], delay: float):
    # Availability may start later than 2014; preserve whatever the source returns and log coverage.
    try:
        tr = Trading(symbol=symbol, source="VCI")
        tasks = {
            "foreign_trade": lambda: tr.foreign_trade(start=TARGET_START, end=TARGET_END),
            "prop_trade": lambda: tr.prop_trade(start=TARGET_START, end=TARGET_END),
        }
        for name, fn in tasks.items():
            try:
                df = retry(fn, f"{symbol}:{name}", attempts=3)
                n = save_df(df, root / "trading" / name / f"{symbol}.parquet")
                manifest.append({"symbol": symbol, "dataset": name, "periodicity": "daily", "rows": n, "status": "OK" if n else "EMPTY"})
            except Exception as exc:
                errors.append({"symbol": symbol, "dataset": name, "error_type": type(exc).__name__, "error": str(exc)})
            time.sleep(delay)
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "trading_init", "error_type": type(exc).__name__, "error": str(exc)})


def extract_macro(root: Path, manifest: list[dict], errors: list[dict]):
    macro_root = root / "macro"
    macro_root.mkdir(parents=True, exist_ok=True)
    mac = Macro()
    tasks = {
        "gdp_quarter": lambda: mac.economy().gdp(start="2013-01-01", end=TARGET_END, period="quarter"),
        "cpi_quarter": lambda: mac.economy().cpi(start="2013-01-01", end=TARGET_END, period="quarter"),
        "industry_prod_quarter": lambda: mac.economy().industry_prod(start="2013-01-01", end=TARGET_END, period="quarter"),
        "import_export_quarter": lambda: mac.economy().import_export(start="2013-01-01", end=TARGET_END, period="quarter"),
        "fdi_quarter": lambda: mac.economy().fdi(start="2013-01-01", end=TARGET_END, period="quarter"),
        "exchange_rate": lambda: mac.currency().exchange_rate(start="2013-01-01", end=TARGET_END, period="quarter"),
        "interest_rate": lambda: mac.currency().interest_rate(start="2013-01-01", end=TARGET_END, period="quarter", format="long"),
        "oil_crude": lambda: mac.commodity().oil_crude(start="2013-01-01", end=TARGET_END),
        "steel_global": lambda: mac.commodity().steel(market="GLOBAL", start="2013-01-01", end=TARGET_END),
        "steel_vn": lambda: mac.commodity().steel(market="VN", start="2013-01-01", end=TARGET_END),
        "iron_ore": lambda: mac.commodity().iron_ore(start="2013-01-01", end=TARGET_END),
        "fertilizer_ure": lambda: mac.commodity().fertilizer_ure(start="2013-01-01", end=TARGET_END),
    }
    for name, fn in tasks.items():
        try:
            df = retry(fn, f"macro:{name}", attempts=3)
            n = save_df(df, macro_root / f"{name}.parquet")
            manifest.append({"symbol": "MACRO", "dataset": name, "periodicity": "mixed", "rows": n, "status": "OK" if n else "EMPTY"})
        except Exception as exc:
            errors.append({"symbol": "MACRO", "dataset": name, "error_type": type(exc).__name__, "error": str(exc)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["universe", "core", "notes", "trading", "macro"], default="universe")
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--symbol", default="")
    p.add_argument("--out", default="output/vn_research_lake")
    p.add_argument("--delay", type=float, default=float(os.getenv("VN_RESEARCH_DELAY", "0.25")))
    args = p.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    universe = build_universe(root)

    summary = {
        "mode": args.mode,
        "target_start": TARGET_START,
        "target_end": TARGET_END,
        "lookback_start": LOOKBACK_START,
        "universe_n": int(len(universe)),
        "batch_index": args.batch_index,
        "batch_size": args.batch_size,
        "started_utc": utcnow(),
    }

    if args.symbol:
        symbols = [args.symbol.upper().strip()]
    else:
        start = args.batch_index * args.batch_size
        stop = start + args.batch_size
        symbols = universe["symbol"].tolist()[start:stop]

    summary["batch_symbols_n"] = len(symbols)
    summary["batch_symbols"] = symbols
    summary["total_batches"] = int(math.ceil(len(universe) / max(args.batch_size, 1)))

    manifest: list[dict] = []
    errors: list[dict] = []

    if args.mode == "universe":
        pass
    elif args.mode == "macro":
        extract_macro(root, manifest, errors)
    else:
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] {symbol} mode={args.mode}", flush=True)
            if args.mode == "core":
                extract_core(symbol, root, manifest, errors, args.delay)
            elif args.mode == "notes":
                extract_notes(symbol, root, manifest, errors, args.delay)
            elif args.mode == "trading":
                extract_trading(symbol, root, manifest, errors, args.delay)

    mf = pd.DataFrame(manifest)
    er = pd.DataFrame(errors)
    mf.to_csv(root / "manifest.csv", index=False, encoding="utf-8-sig")
    er.to_csv(root / "errors.csv", index=False, encoding="utf-8-sig")
    summary.update({
        "manifest_rows": int(len(mf)),
        "errors": int(len(er)),
        "finished_utc": utcnow(),
    })
    (root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN_SUMMARY_BEGIN")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("RUN_SUMMARY_END")


if __name__ == "__main__":
    main()
