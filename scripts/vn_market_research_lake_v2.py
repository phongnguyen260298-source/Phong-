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
LOOKBACK_START = "2013-01-01"  # one-year lookback for lags / YoY growth
START_YEAR = 2013
END_YEAR = 2025
EXCHANGES = {"HOSE", "HSX", "HNX", "UPCOM"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_df(df: pd.DataFrame | None, path: Path, source: str | None = None) -> int:
    if df is None or df.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    x = df.copy()
    if source is not None:
        x["retrieval_source"] = source
    x["retrieval_utc"] = utcnow()
    x.to_parquet(path, index=False, compression="zstd")
    return len(x)


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


def coalesce_from_frame(u: pd.DataFrame, names: list[str]) -> pd.Series:
    candidates = [
        c for c in u.columns
        if c.lower() in names or any(c.lower().startswith(n + "_") for n in names)
    ]
    if not candidates:
        return pd.Series([None] * len(u), index=u.index)
    s = u[candidates[0]]
    for c in candidates[1:]:
        s = s.where(s.notna() & s.astype(str).str.strip().ne(""), u[c])
    return s


def build_universe(out: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    try:
        vci = normalize_symbol_frame(Listing(source="VCI").symbols_by_exchange(), "VCI")
        if not vci.empty:
            frames.append(vci)
    except Exception as exc:
        print(f"[universe] VCI symbols_by_exchange failed: {exc}", flush=True)

    try:
        vnd = normalize_symbol_frame(Listing(source="VND").all_symbols(), "VND")
        if not vnd.empty:
            frames.append(vnd)
    except Exception as exc:
        print(f"[universe] VND all_symbols unavailable: {exc}", flush=True)

    try:
        kbs = normalize_symbol_frame(Listing(source="KBS").symbols_by_exchange(get_all=True), "KBS")
        if not kbs.empty:
            frames.append(kbs)
    except Exception as exc:
        print(f"[universe] KBS symbols_by_exchange unavailable: {exc}", flush=True)

    if not frames:
        raise RuntimeError("Unable to build symbol universe from VNStock listing sources")

    u = frames[0]
    for idx, nxt in enumerate(frames[1:], start=2):
        common = [c for c in u.columns if c in nxt.columns and c != "symbol"]
        nxt = nxt.rename(columns={c: f"{c}_src{idx}" for c in common})
        u = u.merge(nxt, on="symbol", how="outer")

    u["exchange_raw_best"] = coalesce_from_frame(u, ["exchange"])
    u["type_best"] = coalesce_from_frame(u, ["type", "asset_type"])
    u["listed_date_best"] = coalesce_from_frame(u, ["listed_date"])
    u["delisted_date_raw"] = coalesce_from_frame(u, ["delisted_date"])
    u["status_best"] = coalesce_from_frame(u, ["status"])

    # Canonical exchange naming: HSX and HOSE are the same exchange.
    ex = u["exchange_raw_best"].astype(str).str.upper().str.strip()
    u["exchange_best"] = ex.replace({"HSX": "HOSE"})

    # Separate a true terminal delisting from a prior-market exit caused by a transfer.
    status_norm = u["status_best"].astype(str).str.lower().str.strip()
    raw_delisted = pd.to_datetime(u["delisted_date_raw"], errors="coerce")
    listed = pd.to_datetime(u["listed_date_best"], errors="coerce")
    status_missing = status_norm.isin({"", "nan", "none", "nat"})
    is_delisted = status_norm.str.contains(r"delist|huy|hủy", regex=True, na=False)

    true_delisted = raw_delisted.where(is_delisted | status_missing)
    prior_market_exit = raw_delisted.where(~is_delisted & raw_delisted.notna())

    u["delisted_date_best"] = true_delisted.dt.strftime("%Y-%m-%d")
    u["previous_market_exit_date"] = prior_market_exit.dt.strftime("%Y-%m-%d")
    u["listing_interval_issue_flag"] = (
        raw_delisted.notna() & listed.notna() & (raw_delisted < listed) & ~is_delisted
    )

    typ = u["type_best"].astype(str).str.upper().str.strip()
    exchange_ok = u["exchange_best"].isin({"HOSE", "HNX", "UPCOM"}) | u["exchange_best"].isin({"NAN", "NONE", ""})
    type_ok = typ.isin({"STOCK", "EQUITY", "COMMON", "NAN", "NONE", ""})
    u = u[exchange_ok & type_ok].copy()

    listed2 = pd.to_datetime(u["listed_date_best"], errors="coerce")
    delisted2 = pd.to_datetime(u["delisted_date_best"], errors="coerce")
    u = u[
        (listed2.isna() | (listed2 <= pd.Timestamp(TARGET_END)))
        & (delisted2.isna() | (delisted2 >= pd.Timestamp(TARGET_START)))
    ].copy()

    u = u.sort_values("symbol").drop_duplicates("symbol", keep="first").reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    u.to_csv(out / "universe.csv", index=False, encoding="utf-8-sig")
    u.to_parquet(out / "universe.parquet", index=False)
    return u


def parse_year(x: object) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(x))
    return int(m.group(0)) if m else None


def filter_finance_window(df: pd.DataFrame, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    if df is None or df.empty or "period" not in df.columns:
        return df
    x = df.copy()
    years = x["period"].map(parse_year)
    return x[years.notna() & years.between(start_year, end_year)].copy()


def coverage_fields(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {"first_period": None, "last_period": None, "period_count": 0}
    if "period" in df.columns:
        vals = df["period"].dropna().astype(str).drop_duplicates().tolist()
        def key(v: str):
            y = parse_year(v) or -1
            q = 0
            m = re.search(r"Q([1-4])", v.upper())
            if m:
                q = int(m.group(1))
            return (y, q, v)
        vals = sorted(vals, key=key)
        return {
            "first_period": vals[0] if vals else None,
            "last_period": vals[-1] if vals else None,
            "period_count": len(vals),
        }
    for c in ("time", "date", "trading_date"):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce").dropna()
            return {
                "first_period": s.min().strftime("%Y-%m-%d") if len(s) else None,
                "last_period": s.max().strftime("%Y-%m-%d") if len(s) else None,
                "period_count": int(s.dt.normalize().nunique()) if len(s) else 0,
            }
    return {"first_period": None, "last_period": None, "period_count": 0}


def finance_call_source(symbol: str, report: str, period: str, source: str) -> pd.DataFrame:
    f = Finance(source=source, symbol=symbol)
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


def finance_call(symbol: str, report: str, period: str = "quarter") -> tuple[pd.DataFrame, str]:
    # VCI is source of record; standardized MAS/KBS are controlled fallbacks.
    last = None
    for source in ("VCI", "MAS", "KBS"):
        try:
            df = retry(lambda s=source: finance_call_source(symbol, report, period, s), f"{symbol}:{source}:{report}:{period}")
            if df is not None and not df.empty:
                return df, source
        except Exception as exc:
            last = exc
    if last:
        raise last
    raise RuntimeError(f"No data for {symbol} {report}")


def quote_history(symbol: str) -> tuple[pd.DataFrame, str]:
    last = None
    for source in ("VCI", "KBS", "VND", "MAS"):
        try:
            # Constructor-level symbol avoids provider-specific symbol binding issues.
            q = Quote(source=source, symbol=symbol)
            df = retry(
                lambda: q.history(start=LOOKBACK_START, end=TARGET_END, interval="1D"),
                f"{symbol}:{source}:price",
                attempts=3,
            )
            if df is not None and not df.empty:
                return df, source
        except Exception as exc:
            last = exc
    if last:
        raise last
    raise RuntimeError(f"No price history for {symbol}")


def append_manifest(manifest: list[dict], symbol: str, dataset: str, periodicity: str, rows: int, status: str, source: str | None, df: pd.DataFrame | None):
    cov = coverage_fields(df)
    manifest.append({
        "symbol": symbol,
        "dataset": dataset,
        "periodicity": periodicity,
        "rows": rows,
        "status": status,
        "source": source,
        **cov,
    })


def extract_core(symbol: str, root: Path, manifest: list[dict], errors: list[dict], delay: float):
    for report in ("balance_sheet", "income_statement", "cash_flow", "ratio"):
        try:
            df, source = finance_call(symbol, report, "quarter")
            df = filter_finance_window(df)
            n = save_df(df, root / "finance_quarterly" / report / f"{symbol}.parquet", source)
            append_manifest(manifest, symbol, report, "quarter", n, "OK" if n else "EMPTY", source, df)
        except Exception as exc:
            errors.append({"symbol": symbol, "dataset": report, "error_type": type(exc).__name__, "error": str(exc)})
        time.sleep(delay)

    try:
        px, source = quote_history(symbol)
        if px is not None and not px.empty:
            time_col = next((c for c in ("time", "date", "trading_date") if c in px.columns), None)
            if time_col:
                t = pd.to_datetime(px[time_col], errors="coerce")
                px = px[t.between(pd.Timestamp(LOOKBACK_START), pd.Timestamp(TARGET_END))].copy()
        n = save_df(px, root / "price_daily" / f"{symbol}.parquet", source)
        append_manifest(manifest, symbol, "price_daily", "daily", n, "OK" if n else "EMPTY", source, px)
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "price_daily", "error_type": type(exc).__name__, "error": str(exc)})
    time.sleep(delay)

    try:
        ref = Reference()
        info = retry(lambda: ref.company(symbol).info(), f"{symbol}:company_info")
        n = save_df(info, root / "company_info" / f"{symbol}.parquet", "Reference")
        append_manifest(manifest, symbol, "company_info", "snapshot", n, "OK" if n else "EMPTY", "Reference", info)
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "company_info", "error_type": type(exc).__name__, "error": str(exc)})


def extract_notes(symbol: str, root: Path, manifest: list[dict], errors: list[dict], delay: float):
    try:
        df, source = finance_call(symbol, "note", "year")
        df = filter_finance_window(df)
        n = save_df(df, root / "notes_annual" / f"{symbol}.parquet", source)
        append_manifest(manifest, symbol, "note", "year", n, "OK" if n else "EMPTY", source, df)
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
                n = save_df(df, root / "reference" / name / f"{symbol}.parquet", "Reference")
                append_manifest(manifest, symbol, name, "mixed_or_snapshot", n, "OK" if n else "EMPTY", "Reference", df)
            except Exception as exc:
                errors.append({"symbol": symbol, "dataset": name, "error_type": type(exc).__name__, "error": str(exc)})
            time.sleep(delay)
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "reference_init", "error_type": type(exc).__name__, "error": str(exc)})

    try:
        from vnstock_data import Fundamental
        filing = retry(lambda: Fundamental().equity(symbol).filing(), f"{symbol}:filing", attempts=3)
        n = save_df(filing, root / "filings" / f"{symbol}.parquet", "VCI")
        append_manifest(manifest, symbol, "filing", "event", n, "OK" if n else "EMPTY", "VCI", filing)
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "filing", "error_type": type(exc).__name__, "error": str(exc)})


def extract_trading(symbol: str, root: Path, manifest: list[dict], errors: list[dict], delay: float):
    try:
        tr = Trading(symbol=symbol, source="VCI")
        tasks = {
            "foreign_trade": lambda: tr.foreign_trade(start=TARGET_START, end=TARGET_END),
            "prop_trade": lambda: tr.prop_trade(start=TARGET_START, end=TARGET_END),
        }
        for name, fn in tasks.items():
            try:
                df = retry(fn, f"{symbol}:{name}", attempts=3)
                n = save_df(df, root / "trading" / name / f"{symbol}.parquet", "VCI")
                append_manifest(manifest, symbol, name, "daily", n, "OK" if n else "EMPTY", "VCI", df)
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
            n = save_df(df, macro_root / f"{name}.parquet", "Macro")
            append_manifest(manifest, "MACRO", name, "mixed", n, "OK" if n else "EMPTY", "Macro", df)
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
        "version": "v2",
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
