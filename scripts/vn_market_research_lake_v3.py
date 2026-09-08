from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
from vnstock_data import Fundamental, Reference

import vn_market_research_lake_v2 as base


def fundamental_call(symbol: str, report: str, period: str) -> pd.DataFrame:
    eq = Fundamental().equity(symbol)
    fn = getattr(eq, report)
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


def save_task(manifest, errors, symbol, dataset, periodicity, source, df, path):
    try:
        n = base.save_df(df, path, source)
        base.append_manifest(manifest, symbol, dataset, periodicity, n, "OK" if n else "EMPTY", source, df)
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": dataset, "error_type": type(exc).__name__, "error": str(exc)})


def extract_core_v3(symbol: str, root: Path, manifest: list[dict], errors: list[dict], delay: float):
    # Primary full-history structured layer. HPG validation showed 58 quarterly
    # periods (2012Q1-2025Q4) for BS/IS/ratio on the Bronze Fundamental layer.
    for report in ("balance_sheet", "income_statement", "cash_flow", "ratio"):
        try:
            df = base.retry(lambda r=report: fundamental_call(symbol, r, "quarter"), f"{symbol}:Fundamental:{report}:quarter")
            df = base.filter_finance_window(df)
            save_task(
                manifest, errors, symbol,
                f"fundamental_{report}_quarter", "quarter", "Fundamental",
                df, root / "fundamental_quarterly" / report / f"{symbol}.parquet"
            )
        except Exception as exc:
            errors.append({"symbol": symbol, "dataset": f"fundamental_{report}_quarter", "error_type": type(exc).__name__, "error": str(exc)})
        time.sleep(delay)

    # Annual layer is deliberately retained as a separate source. It is useful
    # when quarterly cash-flow history is shorter and for annual-policy studies.
    for report in ("balance_sheet", "income_statement", "cash_flow", "ratio"):
        try:
            df = base.retry(lambda r=report: fundamental_call(symbol, r, "year"), f"{symbol}:Fundamental:{report}:year")
            df = base.filter_finance_window(df)
            save_task(
                manifest, errors, symbol,
                f"fundamental_{report}_year", "year", "Fundamental",
                df, root / "fundamental_annual" / report / f"{symbol}.parquet"
            )
        except Exception as exc:
            errors.append({"symbol": symbol, "dataset": f"fundamental_{report}_year", "error_type": type(exc).__name__, "error": str(exc)})
        time.sleep(delay)

    # VCI quarterly cash flow is kept separately because validation showed that
    # it can contain more quarters than Fundamental cash_flow for some firms.
    try:
        df, source = base.finance_call(symbol, "cash_flow", "quarter")
        df = base.filter_finance_window(df)
        save_task(
            manifest, errors, symbol,
            "finance_cash_flow_quarter", "quarter", source,
            df, root / "finance_quarterly_secondary" / "cash_flow" / f"{symbol}.parquet"
        )
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "finance_cash_flow_quarter", "error_type": type(exc).__name__, "error": str(exc)})
    time.sleep(delay)

    # Daily market outcomes from 2013-2025.
    try:
        px, source = base.quote_history(symbol)
        if px is not None and not px.empty:
            time_col = next((c for c in ("time", "date", "trading_date") if c in px.columns), None)
            if time_col:
                t = pd.to_datetime(px[time_col], errors="coerce")
                px = px[t.between(pd.Timestamp(base.LOOKBACK_START), pd.Timestamp(base.TARGET_END))].copy()
        save_task(
            manifest, errors, symbol,
            "price_daily", "daily", source,
            px, root / "price_daily" / f"{symbol}.parquet"
        )
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "price_daily", "error_type": type(exc).__name__, "error": str(exc)})
    time.sleep(delay)

    # Retrieval-date company metadata. Never treat snapshot fields as historical
    # exposures without separate verification.
    try:
        info = base.retry(lambda: Reference().company(symbol).info(), f"{symbol}:company_info")
        save_task(
            manifest, errors, symbol,
            "company_info", "snapshot", "Reference",
            info, root / "company_info" / f"{symbol}.parquet"
        )
    except Exception as exc:
        errors.append({"symbol": symbol, "dataset": "company_info", "error_type": type(exc).__name__, "error": str(exc)})


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
    universe = base.build_universe(root)

    summary = {
        "version": "v3",
        "mode": args.mode,
        "target_start": base.TARGET_START,
        "target_end": base.TARGET_END,
        "lookback_start": base.LOOKBACK_START,
        "universe_n": int(len(universe)),
        "batch_index": args.batch_index,
        "batch_size": args.batch_size,
        "started_utc": base.utcnow(),
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
        base.extract_macro(root, manifest, errors)
    else:
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] {symbol} mode={args.mode} v3", flush=True)
            if args.mode == "core":
                extract_core_v3(symbol, root, manifest, errors, args.delay)
            elif args.mode == "notes":
                base.extract_notes(symbol, root, manifest, errors, args.delay)
            elif args.mode == "trading":
                base.extract_trading(symbol, root, manifest, errors, args.delay)

    mf = pd.DataFrame(manifest)
    er = pd.DataFrame(errors)
    mf.to_csv(root / "manifest.csv", index=False, encoding="utf-8-sig")
    er.to_csv(root / "errors.csv", index=False, encoding="utf-8-sig")

    summary.update({
        "manifest_rows": int(len(mf)),
        "errors": int(len(er)),
        "finished_utc": base.utcnow(),
    })
    (root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN_SUMMARY_BEGIN")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("RUN_SUMMARY_END")


if __name__ == "__main__":
    main()
