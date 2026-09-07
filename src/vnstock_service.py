import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
from vnstock import register_user
from vnstock.ui import Market

_registered = False


def ensure_registered() -> None:
    global _registered
    if _registered:
        return

    api_key = os.getenv("VNSTOCK_API_KEY")
    if not api_key:
        raise RuntimeError("VNSTOCK_API_KEY is not configured")

    success = register_user(api_key=api_key)
    if not success:
        raise RuntimeError("Vnstock API key registration failed")

    _registered = True


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []

    clean = df.copy()
    clean = clean.reset_index()
    clean = clean.where(pd.notnull(clean), None)

    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].astype(str)

    records = clean.to_dict(orient="records")
    for row in records:
        for key, value in list(row.items()):
            if hasattr(value, "item"):
                try:
                    row[key] = value.item()
                except Exception:
                    pass
    return records


def get_ohlcv(symbol: str, start: str, end: str, interval: str = "1D") -> list[dict[str, Any]]:
    ensure_registered()
    market = Market()
    df = market.equity(symbol.upper()).ohlcv(
        start=start,
        end=end,
        interval=interval,
    )
    return _records(df)


def get_daily_price(symbol: str, trading_date: str) -> dict[str, Any]:
    d = date.fromisoformat(trading_date)
    rows = get_ohlcv(
        symbol=symbol,
        start=d.isoformat(),
        end=(d + timedelta(days=1)).isoformat(),
        interval="1D",
    )
    if not rows:
        raise LookupError(f"No trading data for {symbol.upper()} on {trading_date}")
    return rows[0]
