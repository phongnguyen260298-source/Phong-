from fastapi import FastAPI, HTTPException, Query

from src.vnstock_service import get_daily_price, get_ohlcv

app = FastAPI(
    title="VNStock ChatGPT API",
    version="0.1.0",
    description="Remote VNStock service for ChatGPT and other agents.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/price/{symbol}")
def price(symbol: str, date: str = Query(..., description="Trading date YYYY-MM-DD")) -> dict:
    try:
        data = get_daily_price(symbol=symbol, trading_date=date)
        return {"symbol": symbol.upper(), "date": date, "data": data}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"VNStock error: {exc}") from exc


@app.get("/v1/ohlcv/{symbol}")
def ohlcv(
    symbol: str,
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    interval: str = Query("1D", description="VNStock interval, e.g. 1D"),
) -> dict:
    try:
        rows = get_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
        return {
            "symbol": symbol.upper(),
            "start": start,
            "end": end,
            "interval": interval,
            "count": len(rows),
            "data": rows,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"VNStock error: {exc}") from exc
