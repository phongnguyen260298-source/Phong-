import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.vnstock_service import get_daily_price, get_ohlcv

app = FastAPI(
    title="VNStock ChatGPT API",
    version="0.2.0",
    description="Remote VNStock service for ChatGPT and other agents.",
)

bearer = HTTPBearer(auto_error=False)


def require_server_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    expected = os.getenv("VNSTOCK_SERVER_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VNSTOCK_SERVER_TOKEN is not configured",
        )
    provided = credentials.credentials if credentials else ""
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid server token",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/price/{symbol}", dependencies=[Depends(require_server_token)])
def price(symbol: str, date: str = Query(..., description="Trading date YYYY-MM-DD")) -> dict:
    try:
        data = get_daily_price(symbol=symbol, trading_date=date)
        return {"symbol": symbol.upper(), "date": date, "data": data}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"VNStock error: {exc}") from exc


@app.get("/v1/ohlcv/{symbol}", dependencies=[Depends(require_server_token)])
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
