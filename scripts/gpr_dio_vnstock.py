from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from vnstock_data import Finance, Listing, Macro

OUT = Path("output")
CTX = OUT / "context"
OUT.mkdir(exist_ok=True)
CTX.mkdir(exist_ok=True)

UNIVERSE = {
    "STEEL": ["DTL", "HMC", "HPG", "HSG", "NKG", "SMC", "TLH", "VCA", "KMT", "SSM", "VGS"],
    "FERTILIZER": ["BFC", "DCM", "DPM", "SFG", "DDV", "LAS"],
    "TEXTILE": ["AAT", "ADS", "EVE", "GIL", "GMC", "HTG", "KMR", "MSH", "STK", "SVD", "TCM", "TVT", "TDT", "TET", "TNG"],
    "SEAFOOD": ["AAM", "ABT", "ACL", "ANV", "CMX", "FMC", "IDI", "KHS", "SJ1", "VHC"],
}
INDUSTRY = {s: g for g, arr in UNIVERSE.items() for s in arr}
SYMBOLS = [s for arr in UNIVERSE.values() for s in arr]

START_IS = pd.Period("2019Q1", freq="Q")
START_BS = pd.Period("2019Q4", freq="Q")
START_PANEL = pd.Period("2020Q1", freq="Q")
END_PANEL = pd.Period("2025Q4", freq="Q")
GRID_PERIODS = pd.period_range(START_IS, END_PANEL, freq="Q")
PANEL_PERIODS = pd.period_range(START_PANEL, END_PANEL, freq="Q")
REPORT_START = {
    "balance_sheet": START_BS,
    "income_statement": START_IS,
    "cash_flow": START_IS,
    "ratio": START_IS,
}
REPORTS = tuple(REPORT_START)


def norm(x: object) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def clean_label(x: object) -> str:
    s = norm(x)
    # Remove common leading line numbering such as "1.", "11.", "IV.".
    s = re.sub(r"^(?:\d+|[ivxlcdm]+)\s+", "", s)
    return s.strip()


def qperiod(x: object):
    if pd.isna(x):
        return pd.NaT
    s = str(x).upper().strip()
    m = re.search(r"(20\d{2})\D*Q([1-4])", s)
    if m:
        return pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q")
    m = re.search(r"Q([1-4])\D*(20\d{2})", s)
    if m:
        return pd.Period(f"{m.group(2)}Q{m.group(1)}", freq="Q")
    return pd.NaT


def year_of(x: object):
    m = re.search(r"(20\d{2})", str(x))
    return int(m.group(1)) if m else None


def retry(fn, label: str, attempts: int = 4):
    err = None
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:
            err = e
            print(f"[retry] {label} attempt={k + 1}: {type(e).__name__}: {e}")
            time.sleep(min(12, 2 ** k))
    raise err


def call_finance(symbol: str, report: str, source: str = "VCI", period: str = "quarter") -> pd.DataFrame:
    f = Finance(source=source, symbol=symbol)
    fn = getattr(f, report)

    def invoke():
        # v3.2.8 unified signature. The shorter fallbacks only protect against
        # small provider-specific signature differences without guessing data.
        argsets = [
            dict(period=period, lang="vi", drop_empty=False, com_type="Regular", format="long"),
            dict(period=period, drop_empty=False, com_type="Regular", format="long"),
            dict(period=period, drop_empty=False, format="long"),
        ]
        last = None
        for kwargs in argsets:
            try:
                return fn(**kwargs)
            except TypeError as e:
                last = e
        raise last

    return retry(invoke, f"{symbol}:{source}:{report}:{period}")


def canonical(df: pd.DataFrame, symbol: str, report: str, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().reset_index(drop=True)
    cmap = {norm(c): c for c in df.columns}

    def pick(*names):
        for n in names:
            if norm(n) in cmap:
                return cmap[norm(n)]
        return None

    c_period = pick("period", "report_period", "fiscal_period")
    c_id = pick("id", "item_id", "semantic_id")
    c_name = pick("name", "item", "item_name")
    c_value = pick("value", "amount")
    if not all([c_period, c_id, c_name, c_value]):
        raise RuntimeError(f"Unexpected {source} {report} schema: {list(df.columns)}")

    out = pd.DataFrame({
        "symbol": symbol,
        "industry": INDUSTRY.get(symbol),
        "report": report,
        "period": df[c_period].astype(str),
        "id": df[c_id].astype(str),
        "name": df[c_name].astype(str),
        "value": pd.to_numeric(df[c_value], errors="coerce"),
        "unit": df[pick("unit")] if pick("unit") else None,
        "order": df[pick("order")] if pick("order") else None,
        "level": df[pick("level")] if pick("level") else None,
        "off_balance": df[pick("off_balance")] if pick("off_balance") else None,
        "source": source,
        "retrieval_utc": datetime.now(timezone.utc).isoformat(),
    })
    out["qperiod"] = out["period"].map(qperiod)
    return out


errors: list[dict] = []
frames: list[pd.DataFrame] = []


def fetch_report(symbol: str, report: str) -> pd.DataFrame:
    # VCI is the research source of record. MAS is only a fallback and is
    # explicitly tagged so robustness checks can exclude fallback observations.
    for source in ("VCI", "MAS"):
        try:
            df = call_finance(symbol, report, source=source, period="quarter")
            can = canonical(df, symbol, report, source)
            if not can.empty:
                return can
            errors.append({"symbol": symbol, "industry": INDUSTRY[symbol], "report": report,
                           "source": source, "error_type": "EmptyData", "error": "empty response"})
        except Exception as e:
            errors.append({"symbol": symbol, "industry": INDUSTRY[symbol], "report": report,
                           "source": source, "error_type": type(e).__name__, "error": str(e)})
    return pd.DataFrame()


# Current listing metadata is used as a data-quality check; it does not decide
# inclusion automatically because the thesis sample is historical 2020-2025.
listing_target = pd.DataFrame()
try:
    listing = Listing(source="VCI").symbols_by_exchange()
    if listing is not None and not listing.empty and "symbol" in listing.columns:
        listing_target = listing[listing["symbol"].astype(str).isin(SYMBOLS)].copy()
        listing_target.to_csv(OUT / "gpr_dio_listing_metadata.csv", index=False, encoding="utf-8-sig")
except Exception as e:
    errors.append({"symbol": "ALL", "industry": None, "report": "listing", "source": "VCI",
                   "error_type": type(e).__name__, "error": str(e)})

for i, symbol in enumerate(SYMBOLS, 1):
    print(f"[{i}/{len(SYMBOLS)}] {symbol}")
    for report in REPORTS:
        can = fetch_report(symbol, report)
        if not can.empty:
            start = REPORT_START[report]
            can = can[can["qperiod"].notna() & (can["qperiod"] >= start) & (can["qperiod"] <= END_PANEL)].copy()
            frames.append(can)
        time.sleep(0.15)

if not frames:
    raise RuntimeError("No vnstock_data financial data returned")

raw = pd.concat(frames, ignore_index=True)
raw = raw.drop_duplicates(["symbol", "report", "period", "id", "source"], keep="last")

# Record which source was actually used per company/report.
source_usage = (raw.groupby(["symbol", "report"])["source"]
                .agg(lambda s: "+".join(sorted(set(s))))
                .reset_index())
source_usage.to_csv(OUT / "gpr_dio_source_usage.csv", index=False, encoding="utf-8-sig")


def resolve_id(smoke: pd.DataFrame, report: str, exact_ids=(), exact_names=(), contains=()):
    sub = smoke[smoke["report"].eq(report)].copy()
    if sub.empty:
        return None, None, "not_found"
    # Prefer VCI rows if HPG ever needed a fallback for another report.
    if sub["source"].eq("VCI").any():
        sub = sub[sub["source"].eq("VCI")].copy()
    ids_upper = sub["id"].astype(str).str.upper()
    for sid in exact_ids:
        h = sub[ids_upper.eq(str(sid).upper())]
        if not h.empty:
            r = h.iloc[0]
            return r["id"], r["name"], "exact_id"

    labels = sub["name"].map(clean_label)
    for name in exact_names:
        h = sub[labels.eq(clean_label(name))]
        if not h.empty:
            r = h.iloc[0]
            return r["id"], r["name"], "exact_name"

    for phrase in contains:
        words = clean_label(phrase).split()
        mask = labels.map(lambda s: all(w in s for w in words))
        h = sub[mask].copy()
        if not h.empty:
            h["_label_len"] = h["name"].astype(str).map(clean_label).str.len()
            if "level" in h.columns:
                h["_level_num"] = pd.to_numeric(h["level"], errors="coerce")
                h = h.sort_values(["_label_len", "_level_num"], na_position="last")
            else:
                h = h.sort_values("_label_len")
            r = h.iloc[0]
            return r["id"], r["name"], "contains"
    return None, None, "not_found"


FIELD_SPECS = {
    "Inventory_End": ("balance_sheet", ["BS_INVENTORIES"], ["Hàng tồn kho"], ["hang ton kho"]),
    "TotalAssets": ("balance_sheet", ["BS_TOTAL_ASSETS"], ["Tổng cộng tài sản", "Tổng tài sản"], ["tong tai san"]),
    "Cash": ("balance_sheet", ["BS_CASH_AND_PRECIOUS_METALS", "BS_CASH_AND_CASH_EQUIVALENTS"],
             ["Tiền và các khoản tương đương tiền", "Tiền"], ["tien va cac khoan tuong duong tien"]),
    "ShortReceivables": ("balance_sheet", ["BS_SHORT_TERM_RECEIVABLES"], ["Các khoản phải thu ngắn hạn"], ["phai thu ngan han"]),
    "TradeReceivables": ("balance_sheet", ["BS_TRADE_RECEIVABLES"], ["Phải thu ngắn hạn của khách hàng", "Phải thu khách hàng"], ["phai thu khach hang"]),
    "TotalLiabilities": ("balance_sheet", ["BS_TOTAL_LIABILITIES", "BS_LIABILITIES"], ["Nợ phải trả"], ["no phai tra"]),
    "TradePayables": ("balance_sheet", ["BS_SHORT_TERM_TRADE_PAYABLES", "BS_TRADE_PAYABLES"], ["Phải trả người bán ngắn hạn"], ["phai tra nguoi ban ngan han"]),
    "ShortDebt": ("balance_sheet", [], ["Vay và nợ thuê tài chính ngắn hạn"], ["vay va no thue tai chinh ngan han"]),
    "LongDebt": ("balance_sheet", [], ["Vay và nợ thuê tài chính dài hạn"], ["vay va no thue tai chinh dai han"]),
    "TotalEquity": ("balance_sheet", ["BS_TOTAL_EQUITY", "BS_OWNERS_EQUITY"], ["Vốn chủ sở hữu"], ["von chu so huu"]),
    "Revenue_Q": ("income_statement", ["IS_NET_REVENUE"], ["Doanh thu thuần về bán hàng và cung cấp dịch vụ", "Doanh thu thuần"], ["doanh thu thuan"]),
    "COGS_Q": ("income_statement", ["IS_COST_OF_GOODS_SOLD", "IS_COST_OF_SALES"], ["Giá vốn hàng bán"], ["gia von hang ban"]),
    "GrossProfit": ("income_statement", ["IS_GROSS_PROFIT"], ["Lợi nhuận gộp về bán hàng và cung cấp dịch vụ", "Lợi nhuận gộp"], ["loi nhuan gop"]),
    "ProfitBeforeTax": ("income_statement", ["IS_PROFIT_BEFORE_TAX"], ["Tổng lợi nhuận kế toán trước thuế", "Lợi nhuận trước thuế"], ["loi nhuan truoc thue"]),
    "NetIncome_Q": ("income_statement", ["IS_NET_PROFIT_AFTER_TAX"], ["Lợi nhuận sau thuế thu nhập doanh nghiệp"], ["loi nhuan sau thue"]),
    "OperatingCF_Q": ("cash_flow", ["CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES"], ["Lưu chuyển tiền thuần từ hoạt động kinh doanh"], ["luu chuyen tien thuan tu hoat dong kinh doanh"]),
}

smoke = raw[raw["symbol"].eq("HPG")].copy()
mapping: dict[str, str | None] = {}
map_rows = []
for field, (report, ids, names, contains) in FIELD_SPECS.items():
    sid, label, method = resolve_id(smoke, report, ids, names, contains)
    mapping[field] = sid
    map_rows.append({"field": field, "report": report, "resolved_id": sid,
                     "resolved_name_hpg": label, "method": method})
map_df = pd.DataFrame(map_rows)
print(map_df.to_string(index=False))

# Always persist the raw extraction and HPG schema before any model gate.
raw_out = raw.drop(columns=["qperiod"]).copy()
raw_out.to_csv(OUT / "gpr_dio_vnstock_raw.csv", index=False, encoding="utf-8-sig")
raw_out.to_parquet(OUT / "gpr_dio_vnstock_raw.parquet", index=False)
map_df.to_csv(OUT / "gpr_dio_semantic_mapping.csv", index=False, encoding="utf-8-sig")
smoke[["report", "period", "id", "name", "value", "source"]].drop_duplicates().to_csv(
    OUT / "hpg_schema_audit.csv", index=False, encoding="utf-8-sig")

errors_df = pd.DataFrame(errors, columns=["symbol", "industry", "report", "source", "error_type", "error"])
errors_df.to_csv(OUT / "gpr_dio_error_log.csv", index=False, encoding="utf-8-sig")

needed = ["Inventory_End", "TotalAssets", "Revenue_Q", "COGS_Q", "NetIncome_Q"]
missing_core = [x for x in needed if not mapping.get(x)]
if missing_core:
    summary = {
        "status": "FAIL_SEMANTIC_GATE",
        "missing_core": missing_core,
        "symbols_expected": len(SYMBOLS),
        "raw_rows": len(raw_out),
        "semantic_mapping": map_df.to_dict(orient="records"),
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    raise RuntimeError(f"Missing core semantic IDs on HPG: {missing_core}")

# Build an extended 2019-2025 grid so 2020 Sales_Growth has a full lag-4 base.
grid = pd.MultiIndex.from_product([SYMBOLS, GRID_PERIODS], names=["Symbol", "Quarter"]).to_frame(index=False)
grid["Industry"] = grid["Symbol"].map(INDUSTRY)
grid["QuarterEnd"] = grid["Quarter"].map(lambda p: p.end_time.normalize())
grid["DaysInQuarter"] = grid["Quarter"].map(lambda p: (p.end_time.normalize() - p.start_time.normalize()).days + 1)
panel_full = grid.copy()

for field, sid in mapping.items():
    if not sid:
        panel_full[field] = np.nan
        continue
    x = raw[(raw["id"].astype(str).str.upper().eq(str(sid).upper())) & raw["qperiod"].notna()].copy()
    x = x.groupby(["symbol", "qperiod"], as_index=False)["value"].last()
    x = x.rename(columns={"symbol": "Symbol", "qperiod": "Quarter", "value": field})
    panel_full = panel_full.merge(x, on=["Symbol", "Quarter"], how="left")

# Point-in-time balance-sheet values from the immediately preceding quarter.
for field, lagcol in {
    "Inventory_End": "Inventory_Lag",
    "TotalAssets": "TotalAssets_Lag",
    "TradeReceivables": "TradeReceivables_Lag",
    "ShortReceivables": "ShortReceivables_Lag",
    "TradePayables": "TradePayables_Lag",
}.items():
    sid = mapping.get(field)
    if not sid:
        panel_full[lagcol] = np.nan
        continue
    x = raw[(raw["id"].astype(str).str.upper().eq(str(sid).upper())) & raw["qperiod"].notna()].copy()
    x = x.groupby(["symbol", "qperiod"], as_index=False)["value"].last()
    x["Quarter"] = x["qperiod"] + 1
    x = x.rename(columns={"symbol": "Symbol", "value": lagcol})[["Symbol", "Quarter", lagcol]]
    panel_full = panel_full.merge(x, on=["Symbol", "Quarter"], how="left")

panel_full["COGS_Q_raw"] = panel_full["COGS_Q"]
panel_full["COGS_Q"] = panel_full["COGS_Q_raw"].abs()
panel_full["COGS_NegativeSign_Flag"] = panel_full["COGS_Q_raw"] < 0
panel_full["AvgInventory"] = (panel_full["Inventory_End"] + panel_full["Inventory_Lag"]) / 2
panel_full["AvgTotalAssets"] = (panel_full["TotalAssets"] + panel_full["TotalAssets_Lag"]) / 2
panel_full["Receivables"] = panel_full["TradeReceivables"].where(panel_full["TradeReceivables"].notna(), panel_full["ShortReceivables"])
panel_full["Receivables_Lag"] = panel_full["TradeReceivables_Lag"].where(panel_full["TradeReceivables_Lag"].notna(), panel_full["ShortReceivables_Lag"])
panel_full["AvgReceivables"] = (panel_full["Receivables"] + panel_full["Receivables_Lag"]) / 2
panel_full["AvgTradePayables"] = (panel_full["TradePayables"] + panel_full["TradePayables_Lag"]) / 2

valid_cogs = panel_full["COGS_Q"] > 0
valid_rev = panel_full["Revenue_Q"] > 0
panel_full["DIO"] = np.where(valid_cogs, panel_full["AvgInventory"] / panel_full["COGS_Q"] * panel_full["DaysInQuarter"], np.nan)
panel_full["Inv_Assets"] = panel_full["Inventory_End"] / panel_full["TotalAssets"]
panel_full["Inv_Sales"] = np.where(valid_rev, panel_full["Inventory_End"] / panel_full["Revenue_Q"], np.nan)
panel_full["Size_lnAssets"] = np.where(panel_full["TotalAssets"] > 0, np.log(panel_full["TotalAssets"]), np.nan)
panel_full["Leverage"] = panel_full["TotalLiabilities"] / panel_full["TotalAssets"]
panel_full["Cash_Ratio"] = panel_full["Cash"] / panel_full["TotalAssets"]
panel_full["ROA"] = panel_full["NetIncome_Q"] / panel_full["AvgTotalAssets"]
panel_full["OperatingCF_Assets"] = panel_full["OperatingCF_Q"] / panel_full["AvgTotalAssets"]
panel_full["Revenue_Q_lag4"] = panel_full.groupby("Symbol")["Revenue_Q"].shift(4)
panel_full["Sales_Growth"] = panel_full["Revenue_Q"] / panel_full["Revenue_Q_lag4"] - 1
panel_full["DSO"] = np.where(valid_rev, panel_full["AvgReceivables"] / panel_full["Revenue_Q"] * panel_full["DaysInQuarter"], np.nan)
panel_full["DPO"] = np.where(valid_cogs, panel_full["AvgTradePayables"] / panel_full["COGS_Q"] * panel_full["DaysInQuarter"], np.nan)
panel_full["CCC"] = panel_full["DIO"] + panel_full["DSO"] - panel_full["DPO"]
panel_full["NonCalendarFY_Flag"] = panel_full["Symbol"].eq("HSG")

panel = panel_full[(panel_full["Quarter"] >= START_PANEL) & (panel_full["Quarter"] <= END_PANEL)].copy()

# Attach current exchange metadata for audit only.
exchange_map = {}
if not listing_target.empty and "symbol" in listing_target.columns and "exchange" in listing_target.columns:
    exchange_map = dict(zip(listing_target["symbol"].astype(str), listing_target["exchange"].astype(str)))
panel["CurrentExchange"] = panel["Symbol"].map(exchange_map)
panel["Current_HOSE_HNX_Flag"] = panel["CurrentExchange"].str.upper().isin(["HOSE", "HSX", "HNX"])

quality = []
for symbol in SYMBOLS:
    p = panel[panel["Symbol"].eq(symbol)].copy()
    core_mask = p[["Inventory_End", "COGS_Q", "Revenue_Q", "TotalAssets"]].notna().all(axis=1)
    complete = int(core_mask.sum())
    dio_n = int(p["DIO"].notna().sum())
    first_q = str(p.loc[core_mask, "Quarter"].min()) if core_mask.any() else None
    last_q = str(p.loc[core_mask, "Quarter"].max()) if core_mask.any() else None
    src_rows = source_usage[source_usage["symbol"].eq(symbol)]
    src = "+".join(sorted(set(src_rows["source"]))) if not src_rows.empty else None
    status = "PASS" if complete >= 22 and dio_n >= 22 else ("PARTIAL" if complete > 0 else "FAIL")
    if symbol == "HSG":
        status = "NON_CALENDAR_FY"
    quality.append({
        "ticker": symbol,
        "industry": INDUSTRY[symbol],
        "current_exchange": exchange_map.get(symbol),
        "first_available_quarter": first_q,
        "last_available_quarter": last_q,
        "expected_quarters": 24,
        "core_complete_quarters": complete,
        "dio_available_quarters": dio_n,
        "coverage_pct": complete / 24,
        "inventory_missing_pct": float(p["Inventory_End"].isna().mean()),
        "cogs_missing_pct": float(p["COGS_Q"].isna().mean()),
        "revenue_missing_pct": float(p["Revenue_Q"].isna().mean()),
        "assets_missing_pct": float(p["TotalAssets"].isna().mean()),
        "dio_missing_pct": float(p["DIO"].isna().mean()),
        "financial_source": src,
        "status": status,
    })
quality_df = pd.DataFrame(quality)

# HPG reconciliation: quarterly standalone revenue should sum to annual revenue.
recon = {"symbol": "HPG", "year": 2025}
try:
    annual_df = call_finance("HPG", "income_statement", source="VCI", period="year")
    annual = canonical(annual_df, "HPG", "income_statement", "VCI")
    sid = mapping["Revenue_Q"]
    fy = annual[(annual["id"].astype(str).str.upper().eq(str(sid).upper())) & annual["period"].map(year_of).eq(2025)]["value"]
    fyv = float(fy.iloc[0]) if len(fy) else np.nan
    hpg25 = panel[(panel["Symbol"].eq("HPG")) & panel["Quarter"].astype(str).str.startswith("2025")]
    qsum = hpg25["Revenue_Q"].sum(min_count=4)
    recon.update({
        "quarter_sum": qsum,
        "annual_value": fyv,
        "difference": qsum - fyv if pd.notna(qsum) and pd.notna(fyv) else np.nan,
        "difference_pct": qsum / fyv - 1 if pd.notna(qsum) and pd.notna(fyv) and fyv else np.nan,
    })
    if "GrossProfit" in hpg25 and hpg25["GrossProfit"].notna().any():
        ident = hpg25["Revenue_Q"] - hpg25["COGS_Q"] - hpg25["GrossProfit"]
        recon["gross_profit_identity_max_abs"] = float(ident.abs().max())
except Exception as e:
    recon["error"] = f"{type(e).__name__}: {e}"

# Bronze macro/commodity context: useful for descriptive models and sector
# robustness. Main firm+quarter FE model will not include common macro levels
# mechanically because quarter FE absorb them.
context_errors = []
try:
    mac = Macro()
    context_tasks = {
        "gdp_quarter": lambda: mac.economy().gdp(start="2019-01-01", end="2025-12-31", period="quarter"),
        "cpi_quarter": lambda: mac.economy().cpi(start="2019-01-01", end="2025-12-31", period="quarter"),
        "industry_prod_quarter": lambda: mac.economy().industry_prod(start="2019-01-01", end="2025-12-31", period="quarter"),
        "import_export_quarter": lambda: mac.economy().import_export(start="2019-01-01", end="2025-12-31", period="quarter"),
        "fdi_quarter": lambda: mac.economy().fdi(start="2019-01-01", end="2025-12-31", period="quarter"),
        "exchange_rate_quarter": lambda: mac.currency().exchange_rate(start="2019-01-01", end="2025-12-31", period="quarter"),
        "interest_rate_quarter": lambda: mac.currency().interest_rate(start="2019-01-01", end="2025-12-31", period="quarter", format="long"),
        "steel_global": lambda: mac.commodity().steel(market="GLOBAL", start="2019-01-01", end="2025-12-31"),
        "steel_vn": lambda: mac.commodity().steel(market="VN", start="2019-01-01", end="2025-12-31"),
        "iron_ore": lambda: mac.commodity().iron_ore(start="2019-01-01", end="2025-12-31"),
        "fertilizer_ure": lambda: mac.commodity().fertilizer_ure(start="2019-01-01", end="2025-12-31"),
        "oil_crude": lambda: mac.commodity().oil_crude(start="2019-01-01", end="2025-12-31"),
    }
    for name, fn in context_tasks.items():
        try:
            df = retry(fn, f"context:{name}", attempts=2)
            if df is not None and not df.empty:
                df.to_csv(CTX / f"{name}.csv", index=True, encoding="utf-8-sig")
        except Exception as e:
            context_errors.append({"dataset": name, "error_type": type(e).__name__, "error": str(e)})
except Exception as e:
    context_errors.append({"dataset": "Macro_init", "error_type": type(e).__name__, "error": str(e)})

pd.DataFrame(context_errors).to_csv(CTX / "context_error_log.csv", index=False, encoding="utf-8-sig")

panel_out = panel.copy()
panel_out["Quarter"] = panel_out["Quarter"].astype(str)
panel_out.to_csv(OUT / "gpr_dio_financial_panel.csv", index=False, encoding="utf-8-sig")
panel_out.to_parquet(OUT / "gpr_dio_financial_panel.parquet", index=False)
quality_df.to_csv(OUT / "gpr_dio_data_quality.csv", index=False, encoding="utf-8-sig")
map_df.to_csv(OUT / "gpr_dio_semantic_mapping.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter(OUT / "gpr_dio_bronze_research_pack.xlsx", engine="openpyxl") as xw:
    panel_out.to_excel(xw, sheet_name="Financial_Panel", index=False)
    quality_df.to_excel(xw, sheet_name="Data_Quality", index=False)
    map_df.to_excel(xw, sheet_name="Semantic_Mapping", index=False)
    source_usage.to_excel(xw, sheet_name="Source_Usage", index=False)
    errors_df.to_excel(xw, sheet_name="Error_Log", index=False)
    if not listing_target.empty:
        listing_target.to_excel(xw, sheet_name="Listing_Metadata", index=False)

summary = {
    "status": "PASS_PIPELINE" if int((quality_df["status"] == "PASS").sum()) > 0 else "HOLD_DATA_QUALITY",
    "tier": "bronze",
    "symbols_expected": len(SYMBOLS),
    "symbols_with_core_data": int((quality_df["core_complete_quarters"] > 0).sum()),
    "panel_rows_expected": len(SYMBOLS) * 24,
    "panel_rows": len(panel_out),
    "pass": int((quality_df["status"] == "PASS").sum()),
    "partial": int((quality_df["status"] == "PARTIAL").sum()),
    "fail": int((quality_df["status"] == "FAIL").sum()),
    "non_calendar_fy": int((quality_df["status"] == "NON_CALENDAR_FY").sum()),
    "financial_errors": len(errors_df),
    "context_errors": len(context_errors),
    "hpg_reconciliation": recon,
    "semantic_mapping": map_df.to_dict(orient="records"),
}
(OUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("RUN_SUMMARY_BEGIN")
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print("RUN_SUMMARY_END")
