from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from vnstock_data import Fundamental

OUT = Path("output")
OUT.mkdir(exist_ok=True)

UNIVERSE = {
    "STEEL": ["DTL","HMC","HPG","HSG","NKG","SMC","TLH","VCA","KMT","SSM","VGS"],
    "FERTILIZER": ["BFC","DCM","DPM","SFG","DDV","LAS"],
    "TEXTILE": ["AAT","ADS","EVE","GIL","GMC","HTG","KMR","MSH","STK","SVD","TCM","TVT","TDT","TET","TNG"],
    "SEAFOOD": ["AAM","ABT","ACL","ANV","CMX","FMC","IDI","KHS","SJ1","VHC"],
}
INDUSTRY = {s:g for g, arr in UNIVERSE.items() for s in arr}
SYMBOLS = [s for arr in UNIVERSE.values() for s in arr]
START_BS = pd.Period("2019Q4", freq="Q")
START_PANEL = pd.Period("2020Q1", freq="Q")
END_PANEL = pd.Period("2025Q4", freq="Q")
PANEL_PERIODS = pd.period_range(START_PANEL, END_PANEL, freq="Q")


def norm(x):
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def qperiod(x):
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


def year_of(x):
    m = re.search(r"(20\d{2})", str(x))
    return int(m.group(1)) if m else None


def retry(fn, label, attempts=4):
    err = None
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:
            err = e
            print(f"[retry] {label} attempt={k+1}: {type(e).__name__}: {e}")
            time.sleep(2 ** k)
    raise err


def fetch(symbol, report, period="quarter"):
    eq = Fundamental().equity(symbol)
    fn = getattr(eq, report)
    df = retry(lambda: fn(period=period, format="long", drop_empty=False, com_type="Regular"), f"{symbol}:{report}:{period}")
    if df is None:
        return pd.DataFrame()
    df = df.copy().reset_index(drop=True)
    df["symbol"] = symbol
    df["industry"] = INDUSTRY.get(symbol)
    df["report"] = report
    df["source"] = "VCI"
    df["retrieval_utc"] = datetime.now(timezone.utc).isoformat()
    return df


def canonical(df):
    if df.empty:
        return df
    cmap = {norm(c):c for c in df.columns}
    def pick(*names):
        for n in names:
            if norm(n) in cmap:
                return cmap[norm(n)]
        return None
    c_period = pick("period","report_period","fiscal_period")
    c_id = pick("id","item_id","semantic_id")
    c_name = pick("name","item","item_name")
    c_value = pick("value","amount")
    if not all([c_period,c_id,c_name,c_value]):
        raise RuntimeError(f"Unexpected vnstock_data schema: {list(df.columns)}")
    out = pd.DataFrame({
        "symbol": df["symbol"],
        "industry": df["industry"],
        "report": df["report"],
        "period": df[c_period].astype(str),
        "id": df[c_id].astype(str),
        "name": df[c_name].astype(str),
        "value": pd.to_numeric(df[c_value], errors="coerce"),
        "unit": df[pick("unit")] if pick("unit") else None,
        "order": df[pick("order")] if pick("order") else None,
        "level": df[pick("level")] if pick("level") else None,
        "off_balance": df[pick("off_balance")] if pick("off_balance") else None,
        "source": df["source"],
        "retrieval_utc": df["retrieval_utc"],
    })
    out["qperiod"] = out["period"].map(qperiod)
    return out


def resolve_id(smoke, report, exact_ids=(), exact_names=(), contains=()):
    sub = smoke[smoke["report"].eq(report)].copy()
    for sid in exact_ids:
        h = sub[sub["id"].eq(sid)]
        if not h.empty:
            r = h.iloc[0]
            return r["id"], r["name"], "exact_id"
    nn = sub["name"].map(norm)
    for name in exact_names:
        h = sub[nn.eq(norm(name))]
        if not h.empty:
            r = h.iloc[0]
            return r["id"], r["name"], "exact_name"
    for pat in contains:
        words = norm(pat).split()
        h = sub[nn.map(lambda s: all(w in s for w in words))]
        if not h.empty:
            h = h.assign(_len=h["name"].astype(str).str.len()).sort_values("_len")
            r = h.iloc[0]
            return r["id"], r["name"], "contains"
    return None, None, "not_found"


FIELD_SPECS = {
    "Inventory_End": ("balance_sheet", ["BS_INVENTORIES"], ["Hàng tồn kho"], ["inventories"]),
    "TotalAssets": ("balance_sheet", ["BS_TOTAL_ASSETS"], ["Tổng tài sản"], ["total assets"]),
    "Cash": ("balance_sheet", ["BS_CASH_AND_PRECIOUS_METALS"], ["Tiền và các khoản tương đương tiền"], ["cash equivalents"]),
    "ShortReceivables": ("balance_sheet", ["BS_SHORT_TERM_RECEIVABLES"], ["Các khoản phải thu ngắn hạn"], ["short term receivables"]),
    "TradeReceivables": ("balance_sheet", ["BS_TRADE_RECEIVABLES"], ["Phải thu ngắn hạn của khách hàng"], ["trade receivables"]),
    "TotalLiabilities": ("balance_sheet", [], ["Nợ phải trả"], ["total liabilities"]),
    "TradePayables": ("balance_sheet", [], ["Phải trả người bán ngắn hạn"], ["trade payables","phai tra nguoi ban"]),
    "ShortDebt": ("balance_sheet", [], ["Vay và nợ thuê tài chính ngắn hạn"], ["short term borrowings","vay va no thue tai chinh ngan han"]),
    "LongDebt": ("balance_sheet", [], ["Vay và nợ thuê tài chính dài hạn"], ["long term borrowings","vay va no thue tai chinh dai han"]),
    "TotalEquity": ("balance_sheet", [], ["Vốn chủ sở hữu"], ["total equity"]),
    "Revenue_Q": ("income_statement", ["IS_NET_REVENUE"], ["Doanh thu thuần về bán hàng và cung cấp dịch vụ"], ["net revenue","doanh thu thuan"]),
    "COGS_Q": ("income_statement", [], ["Giá vốn hàng bán"], ["cost of goods sold","gia von hang ban"]),
    "GrossProfit": ("income_statement", [], ["Lợi nhuận gộp về bán hàng và cung cấp dịch vụ"], ["gross profit","loi nhuan gop"]),
    "ProfitBeforeTax": ("income_statement", [], ["Tổng lợi nhuận kế toán trước thuế"], ["profit before tax","loi nhuan ke toan truoc thue"]),
    "NetIncome_Q": ("income_statement", ["IS_NET_PROFIT_AFTER_TAX"], ["Lợi nhuận sau thuế thu nhập doanh nghiệp"], ["net profit after tax","loi nhuan sau thue"]),
}

errors, frames = [], []
for i, symbol in enumerate(SYMBOLS, 1):
    print(f"[{i}/{len(SYMBOLS)}] {symbol}")
    for report in ("balance_sheet","income_statement"):
        try:
            frames.append(canonical(fetch(symbol, report)))
        except Exception as e:
            errors.append({"symbol":symbol,"industry":INDUSTRY[symbol],"report":report,"error_type":type(e).__name__,"error":str(e)})
    time.sleep(0.25)

if not frames:
    raise RuntimeError("No data returned")
raw = pd.concat(frames, ignore_index=True)
raw = raw[
    ((raw["report"].eq("balance_sheet")) & (raw["qperiod"].isna() | (raw["qperiod"] >= START_BS))) |
    ((raw["report"].eq("income_statement")) & (raw["qperiod"].isna() | (raw["qperiod"] >= START_PANEL)))
].copy()

smoke = raw[raw["symbol"].eq("HPG")].copy()
mapping, map_rows = {}, []
for field, (report, ids, names, contains) in FIELD_SPECS.items():
    sid, label, method = resolve_id(smoke, report, ids, names, contains)
    mapping[field] = sid
    map_rows.append({"field":field,"report":report,"resolved_id":sid,"resolved_name_hpg":label,"method":method})
map_df = pd.DataFrame(map_rows)
print(map_df.to_string(index=False))

needed = ["Inventory_End","TotalAssets","Revenue_Q","COGS_Q","NetIncome_Q"]
missing_core = [x for x in needed if not mapping.get(x)]
if missing_core:
    raise RuntimeError(f"Missing core semantic IDs on HPG: {missing_core}")

grid = pd.MultiIndex.from_product([SYMBOLS, PANEL_PERIODS], names=["Symbol","Quarter"]).to_frame(index=False)
grid["Industry"] = grid["Symbol"].map(INDUSTRY)
grid["QuarterEnd"] = grid["Quarter"].map(lambda p: p.end_time.normalize())
grid["DaysInQuarter"] = grid["Quarter"].map(lambda p: (p.end_time.normalize()-p.start_time.normalize()).days+1)
panel = grid.copy()

for field, sid in mapping.items():
    if not sid:
        panel[field] = np.nan
        continue
    x = raw[(raw["id"].eq(sid)) & raw["qperiod"].notna()].copy()
    x = x[(x["qperiod"] >= START_BS) & (x["qperiod"] <= END_PANEL)]
    x = x.groupby(["symbol","qperiod"], as_index=False)["value"].last()
    x = x.rename(columns={"symbol":"Symbol","qperiod":"Quarter","value":field})
    panel = panel.merge(x, on=["Symbol","Quarter"], how="left")

for field, lagcol in {
    "Inventory_End":"Inventory_Lag",
    "TotalAssets":"TotalAssets_Lag",
    "TradeReceivables":"TradeReceivables_Lag",
    "ShortReceivables":"ShortReceivables_Lag",
    "TradePayables":"TradePayables_Lag",
}.items():
    sid = mapping.get(field)
    if not sid:
        panel[lagcol] = np.nan
        continue
    x = raw[(raw["id"].eq(sid)) & raw["qperiod"].notna()].copy()
    x = x[(x["qperiod"] >= START_BS) & (x["qperiod"] <= END_PANEL)]
    x = x.groupby(["symbol","qperiod"], as_index=False)["value"].last()
    x["Quarter"] = x["qperiod"] + 1
    x = x.rename(columns={"symbol":"Symbol","value":lagcol})[["Symbol","Quarter",lagcol]]
    panel = panel.merge(x, on=["Symbol","Quarter"], how="left")

panel["AvgInventory"] = (panel["Inventory_End"] + panel["Inventory_Lag"]) / 2
panel["AvgTotalAssets"] = (panel["TotalAssets"] + panel["TotalAssets_Lag"]) / 2
panel["Receivables"] = panel["TradeReceivables"].where(panel["TradeReceivables"].notna(), panel["ShortReceivables"])
panel["Receivables_Lag"] = panel["TradeReceivables_Lag"].where(panel["TradeReceivables_Lag"].notna(), panel["ShortReceivables_Lag"])
panel["AvgReceivables"] = (panel["Receivables"] + panel["Receivables_Lag"]) / 2
panel["AvgTradePayables"] = (panel["TradePayables"] + panel["TradePayables_Lag"]) / 2

valid_cogs = panel["COGS_Q"] > 0
valid_rev = panel["Revenue_Q"] > 0
panel["DIO"] = np.where(valid_cogs, panel["AvgInventory"] / panel["COGS_Q"] * panel["DaysInQuarter"], np.nan)
panel["Inv_Assets"] = panel["Inventory_End"] / panel["TotalAssets"]
panel["Inv_Sales"] = np.where(valid_rev, panel["Inventory_End"] / panel["Revenue_Q"], np.nan)
panel["Size_lnAssets"] = np.where(panel["TotalAssets"] > 0, np.log(panel["TotalAssets"]), np.nan)
panel["Leverage"] = panel["TotalLiabilities"] / panel["TotalAssets"]
panel["Cash_Ratio"] = panel["Cash"] / panel["TotalAssets"]
panel["ROA"] = panel["NetIncome_Q"] / panel["AvgTotalAssets"]

lag4 = panel[["Symbol","Quarter","Revenue_Q"]].copy()
lag4["Quarter"] = lag4["Quarter"] + 4
lag4 = lag4.rename(columns={"Revenue_Q":"Revenue_Q_lag4"})
panel = panel.merge(lag4, on=["Symbol","Quarter"], how="left")
panel["Sales_Growth"] = panel["Revenue_Q"] / panel["Revenue_Q_lag4"] - 1
panel["DSO"] = np.where(valid_rev, panel["AvgReceivables"] / panel["Revenue_Q"] * panel["DaysInQuarter"], np.nan)
panel["DPO"] = np.where(valid_cogs, panel["AvgTradePayables"] / panel["COGS_Q"] * panel["DaysInQuarter"], np.nan)
panel["CCC"] = panel["DIO"] + panel["DSO"] - panel["DPO"]
panel["NonCalendarFY_Flag"] = panel["Symbol"].eq("HSG")

quality = []
for symbol in SYMBOLS:
    p = panel[panel["Symbol"].eq(symbol)]
    complete = int(p[["Inventory_End","COGS_Q","Revenue_Q","TotalAssets"]].notna().all(axis=1).sum())
    status = "PASS" if complete >= 22 else ("PARTIAL" if complete > 0 else "FAIL")
    if symbol == "HSG":
        status = "NON_CALENDAR_FY"
    quality.append({
        "ticker":symbol,"industry":INDUSTRY[symbol],"expected_quarters":24,
        "core_complete_quarters":complete,"coverage_pct":complete/24,
        "inventory_missing_pct":p["Inventory_End"].isna().mean(),
        "cogs_missing_pct":p["COGS_Q"].isna().mean(),
        "revenue_missing_pct":p["Revenue_Q"].isna().mean(),
        "assets_missing_pct":p["TotalAssets"].isna().mean(),
        "dio_missing_pct":p["DIO"].isna().mean(),"status":status,
    })
quality_df = pd.DataFrame(quality)

recon = {"symbol":"HPG","year":2025}
try:
    annual = canonical(fetch("HPG","income_statement","year"))
    sid = mapping["Revenue_Q"]
    fy = annual[(annual["id"].eq(sid)) & annual["period"].map(year_of).eq(2025)]["value"]
    fyv = float(fy.iloc[0]) if len(fy) else np.nan
    qsum = panel[(panel["Symbol"].eq("HPG")) & panel["Quarter"].astype(str).str.startswith("2025")]["Revenue_Q"].sum(min_count=4)
    recon.update({"quarter_sum":qsum,"annual_value":fyv,"difference":qsum-fyv if pd.notna(qsum) and pd.notna(fyv) else np.nan,"difference_pct":qsum/fyv-1 if pd.notna(qsum) and pd.notna(fyv) and fyv else np.nan})
except Exception as e:
    recon["error"] = f"{type(e).__name__}: {e}"

raw_out = raw.drop(columns=["qperiod"]).copy()
panel_out = panel.copy()
panel_out["Quarter"] = panel_out["Quarter"].astype(str)
errors_df = pd.DataFrame(errors, columns=["symbol","industry","report","error_type","error"])

raw_out.to_csv(OUT/"gpr_dio_vnstock_raw.csv", index=False, encoding="utf-8-sig")
panel_out.to_csv(OUT/"gpr_dio_financial_panel.csv", index=False, encoding="utf-8-sig")
quality_df.to_csv(OUT/"gpr_dio_data_quality.csv", index=False, encoding="utf-8-sig")
map_df.to_csv(OUT/"gpr_dio_semantic_mapping.csv", index=False, encoding="utf-8-sig")
errors_df.to_csv(OUT/"gpr_dio_error_log.csv", index=False, encoding="utf-8-sig")
raw_out.to_parquet(OUT/"gpr_dio_vnstock_raw.parquet", index=False)
panel_out.to_parquet(OUT/"gpr_dio_financial_panel.parquet", index=False)

summary = {
    "symbols_expected":len(SYMBOLS),
    "symbols_with_core_data":int((quality_df["core_complete_quarters"]>0).sum()),
    "panel_rows_expected":len(SYMBOLS)*24,
    "panel_rows":len(panel_out),
    "pass":int((quality_df["status"]=="PASS").sum()),
    "partial":int((quality_df["status"]=="PARTIAL").sum()),
    "fail":int((quality_df["status"]=="FAIL").sum()),
    "non_calendar_fy":int((quality_df["status"]=="NON_CALENDAR_FY").sum()),
    "errors":len(errors_df),
    "hpg_reconciliation":recon,
    "semantic_mapping":map_df.to_dict(orient="records"),
}
(OUT/"run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("RUN_SUMMARY_BEGIN")
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print("RUN_SUMMARY_END")
