from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from vnstock import Finance, register_user

OUT = Path('output/community')
OUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = {
    'STEEL': ['DTL','HMC','HPG','HSG','NKG','SMC','TLH','VCA','KMT','SSM','VGS'],
    'FERTILIZER': ['BFC','DCM','DPM','SFG','DDV','LAS'],
    'TEXTILE': ['AAT','ADS','EVE','GIL','GMC','HTG','KMR','MSH','STK','SVD','TCM','TVT','TDT','TET','TNG'],
    'SEAFOOD': ['AAM','ABT','ACL','ANV','CMX','FMC','IDI','KHS','SJ1','VHC'],
}
INDUSTRY = {s:g for g, arr in UNIVERSE.items() for s in arr}
SYMBOLS = [s for arr in UNIVERSE.values() for s in arr]


def norm(x: object) -> str:
    s = '' if x is None else str(x)
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = s.lower().replace('đ','d')
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()


def period_cols(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if re.fullmatch(r'20\d{2}-Q[1-4](?:_\d+)?', str(c))]


def safe_call(obj, method: str, attempts: int = 4) -> pd.DataFrame:
    err = None
    for k in range(attempts):
        try:
            return getattr(obj, method)(period='quarter', lang='vi', dropna=False, show_log=False)
        except TypeError:
            try:
                return getattr(obj, method)(period='quarter', dropna=False, show_log=False)
            except Exception as e:
                err = e
        except Exception as e:
            err = e
        print(f'[retry] {method} attempt={k+1}: {type(err).__name__}: {err}')
        time.sleep(min(12, 2 ** k))
    raise err


def to_long(df: pd.DataFrame, symbol: str, report: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    pcs = period_cols(df)
    if not pcs:
        return pd.DataFrame()
    meta = [c for c in ['item','item_en','item_id'] if c in df.columns]
    x = df[meta + pcs].copy()
    x = x.melt(id_vars=meta, value_vars=pcs, var_name='period', value_name='value')
    x['symbol'] = symbol
    x['industry'] = INDUSTRY[symbol]
    x['report'] = report
    x['source'] = 'VCI'
    x['retrieval_utc'] = datetime.now(timezone.utc).isoformat()
    return x[['symbol','industry','report','period'] + meta + ['value','source','retrieval_utc']]


def pick_row(df: pd.DataFrame, ids: list[str], names: list[str], contains: list[str] | None = None):
    if df is None or df.empty:
        return None
    contains = contains or []
    iid = df['item_id'].astype(str).map(norm) if 'item_id' in df.columns else pd.Series('', index=df.index)
    item = df['item'].astype(str).map(norm) if 'item' in df.columns else pd.Series('', index=df.index)
    ien = df['item_en'].astype(str).map(norm) if 'item_en' in df.columns else pd.Series('', index=df.index)

    for sid in ids:
        m = iid.eq(norm(sid))
        if m.any():
            return df.loc[m].iloc[0]
    for name in names:
        n = norm(name)
        m = item.eq(n) | ien.eq(n)
        if m.any():
            return df.loc[m].iloc[0]
    for phrase in contains:
        words = norm(phrase).split()
        m = item.map(lambda s: all(w in s for w in words)) | ien.map(lambda s: all(w in s for w in words))
        if m.any():
            h = df.loc[m].copy()
            h['_len'] = h.get('item', '').astype(str).str.len()
            return h.sort_values('_len').iloc[0]
    return None


FIELD_SPECS = {
    'Inventory_End': ('balance_sheet', ['inventories_net','inventories'], ['Hàng tồn kho, ròng','Hàng tồn kho'], ['inventory']),
    'TotalAssets': ('balance_sheet', ['total_assets'], ['Tổng cộng tài sản','Tổng tài sản'], ['total assets']),
    'Cash': ('balance_sheet', ['cash_and_cash_equivalents'], ['Tiền và tương đương tiền','Tiền và các khoản tương đương tiền'], ['cash and cash equivalents']),
    'AccountsReceivable': ('balance_sheet', ['accounts_receivable'], ['Các khoản phải thu'], ['accounts receivable']),
    'TradeReceivables': ('balance_sheet', ['trade_accounts_receivable'], ['Phải thu khách hàng'], ['trade accounts receivable']),
    'TotalLiabilities': ('balance_sheet', ['total_liabilities','liabilities'], ['Nợ phải trả','Tổng nợ phải trả'], ['total liabilities']),
    'TradePayables': ('balance_sheet', ['trade_accounts_payable','accounts_payable','trade_payables'], ['Phải trả người bán'], ['trade accounts payable','accounts payable']),
    'ShortDebt': ('balance_sheet', ['short_term_borrowings','short_term_debt'], ['Vay và nợ thuê tài chính ngắn hạn'], ['short term borrowings']),
    'LongDebt': ('balance_sheet', ['long_term_borrowings','long_term_debt'], ['Vay và nợ thuê tài chính dài hạn'], ['long term borrowings']),
    'TotalEquity': ('balance_sheet', ['total_equity','owners_equity','equity'], ['Vốn chủ sở hữu'], ['total equity']),
    'Revenue_Q': ('income_statement', ['net_sales','revenue'], ['Doanh thu thuần','Doanh thu thuần về bán hàng và cung cấp dịch vụ'], ['net sales']),
    'COGS_Q': ('income_statement', ['cost_of_sales','cost_of_goods_sold'], ['Giá vốn hàng bán'], ['cost of sales','cost of goods sold']),
    'GrossProfit': ('income_statement', ['gross_profit'], ['Lợi nhuận gộp'], ['gross profit']),
    'ProfitBeforeTax': ('income_statement', ['net_accounting_profit_loss_before_tax','profit_before_tax'], ['Lợi nhuận trước thuế','Tổng lợi nhuận kế toán trước thuế'], ['profit before tax']),
    'NetIncome_Q': ('income_statement', ['net_profit_loss_after_tax','net_profit'], ['Lợi nhuận sau thuế thu nhập doanh nghiệp','Lãi/(lỗ) thuần sau thuế'], ['net profit after tax']),
}

key = os.getenv('VNSTOCK_API_KEY')
if not key:
    raise RuntimeError('VNSTOCK_API_KEY missing')
registered = register_user(api_key=key)
print('register_user=', registered)

raw_frames: list[pd.DataFrame] = []
errors: list[dict] = []
wide_store: dict[tuple[str,str], pd.DataFrame] = {}

for i, symbol in enumerate(SYMBOLS, 1):
    print(f'[{i}/{len(SYMBOLS)}] {symbol}')
    try:
        f = Finance(symbol=symbol, source='VCI')
        time.sleep(1.1)
    except Exception as e:
        errors.append({'symbol':symbol,'report':'INIT','error_type':type(e).__name__,'error':str(e)})
        continue

    for report in ['balance_sheet','income_statement']:
        try:
            df = safe_call(f, report)
            wide_store[(symbol,report)] = df
            raw_frames.append(to_long(df, symbol, report))
            print(report, 'shape=', getattr(df,'shape',None), 'periods=', period_cols(df))
        except Exception as e:
            errors.append({'symbol':symbol,'report':report,'error_type':type(e).__name__,'error':str(e)})
        time.sleep(1.2)

raw = pd.concat([x for x in raw_frames if x is not None and not x.empty], ignore_index=True) if any((x is not None and not x.empty) for x in raw_frames) else pd.DataFrame()
if raw.empty:
    raise RuntimeError('No financial data returned from VCI')

rows = []
map_rows = []
for symbol in SYMBOLS:
    b = wide_store.get((symbol,'balance_sheet'), pd.DataFrame())
    inc = wide_store.get((symbol,'income_statement'), pd.DataFrame())
    source_df = {'balance_sheet':b,'income_statement':inc}
    all_periods = sorted(set(period_cols(b) + period_cols(inc)), key=lambda s: pd.Period(s.split('_')[0], freq='Q'))
    field_values: dict[str, dict[str,float]] = {}
    for field,(report,ids,names,contains) in FIELD_SPECS.items():
        df = source_df[report]
        r = pick_row(df, ids, names, contains)
        if r is None:
            field_values[field] = {}
            map_rows.append({'symbol':symbol,'field':field,'report':report,'resolved_item_id':None,'resolved_item':None})
        else:
            field_values[field] = {p: pd.to_numeric(r.get(p), errors='coerce') for p in period_cols(df)}
            map_rows.append({'symbol':symbol,'field':field,'report':report,'resolved_item_id':r.get('item_id'),'resolved_item':r.get('item')})

    for p in all_periods:
        base_p = p.split('_')[0]
        rec = {'Symbol':symbol,'Industry':INDUSTRY[symbol],'Quarter':base_p,'Source':'VCI','NonCalendarFY_Flag':symbol=='HSG'}
        for field in FIELD_SPECS:
            rec[field] = field_values[field].get(p, field_values[field].get(base_p, np.nan))
        rows.append(rec)

panel = pd.DataFrame(rows)
panel['QuarterP'] = panel['Quarter'].map(lambda x: pd.Period(x, freq='Q'))
panel = panel.sort_values(['Symbol','QuarterP']).drop_duplicates(['Symbol','Quarter'], keep='last').reset_index(drop=True)
panel['QuarterEnd'] = panel['QuarterP'].map(lambda p: p.end_time.normalize())
panel['DaysInQuarter'] = panel['QuarterP'].map(lambda p: (p.end_time.normalize()-p.start_time.normalize()).days+1)

# VCI reports cost of sales as a negative expense; research COGS denominator is positive.
panel['COGS_Q_raw'] = panel['COGS_Q']
panel['COGS_Q'] = panel['COGS_Q'].abs()

for col, lagname in [
    ('Inventory_End','Inventory_Lag'),
    ('TotalAssets','TotalAssets_Lag'),
    ('TradeReceivables','TradeReceivables_Lag'),
    ('AccountsReceivable','AccountsReceivable_Lag'),
    ('TradePayables','TradePayables_Lag'),
]:
    panel[lagname] = panel.groupby('Symbol')[col].shift(1)

prev_q = panel.groupby('Symbol')['QuarterP'].shift(1)
consecutive = (panel['QuarterP'].astype('int64') - prev_q.astype('Int64')) == 1
for lagname in ['Inventory_Lag','TotalAssets_Lag','TradeReceivables_Lag','AccountsReceivable_Lag','TradePayables_Lag']:
    panel.loc[~consecutive.fillna(False), lagname] = np.nan

panel['AvgInventory'] = (panel['Inventory_End'] + panel['Inventory_Lag'])/2
panel['AvgTotalAssets'] = (panel['TotalAssets'] + panel['TotalAssets_Lag'])/2
panel['Receivables'] = panel['TradeReceivables'].where(panel['TradeReceivables'].notna(), panel['AccountsReceivable'])
panel['Receivables_Lag'] = panel['TradeReceivables_Lag'].where(panel['TradeReceivables_Lag'].notna(), panel['AccountsReceivable_Lag'])
panel['AvgReceivables'] = (panel['Receivables'] + panel['Receivables_Lag'])/2
panel['AvgTradePayables'] = (panel['TradePayables'] + panel['TradePayables_Lag'])/2

valid_cogs = panel['COGS_Q'] > 0
valid_rev = panel['Revenue_Q'] > 0
panel['DIO'] = np.where(valid_cogs, panel['AvgInventory']/panel['COGS_Q']*panel['DaysInQuarter'], np.nan)
panel['Inv_Assets'] = panel['Inventory_End']/panel['TotalAssets']
panel['Inv_Sales'] = np.where(valid_rev, panel['Inventory_End']/panel['Revenue_Q'], np.nan)
panel['Size_lnAssets'] = np.where(panel['TotalAssets']>0, np.log(panel['TotalAssets']), np.nan)
panel['Leverage'] = panel['TotalLiabilities']/panel['TotalAssets']
panel['Cash_Ratio'] = panel['Cash']/panel['TotalAssets']
panel['ROA'] = panel['NetIncome_Q']/panel['AvgTotalAssets']
panel['Revenue_Q_lag4'] = panel.groupby('Symbol')['Revenue_Q'].shift(4)
panel['Sales_Growth'] = panel['Revenue_Q']/panel['Revenue_Q_lag4'] - 1
panel['DSO'] = np.where(valid_rev, panel['AvgReceivables']/panel['Revenue_Q']*panel['DaysInQuarter'], np.nan)
panel['DPO'] = np.where(valid_cogs, panel['AvgTradePayables']/panel['COGS_Q']*panel['DaysInQuarter'], np.nan)
panel['CCC'] = panel['DIO'] + panel['DSO'] - panel['DPO']

quality = []
for symbol in SYMBOLS:
    p = panel[panel['Symbol'].eq(symbol)].copy()
    n = len(p)
    dio_n = int(p['DIO'].notna().sum()) if n else 0
    status = 'FAIL' if n == 0 else ('PASS' if dio_n >= 5 else 'PARTIAL')
    if symbol == 'HSG' and n:
        status = 'NON_CALENDAR_FY'
    quality.append({
        'Symbol':symbol,
        'Industry':INDUSTRY[symbol],
        'first_quarter':None if n==0 else str(p['Quarter'].min()),
        'last_quarter':None if n==0 else str(p['Quarter'].max()),
        'quarters_returned':n,
        'DIO_available':dio_n,
        'inventory_missing_pct':None if n==0 else float(p['Inventory_End'].isna().mean()),
        'cogs_missing_pct':None if n==0 else float(p['COGS_Q'].isna().mean()),
        'revenue_missing_pct':None if n==0 else float(p['Revenue_Q'].isna().mean()),
        'assets_missing_pct':None if n==0 else float(p['TotalAssets'].isna().mean()),
        'status':status,
    })
quality_df = pd.DataFrame(quality)
map_df = pd.DataFrame(map_rows)
errors_df = pd.DataFrame(errors)

# Preserve raw data independently from derived panel.
raw.to_parquet(OUT/'gpr_dio_vnstock_community_raw.parquet', index=False)
raw.to_csv(OUT/'gpr_dio_vnstock_community_raw.csv', index=False, encoding='utf-8-sig')
panel.drop(columns=['QuarterP']).to_excel(OUT/'gpr_dio_vnstock_community_panel.xlsx', index=False)
quality_df.to_excel(OUT/'gpr_dio_vnstock_community_quality.xlsx', index=False)
map_df.to_excel(OUT/'gpr_dio_vnstock_community_field_mapping.xlsx', index=False)
errors_df.to_excel(OUT/'gpr_dio_vnstock_community_errors.xlsx', index=False)

summary = {
    'retrieval_utc': datetime.now(timezone.utc).isoformat(),
    'edition': 'VNStock Community',
    'source': 'VCI',
    'company_count_requested': len(SYMBOLS),
    'company_count_with_any_data': int(quality_df['quarters_returned'].gt(0).sum()),
    'company_count_pass': int(quality_df['status'].eq('PASS').sum()),
    'panel_rows': int(len(panel)),
    'DIO_nonmissing': int(panel['DIO'].notna().sum()),
    'min_quarter': None if panel.empty else str(panel['Quarter'].min()),
    'max_quarter': None if panel.empty else str(panel['Quarter'].max()),
    'community_financial_statement_limit': 8,
    'target_2020Q1_2025Q4_fully_available_from_current_key': False,
    'note': 'Community edition is limited by VNStock to the latest 8 financial-statement periods. This script does not bypass that product limit.'
}
(OUT/'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
