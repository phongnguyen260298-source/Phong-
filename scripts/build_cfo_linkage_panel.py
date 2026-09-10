from __future__ import annotations
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

ROOT=Path('input/vci42')
OUT=Path('output/xy_screen'); OUT.mkdir(parents=True,exist_ok=True)

TARGETS={
    'balance_sheet': {
        'BS_TOTAL_ASSETS':'assets',
        'BS_TOTAL_LIABILITIES':'liabilities',
    },
    'cash_flow': {
        'CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES':'cfo',
    },
}

def period_q(v):
    s=str(v)
    m=re.search(r'(20\d{2}|19\d{2})\D*Q\D*([1-4])',s,re.I)
    if not m:
        m=re.search(r'(20\d{2}|19\d{2}).*?([1-4])',s)
    return f'{m.group(1)}Q{m.group(2)}' if m else None

def load(report, mapping):
    frames=[]; files=sorted(ROOT.glob(f'chunk_*/fundamental_quarter/{report}/*.parquet'))
    for f in files:
        d=pd.read_parquet(f,columns=['period','item_id','value'])
        d=d[d.item_id.astype(str).isin(mapping)].copy()
        if d.empty: continue
        d['symbol']=f.stem.upper().strip()
        d['period_q']=d.period.map(period_q)
        d['metric']=d.item_id.astype(str).map(mapping)
        d['value']=pd.to_numeric(d.value,errors='coerce')
        frames.append(d[['symbol','period_q','metric','value']])
    if not frames:
        raise RuntimeError(f'No usable {report} rows from {len(files)} files')
    z=pd.concat(frames,ignore_index=True).dropna(subset=['period_q'])
    # The exact VCI item_id is authoritative. If the artifact contains duplicate rows,
    # keep the last retrieval value at the same firm-quarter-item grain.
    z=z.drop_duplicates(['symbol','period_q','metric'],keep='last')
    return z.pivot(index=['symbol','period_q'],columns='metric',values='value').reset_index(), len(files)

bs,n_bs=load('balance_sheet',TARGETS['balance_sheet'])
cf,n_cf=load('cash_flow',TARGETS['cash_flow'])
p=bs.merge(cf,on=['symbol','period_q'],how='outer',validate='one_to_one')
p[['year','q']]=p.period_q.str.extract(r'(\d{4})Q([1-4])').astype(float)
p=p.sort_values(['symbol','year','q'])
g=p.groupby('symbol',group_keys=False)
p['avg_assets']=(p.assets+g.assets.shift(1))/2
p['cfo_assets']=p.cfo/p.avg_assets
p['leverage']=p.liabilities/p.assets
p['log_assets']=np.log(p.assets.where(p.assets>0))
for c in ['cfo_assets','leverage']:
    p.loc[p[c].abs()>5,c]=np.nan
p=p[(p.year>=2020)&(p.year<=2025)].copy()
cols=['symbol','period_q','year','q','assets','liabilities','cfo','avg_assets','cfo_assets','leverage','log_assets']
p[cols].to_csv(OUT/'vci42_analysis_panel_2020_2025.csv',index=False,encoding='utf-8-sig')
qa={
    'source':'VCI v4.2 artifact vn-research-v42-vci-full, run 34329813990',
    'builder':'exact VCI item_id only',
    'balance_sheet_files':n_bs,'cash_flow_files':n_cf,
    'rows_2020_2025':int(len(p)),'symbols_2020_2025':int(p.symbol.nunique()),
    'period_min':str(p.period_q.min()),'period_max':str(p.period_q.max()),
    'duplicate_symbol_quarter':int(p.duplicated(['symbol','period_q']).sum()),
    'missing':{c:float(p[c].isna().mean()) for c in ['assets','liabilities','cfo','cfo_assets','leverage','log_assets']}
}
(OUT/'cfo_linkage_builder_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(qa,ensure_ascii=False,indent=2))
