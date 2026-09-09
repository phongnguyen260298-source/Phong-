from __future__ import annotations
import os, time
from pathlib import Path
import pandas as pd
from vnstock import Market

SHARD=int(os.getenv('SHARD','0')); N_SHARDS=int(os.getenv('N_SHARDS','20'))
PAUSE=float(os.getenv('REQUEST_PAUSE','4.2'))
OUT=Path('data/gpr_market_top5_v4/shards'); OUT.mkdir(parents=True, exist_ok=True)
base=pd.read_csv('data/research_cohorts/vci578_preexposure.csv')
base['symbol']=base['symbol'].astype(str).str.upper()
symbols=base.symbol.dropna().drop_duplicates().tolist()
symbols=[s for i,s in enumerate(symbols) if i % N_SHARDS == SHARD]
# Community endpoint caps each response at ~100 bars, so split 2020-2025 into <=4-month windows.
WINDOWS=[]
for y in range(2020,2026):
    WINDOWS += [(f'{y}-01-01',f'{y}-04-30'),(f'{y}-05-01',f'{y}-08-31'),(f'{y}-09-01',f'{y}-12-31')]

def one_window(sym,start,end):
    last=None
    for k in range(4):
        try:
            df=Market().equity(sym).ohlcv(start=start,end=end,interval='1D')
            if df is None or df.empty: return None,None
            tc=next((c for c in ['time','date','trading_date'] if c in df.columns),None)
            cc=next((c for c in ['close','Close','close_price'] if c in df.columns),None)
            if tc is None or cc is None: return None,f'schema={list(df.columns)}'
            z=df[[tc,cc]].rename(columns={tc:'date',cc:'close'}).copy()
            z['date']=pd.to_datetime(z['date'],errors='coerce').dt.normalize()
            z['close']=pd.to_numeric(z['close'],errors='coerce')
            return z.dropna().drop_duplicates('date').sort_values('date'),None
        except BaseException as e:
            last=e; msg=str(e).lower()
            time.sleep(65 if ('limit' in msg or 'rate' in msg) else 15*(k+1))
    return None,f'{type(last).__name__}:{last}'

def fetch_one(sym):
    parts=[]; errs=[]
    for start,end in WINDOWS:
        z,e=one_window(sym,start,end)
        if z is not None and not z.empty: parts.append(z)
        if e: errs.append(f'{start}:{e}')
        time.sleep(PAUSE)
    if not parts:
        return sym,None,'; '.join(errs) if errs else 'EMPTY_ALL_WINDOWS'
    z=pd.concat(parts,ignore_index=True).drop_duplicates('date').sort_values('date')
    z['symbol']=sym
    return sym,z,'; '.join(errs) if errs else None

frames=[]; errors=[]; warnings=[]
for i,sym0 in enumerate(symbols,1):
    sym,z,e=fetch_one(sym0)
    if z is not None: frames.append(z)
    else: errors.append({'symbol':sym,'error':e})
    if z is not None and e: warnings.append({'symbol':sym,'warning':e})
    print(f'shard {SHARD} {i}/{len(symbols)} {sym} rows={0 if z is None else len(z)} ok={z is not None}',flush=True)

if frames:
    pd.concat(frames,ignore_index=True).to_csv(OUT/f'price_shard_{SHARD}.csv.gz',index=False,compression='gzip')
pd.DataFrame(errors).to_csv(OUT/f'errors_shard_{SHARD}.csv',index=False)
pd.DataFrame(warnings).to_csv(OUT/f'warnings_shard_{SHARD}.csv',index=False)
print({'shard':SHARD,'target':len(symbols),'ok':len(frames),'errors':len(errors),'warnings':len(warnings)},flush=True)
