from __future__ import annotations
import os, time
from pathlib import Path
import pandas as pd
from vnstock_data import Quote

SHARD=int(os.getenv('SHARD','0'))
N_SHARDS=int(os.getenv('N_SHARDS','4'))
PAUSE=float(os.getenv('REQUEST_PAUSE','1.5'))
OUT=Path('data/gpr_market_top5_v4/shards'); OUT.mkdir(parents=True, exist_ok=True)
base=pd.read_csv('data/research_cohorts/vci578_preexposure.csv')
base['symbol']=base['symbol'].astype(str).str.upper()
symbols=base.symbol.dropna().drop_duplicates().tolist()
symbols=[s for i,s in enumerate(symbols) if i % N_SHARDS == SHARD]

START='2020-01-01'
END='2025-12-31'

def fetch_one(sym):
    last=None
    for k in range(5):
        try:
            q=Quote(source='VCI', symbol=sym)
            df=q.history(start=START, end=END, interval='1D')
            if df is None or df.empty:
                return sym,None,'EMPTY'
            tc=next((c for c in ['time','date','trading_date'] if c in df.columns),None)
            cc=next((c for c in ['close','Close','close_price'] if c in df.columns),None)
            if tc is None or cc is None:
                return sym,None,f'schema={list(df.columns)}'
            z=df[[tc,cc]].rename(columns={tc:'date',cc:'close'}).copy()
            z['date']=pd.to_datetime(z['date'],errors='coerce').dt.normalize()
            z['close']=pd.to_numeric(z['close'],errors='coerce')
            z=z.dropna().drop_duplicates('date').sort_values('date')
            z['symbol']=sym
            return sym,z,None
        except BaseException as e:
            last=e
            msg=str(e).lower()
            if 'limit' in msg or 'rate' in msg:
                time.sleep(25 + 15*k)
            else:
                time.sleep(5*(k+1))
    return sym,None,f'{type(last).__name__}:{last}'

frames=[]; errors=[]
for i,sym in enumerate(symbols,1):
    sym,z,e=fetch_one(sym)
    if z is not None:
        frames.append(z)
    else:
        errors.append({'symbol':sym,'error':e})
    print(f'shard {SHARD} {i}/{len(symbols)} {sym} rows={0 if z is None else len(z)} ok={z is not None}',flush=True)
    time.sleep(PAUSE)

if frames:
    pd.concat(frames,ignore_index=True).to_csv(OUT/f'price_shard_{SHARD}.csv.gz',index=False,compression='gzip')
pd.DataFrame(errors).to_csv(OUT/f'errors_shard_{SHARD}.csv',index=False)
print({'shard':SHARD,'target':len(symbols),'ok':len(frames),'errors':len(errors)},flush=True)
