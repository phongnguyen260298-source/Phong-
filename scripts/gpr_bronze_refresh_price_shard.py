from __future__ import annotations
import os,time,json
from pathlib import Path
import pandas as pd
from vnstock_data import Quote

ROOT=Path(os.getenv('BASELINE_ROOT','input/vci42'))
OUT=Path('data/gpr_bronze_refresh/price_shards'); OUT.mkdir(parents=True,exist_ok=True)
SHARD=int(os.getenv('SHARD','0')); N=int(os.getenv('N_SHARDS','4')); PAUSE=float(os.getenv('REQUEST_PAUSE','1.6'))
syms=sorted({p.stem.upper().strip() for p in ROOT.glob('chunk_*/fundamental_quarter/balance_sheet/*.parquet')})
sel=syms[SHARD::N]
rows=[]; errs=[]; cov=[]
for i,sym in enumerate(sel,1):
    last=None; d=None
    for a in range(4):
        try:
            d=Quote(source='KBS',symbol=sym).history(start='2020-01-01',end='2025-12-31',interval='1D')
            if d is None or d.empty: raise RuntimeError('empty price')
            break
        except Exception as e:
            last=e
            if a<3: time.sleep(min(8,2**(a+1)))
    if d is None or d.empty:
        errs.append({'symbol':sym,'error_type':type(last).__name__ if last else 'Unknown','error':str(last)})
        cov.append({'symbol':sym,'n':0,'start':None,'end':None}); time.sleep(PAUSE); continue
    d=d.copy(); d['symbol']=sym
    datecol='time' if 'time' in d.columns else ('date' if 'date' in d.columns else None)
    if not datecol: raise RuntimeError(f'no date column for {sym}: {list(d.columns)}')
    d['date']=pd.to_datetime(d[datecol],errors='coerce').dt.normalize()
    closecol='close' if 'close' in d.columns else next((c for c in d.columns if str(c).lower()=='close'),None)
    if not closecol: raise RuntimeError(f'no close column for {sym}')
    d['close']=pd.to_numeric(d[closecol],errors='coerce')
    z=d[['symbol','date','close']].dropna().drop_duplicates(['symbol','date']).sort_values('date')
    rows.append(z); cov.append({'symbol':sym,'n':len(z),'start':z.date.min(),'end':z.date.max()})
    print(f'[{SHARD}] {i}/{len(sel)} {sym} n={len(z)}',flush=True); time.sleep(PAUSE)

px=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['symbol','date','close'])
px.to_parquet(OUT/f'price_shard_{SHARD}.parquet',index=False)
pd.DataFrame(cov).to_csv(OUT/f'coverage_shard_{SHARD}.csv',index=False)
pd.DataFrame(errs,columns=['symbol','error_type','error']).to_csv(OUT/f'errors_shard_{SHARD}.csv',index=False)
meta={'tier':'Bronze','source':'KBS','shard':SHARD,'n_shards':N,'universe':len(syms),'assigned':len(sel),'rows':len(px),'errors':len(errs),'pause_sec':PAUSE}
(OUT/f'meta_shard_{SHARD}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False))