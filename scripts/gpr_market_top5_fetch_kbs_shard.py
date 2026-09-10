from __future__ import annotations
import json, os, random, time
from pathlib import Path
import pandas as pd
from vnstock import Quote

SHARD=int(os.getenv('SHARD','0'))
N_SHARDS=int(os.getenv('N_SHARDS','4'))
PAUSE=float(os.getenv('REQUEST_PAUSE','5.0'))
START='2020-01-01'; END='2025-12-31'
OUT=Path('data/gpr_market_top5_v4/shards'); OUT.mkdir(parents=True,exist_ok=True)

base=pd.read_csv('data/research_cohorts/vci578_preexposure.csv')
base['symbol']=base['symbol'].astype(str).str.upper().str.strip()
all_symbols=base.symbol.dropna().drop_duplicates().tolist()
symbols=[s for i,s in enumerate(all_symbols) if i % N_SHARDS == SHARD]

# De-synchronize runner request bursts while keeping aggregate rate below 60 rpm.
time.sleep(random.uniform(0,2.5))

def fetch_one(sym):
    last=None
    for k in range(5):
        t0=time.monotonic()
        try:
            q=Quote(symbol=sym,source='KBS')
            d=q.history(start=START,end=END,interval='1D')
            if d is None or d.empty:
                return sym,None,'EMPTY'
            tc=next((c for c in ['time','date','trading_date'] if c in d.columns),None)
            cc=next((c for c in ['close','Close','close_price'] if c in d.columns),None)
            if tc is None or cc is None:
                return sym,None,f'SCHEMA:{list(d.columns)}'
            z=d[[tc,cc]].rename(columns={tc:'date',cc:'close'}).copy()
            z['date']=pd.to_datetime(z['date'],errors='coerce').dt.normalize()
            z['close']=pd.to_numeric(z['close'],errors='coerce')
            z=z.dropna(subset=['date','close'])
            z=z[(z.date>=pd.Timestamp(START))&(z.date<=pd.Timestamp(END))]
            z=z.drop_duplicates('date').sort_values('date')
            if z.empty:
                return sym,None,'EMPTY_AFTER_FILTER'
            z['symbol']=sym
            # A complete 2020-2025 daily history should be far above 500 bars for firms listed early enough.
            # Do not reject young listings here; coverage is audited downstream.
            return sym,z,None
        except BaseException as e:
            last=e
            msg=str(e).lower()
            backoff=(18*(k+1) if ('rate' in msg or 'limit' in msg or '429' in msg) else 3*(k+1))
            time.sleep(backoff+random.uniform(0,2))
        finally:
            elapsed=time.monotonic()-t0
            if elapsed < PAUSE:
                time.sleep(PAUSE-elapsed)
    return sym,None,f'{type(last).__name__}:{last}'

frames=[]; errors=[]; coverage=[]
for i,sym in enumerate(symbols,1):
    sym,z,e=fetch_one(sym)
    if z is not None:
        frames.append(z)
        coverage.append({'symbol':sym,'n':len(z),'start':z.date.min(),'end':z.date.max()})
    else:
        errors.append({'symbol':sym,'error':e})
    print(f'KBS shard {SHARD} {i}/{len(symbols)} {sym} rows={0 if z is None else len(z)} ok={z is not None}',flush=True)

if frames:
    out=pd.concat(frames,ignore_index=True)
    raw=len(out); dup=int(out.duplicated(['symbol','date']).sum())
    out=out.drop_duplicates(['symbol','date']).sort_values(['symbol','date'])
    out.to_csv(OUT/f'price_shard_{SHARD}.csv.gz',index=False,compression='gzip')
else:
    raw=dup=0
pd.DataFrame(errors,columns=['symbol','error']).to_csv(OUT/f'errors_shard_{SHARD}.csv',index=False)
pd.DataFrame(coverage,columns=['symbol','n','start','end']).to_csv(OUT/f'coverage_shard_{SHARD}.csv',index=False)
meta={'shard':SHARD,'n_shards':N_SHARDS,'symbols_total_cohort':len(all_symbols),'target':len(symbols),'ok':len(frames),'errors':len(errors),'rows_raw':raw,'duplicate_symbol_date':dup,'source':'vnstock-community-KBS','start':START,'end':END}
(OUT/f'meta_shard_{SHARD}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(meta,flush=True)
if len(frames)==0:
    raise RuntimeError(f'KBS shard {SHARD} returned no usable price data')
