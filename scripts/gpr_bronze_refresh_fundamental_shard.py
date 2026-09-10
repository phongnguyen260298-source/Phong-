from __future__ import annotations
import os,time,json,re
from pathlib import Path
import pandas as pd
from vnstock_data import Finance

ROOT=Path(os.getenv('BASELINE_ROOT','input/vci42'))
OUT=Path('data/gpr_bronze_refresh/fundamental_shards'); OUT.mkdir(parents=True,exist_ok=True)
SHARD=int(os.getenv('SHARD','0')); N=int(os.getenv('N_SHARDS','4')); PAUSE=float(os.getenv('REQUEST_PAUSE','1.6'))

# Baseline universe comes from every balance-sheet file in the locked VCI artifact.
syms=sorted({p.stem.upper().strip() for p in ROOT.glob('chunk_*/fundamental_quarter/balance_sheet/*.parquet')})
sel=syms[SHARD::N]

def pq(v):
    m=re.search(r'(20\d{2}|19\d{2})\D*Q\D*([1-4])',str(v),re.I)
    return f'{m.group(1)}Q{m.group(2)}' if m else None

def get(sym,report):
    f=Finance(source='VCI',symbol=sym); fn=getattr(f,report); last=None
    for a in range(4):
        try:
            d=fn(period='quarter',lang='vi',drop_empty=False,format='long')
            if d is None or d.empty: raise RuntimeError('empty dataframe')
            return d
        except Exception as e:
            last=e
            if a==3: raise
            time.sleep(min(8,2**(a+1)))
    raise last

rows=[]; errs=[]; cov=[]
for i,sym in enumerate(sel,1):
    repcov={}
    for report in ['balance_sheet','income_statement','cash_flow']:
        try:
            d=get(sym,report).copy(); d['symbol']=sym; d['report']=report
            if 'item_id' not in d.columns and 'id' in d.columns: d=d.rename(columns={'id':'item_id'})
            d['period_q']=d['period'].map(pq) if 'period' in d.columns else None
            d=d[d.period_q.notna()].copy(); d['year']=pd.to_numeric(d.period_q.str[:4],errors='coerce')
            # Need pre-treatment history plus outcome window.
            d=d[(d.year>=2017)&(d.year<=2025)].copy()
            repcov[report]=int(d.period_q.nunique())
            keep=[c for c in ['symbol','report','period','period_q','item_id','item','value','unit','order','level','off_balance'] if c in d.columns]
            rows.append(d[keep])
        except Exception as e:
            errs.append({'symbol':sym,'report':report,'error_type':type(e).__name__,'error':str(e)})
            repcov[report]=0
        time.sleep(PAUSE)
    cov.append({'symbol':sym,**{f'{k}_periods':v for k,v in repcov.items()}})
    print(f'[{SHARD}] {i}/{len(sel)} {sym} {repcov}',flush=True)

raw=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
raw.to_parquet(OUT/f'fundamental_shard_{SHARD}.parquet',index=False)
pd.DataFrame(cov).to_csv(OUT/f'coverage_shard_{SHARD}.csv',index=False)
pd.DataFrame(errs,columns=['symbol','report','error_type','error']).to_csv(OUT/f'errors_shard_{SHARD}.csv',index=False)
meta={'tier':'Bronze','source':'VCI','shard':SHARD,'n_shards':N,'universe':len(syms),'assigned':len(sel),'rows':len(raw),'errors':len(errs),'pause_sec':PAUSE}
(OUT/f'meta_shard_{SHARD}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False))