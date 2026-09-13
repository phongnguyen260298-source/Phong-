from pathlib import Path
import re, json
import numpy as np, pandas as pd

ROOT=Path('data/gpr_repro_pack_v1'); ROOT.mkdir(parents=True,exist_ok=True)
SH=Path('data/gpr_bronze_refresh/fundamental_shards')
raw=pd.concat([pd.read_parquet(f) for f in sorted(SH.glob('fundamental_shard_*.parquet'))],ignore_index=True)
raw['symbol']=raw.symbol.astype(str).str.upper().str.strip(); raw['period_q']=raw.period_q.astype(str); raw['value']=pd.to_numeric(raw.value,errors='coerce')
raw=raw.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
ids=['BS_TOTAL_ASSETS','BS_TOTAL_LIABILITIES','BS_SHORT_TERM_LIABILITIES','CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES']
finraw=raw[raw.item_id.isin(ids)].copy()
finraw=finraw[finraw.period_q.str.match(r'^(2017|2018|2019|2020|2021|2022|2023|2024|2025)Q[1-4]$',na=False)]
finraw[['year','quarter_num']]=finraw.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int)
finraw=finraw[['symbol','report','period_q','year','quarter_num','item_id','value']].sort_values(['symbol','period_q','item_id'])
finraw.to_csv(ROOT/'financial_raw_relevant.csv',index=False)

use={'BS_TOTAL_ASSETS':'assets','BS_TOTAL_LIABILITIES':'liabilities','BS_SHORT_TERM_LIABILITIES':'current_liab','CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES':'cfo'}
z=finraw.copy(); z['metric']=z.item_id.map(use)
wide=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index()
wide[['year','quarter_num']]=wide.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); wide=wide.sort_values(['symbol','year','quarter_num']); wide['pidx']=wide.year*4+wide.quarter_num
wide['avg_assets']=(wide.assets+wide.groupby('symbol').assets.shift(1))/2
wide['cfo_assets']=wide.cfo/wide.avg_assets; wide['leverage']=wide.liabilities/wide.assets; wide['log_assets']=np.log(wide.assets.where(wide.assets>0))
pre=wide[(wide.year>=2017)&(wide.year<=2019)].copy(); pre['liab_ratio']=pre.current_liab/pre.assets
stat=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),baseline_liab_ratio=('liab_ratio','median')).reset_index(); stat=stat[stat.pre_quarters>=2].copy()
for pct,name in [(0.60,'HL60'),(2/3,'HL67'),(0.75,'HL75')]:
    cut=stat.baseline_liab_ratio.quantile(pct); stat[name]=(stat.baseline_liab_ratio>=cut).astype(int); stat[name+'_cut']=cut
stat.to_csv(ROOT/'baseline_highliab.csv',index=False)

g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv')
def nt(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(pats):
    for p in pats:
        for c in g.columns:
            if nt(c)==nt(p) or nt(p) in nt(c): return c
    return None
dc=pick(['date','day']); th=pick(['THREATS_GPR_AI','GPR_THREATS']); ac=pick(['ACTS_GPR_AI','GPR_ACTS'])
if not all([dc,th,ac]): raise RuntimeError(list(g.columns))
g[dc]=pd.to_datetime(g[dc],errors='coerce'); g=g[(g[dc]>='2020-01-01')&(g[dc]<='2025-12-31')].copy()
# Preserve all official source columns for transparency, but put the three key columns first.
cols=[dc,th,ac]+[c for c in g.columns if c not in [dc,th,ac]]
g=g[cols]
g.to_csv(ROOT/'gpr_raw_daily_2020_2025.csv',index=False)

g2=g.rename(columns={dc:'date',th:'Threat',ac:'Acts'}).copy(); g2['period_q']=g2.date.dt.to_period('Q').astype(str)
qg=g2.groupby('period_q').agg(Threat_mean=('Threat','mean'),Threat_max=('Threat','max'),Threat_p90=('Threat',lambda s:s.quantile(.90)),Acts_mean=('Acts','mean'),Acts_max=('Acts','max'),Acts_p90=('Acts',lambda s:s.quantile(.90))).reset_index()
thr90=float(g2.Threat.quantile(.90)); act90=float(g2.Acts.quantile(.90))
sp1=g2.assign(x=(g2.Threat>=thr90).astype(float)).groupby('period_q').x.mean().reset_index(name='Threat_spikeshare')
sp2=g2.assign(x=(g2.Acts>=act90).astype(float)).groupby('period_q').x.mean().reset_index(name='Acts_spikeshare')
qg=qg.merge(sp1,on='period_q').merge(sp2,on='period_q')
for c in ['Threat_mean','Threat_max','Threat_p90','Threat_spikeshare','Acts_mean','Acts_max','Acts_p90','Acts_spikeshare']:
    qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)
qg=qg.sort_values('period_q')
for base in ['Threat_mean_z','Acts_mean_z']:
    qg[base+'_lead1']=qg[base].shift(-1); qg[base+'_lag1']=qg[base].shift(1); qg[base+'_lag2']=qg[base].shift(2)
qg.to_csv(ROOT/'gpr_quarterly_processed.csv',index=False)

panel=wide[(wide.year>=2020)&(wide.year<=2025)].merge(stat,on='symbol',how='inner').merge(qg,on='period_q',how='left')
panel['Post2022']=(panel.year>=2022).astype(int); panel['FakePost2021']=(panel.year>=2021).astype(int)
lo,hi=panel.cfo_assets.quantile([.01,.99]); panel['cfo_assets_w']=panel.cfo_assets.clip(lo,hi)
panel.to_csv(ROOT/'processed_panel_2020_2025.csv',index=False)

# Combine all existing model-result files if present.
parts=[]
for pa,path in [('PA1','data/gpr_pa1_final_gate_v1/pa1_final_gate_results.csv'),('PA2','data/gpr_pa2_final_gate_v1/pa2_final_gate_results.csv')]:
    p=Path(path)
    if p.exists():
        d=pd.read_csv(p); d.insert(0,'PA',pa); parts.append(d)
# patch corrected PA1 exclusion result if available
p=Path('data/gpr_pa1_exclude_2022h1_v1/pa1_exclude_2022h1.csv')
if p.exists():
    d=pd.read_csv(p); d.insert(0,'PA','PA1'); d['test']='exclusion_corrected'; parts.append(d)
if parts: pd.concat(parts,ignore_index=True,sort=False).to_csv(ROOT/'all_test_results.csv',index=False)
for pa,path in [('PA1','data/gpr_pa1_final_gate_v1/dynamic_did.csv'),('PA2','data/gpr_pa2_final_gate_v1/dynamic_did.csv')]:
    p=Path(path)
    if p.exists():
        d=pd.read_csv(p); d.insert(0,'PA',pa); d.to_csv(ROOT/f'{pa.lower()}_dynamic_did.csv',index=False)

meta={'financial_raw_rows':len(finraw),'processed_panel_rows':len(panel),'processed_panel_firms':int(panel.symbol.nunique()),'gpr_daily_rows':len(g),'quarters':int(qg.period_q.nunique()),'analysis_period':'2020Q1-2025Q4','baseline_period':'2017-2019','source_gpr':'https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv','source_financial':'VCI via authenticated VNStock Bronze workflow'}
(ROOT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2))