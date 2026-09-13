from __future__ import annotations
import re
from pathlib import Path
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS

NEW=Path('data/gpr_bronze_refresh/fundamental_shards')
OUT=Path('data/gpr_pa1_exclude_2022h1_v1'); OUT.mkdir(parents=True,exist_ok=True)
new=pd.concat([pd.read_parquet(f) for f in sorted(NEW.glob('fundamental_shard_*.parquet'))],ignore_index=True)
new['symbol']=new.symbol.astype(str).str.upper().str.strip(); new['period_q']=new.period_q.astype(str); new['value']=pd.to_numeric(new.value,errors='coerce')
new=new.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
need={'BS_TOTAL_ASSETS':'assets','BS_TOTAL_LIABILITIES':'liabilities','BS_SHORT_TERM_LIABILITIES':'current_liab','CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES':'cfo'}
z=new[new.item_id.isin(need)].copy(); z['metric']=z.item_id.map(need)
wide=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index()
wide[['year','q']]=wide.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); wide=wide.sort_values(['symbol','year','q']); wide['pidx']=wide.year*4+wide.q
wide['avg_assets']=(wide.assets+wide.groupby('symbol').assets.shift(1))/2; wide['cfo_assets']=wide.cfo/wide.avg_assets; wide['leverage']=wide.liabilities/wide.assets; wide['log_assets']=np.log(wide.assets.where(wide.assets>0))
pre=wide[(wide.year>=2017)&(wide.year<=2019)].copy(); pre['liab_ratio']=pre.current_liab/pre.assets
stat=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),liab_ratio=('liab_ratio','median')).reset_index(); stat=stat[stat.pre_quarters>=2].copy(); cut=stat.liab_ratio.quantile(2/3); stat['HL67']=(stat.liab_ratio>=cut).astype(int)

g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv')
def nt(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(pats):
    for p in pats:
        for c in g.columns:
            if nt(c)==nt(p) or nt(p) in nt(c): return c
    return None
dc=pick(['date','day']); th=pick(['THREATS_GPR_AI','GPR_THREATS'])
g=g.rename(columns={dc:'date',th:'Threat'}); g.date=pd.to_datetime(g.date,errors='coerce'); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].copy(); g['period_q']=g.date.dt.to_period('Q').astype(str)
qg=g.groupby('period_q').agg(Threat_mean=('Threat','mean')).reset_index(); qg['Threat_mean_z']=(qg.Threat_mean-qg.Threat_mean.mean())/qg.Threat_mean.std(ddof=0)
panel=wide[(wide.year>=2020)&(wide.year<=2025)].merge(stat,on='symbol',how='inner').merge(qg,on='period_q',how='left')
panel=panel[~panel.period_q.isin(['2022Q1','2022Q2'])].copy()
x=panel[['symbol','pidx','cfo_assets','Threat_mean_z','HL67','log_assets','leverage']].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['inter']=x.Threat_mean_z*x.HL67; x=x.set_index(['symbol','pidx']).sort_index()
r=PanelOLS(x.cfo_assets,x[['inter','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
out=pd.DataFrame([{'test':'exclusion','label':'exclude_2022H1','n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params.inter),'se':float(r.std_errors.inter),'p':float(r.pvalues.inter),'r2_within':float(r.rsquared_within),'hl67_cut':float(cut)}])
out.to_csv(OUT/'pa1_exclude_2022h1.csv',index=False)
print(out.to_string(index=False))