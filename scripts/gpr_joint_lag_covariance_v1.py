from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import chi2
NEW=Path('data/gpr_bronze_refresh/fundamental_shards'); OUT=Path('data/gpr_joint_lag_covariance_v1'); OUT.mkdir(parents=True,exist_ok=True)
new=pd.concat([pd.read_parquet(f) for f in sorted(NEW.glob('fundamental_shard_*.parquet'))],ignore_index=True)
new['symbol']=new.symbol.astype(str).str.upper().str.strip(); new['period_q']=new.period_q.astype(str); new['value']=pd.to_numeric(new.value,errors='coerce'); new=new.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
ids={'assets':'BS_TOTAL_ASSETS','liabilities':'BS_TOTAL_LIABILITIES','current_liab':'BS_SHORT_TERM_LIABILITIES','cfo':'CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'}
z=new[new.item_id.isin(ids.values())].copy(); z['metric']=z.item_id.map({v:k for k,v in ids.items()}); wide=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index(); wide[['year','q']]=wide.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); wide=wide.sort_values(['symbol','year','q']); wide['pidx']=wide.year*4+wide.q; wide['avg_assets']=(wide.assets+wide.groupby('symbol').assets.shift(1))/2; wide['cfo_assets']=wide.cfo/wide.avg_assets; wide['leverage']=wide.liabilities/wide.assets; wide['log_assets']=np.log(wide.assets.where(wide.assets>0))
pre=wide[(wide.year>=2017)&(wide.year<=2019)].copy(); pre['liab_ratio']=pre.current_liab/pre.assets; stat=pre.groupby('symbol').agg(n=('liab_ratio','count'),liab_ratio=('liab_ratio','median')).reset_index(); stat=stat[stat.n>=2].copy(); cut=stat.liab_ratio.quantile(2/3); stat['HL67']=(stat.liab_ratio>=cut).astype(int)
g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv'); g['date']=pd.to_datetime(g['Date']); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].copy(); g['period_q']=g.date.dt.to_period('Q').astype(str); qg=g.groupby('period_q').agg(Threat=('THREATS_GPR_AI','mean'),Acts=('ACTS_GPR_AI','mean')).reset_index().sort_values('period_q')
for c in ['Threat','Acts']:
 qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0); qg[c+'_lag1']=qg[c+'_z'].shift(1); qg[c+'_lag2']=qg[c+'_z'].shift(2)
p=wide[(wide.year>=2020)&(wide.year<=2025)].merge(stat,on='symbol').merge(qg,on='period_q'); cols=['symbol','pidx','cfo_assets','log_assets','leverage','HL67','Threat_z','Threat_lag1','Threat_lag2','Acts_z','Acts_lag1','Acts_lag2']; x=p[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
base=['Threat_z','Threat_lag1','Threat_lag2','Acts_z','Acts_lag1','Acts_lag2']
for n in base: x[n+'_x_HL']=x[n]*x.HL67
ints=[n+'_x_HL' for n in base]; x=x.set_index(['symbol','pidx']).sort_index(); mod=PanelOLS(x.cfo_assets,x[ints+['log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True)
rows=[]
for covtype in ['twoway','entity','dk','robust']:
 if covtype=='twoway': r=mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
 elif covtype=='entity': r=mod.fit(cov_type='clustered',cluster_entity=True)
 elif covtype=='dk': r=mod.fit(cov_type='kernel',kernel='bartlett',bandwidth=3)
 else: r=mod.fit(cov_type='robust')
 names=list(r.params.index); w=np.zeros(len(names));
 for nm,v in {'Threat_lag1_x_HL':1,'Threat_lag2_x_HL':1,'Acts_lag1_x_HL':-1,'Acts_lag2_x_HL':-1}.items(): w[names.index(nm)]=v
 est=float(w@r.params.values); var=float(w@r.cov.values@w); se=float(np.sqrt(var)) if var>=0 else np.nan; z=est/se if se>0 else np.nan; pv=float(chi2.sf(z*z,1)) if np.isfinite(z) else np.nan
 rows.append({'covariance':covtype,'lag_sum_difference_Threat_minus_Acts':est,'se':se,'z':z,'p':pv,'significant_5pct':bool(pv<.05) if np.isfinite(pv) else False})
out={'sample_n':len(x),'sample_firms':int(x.index.get_level_values(0).nunique()),'results':rows,'interpretation':'Positive difference means the net lagged association (lags 1+2) is more positive/less negative for Threat than Acts. The test is about lag structure, not causal effect size.'}; (OUT/'assessment.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
