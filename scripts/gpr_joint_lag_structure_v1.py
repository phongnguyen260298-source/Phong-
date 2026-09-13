from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import chi2

NEW=Path('data/gpr_bronze_refresh/fundamental_shards')
OUT=Path('data/gpr_joint_lag_structure_v1'); OUT.mkdir(parents=True,exist_ok=True)
new=pd.concat([pd.read_parquet(f) for f in sorted(NEW.glob('fundamental_shard_*.parquet'))],ignore_index=True)
new['symbol']=new.symbol.astype(str).str.upper().str.strip(); new['period_q']=new.period_q.astype(str); new['value']=pd.to_numeric(new.value,errors='coerce')
new=new.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
allids=set(new.item_id.dropna().astype(str))
def resolve(preferred,tokens):
    for x in preferred:
        if x in allids: return x
    cand=sorted([x for x in allids if all(t.upper() in x.upper() for t in tokens)])
    return cand[0] if len(cand)==1 else None
mapping={
 'assets':resolve(['BS_TOTAL_ASSETS'],['TOTAL','ASSET']),
 'liabilities':resolve(['BS_TOTAL_LIABILITIES'],['TOTAL','LIABIL']),
 'current_liab':resolve(['BS_SHORT_TERM_LIABILITIES'],['SHORT','LIABIL']),
 'cfo':resolve(['CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'],['OPERATING','CASH'])}
use={v:k for k,v in mapping.items()}
z=new[new.item_id.isin(use)].copy(); z['metric']=z.item_id.map(use)
wide=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index()
wide[['year','q']]=wide.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); wide=wide.sort_values(['symbol','year','q']); wide['pidx']=wide.year*4+wide.q
wide['avg_assets']=(wide.assets+wide.groupby('symbol').assets.shift(1))/2
wide['cfo_assets']=wide.cfo/wide.avg_assets; wide['leverage']=wide.liabilities/wide.assets; wide['log_assets']=np.log(wide.assets.where(wide.assets>0))
pre=wide[(wide.year>=2017)&(wide.year<=2019)].copy(); pre['liab_ratio']=pre.current_liab/pre.assets
stat=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),liab_ratio=('liab_ratio','median')).reset_index(); stat=stat[stat.pre_quarters>=2].copy()
cut=stat.liab_ratio.quantile(2/3); stat['HL67']=(stat.liab_ratio>=cut).astype(int)

g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv')
def nt(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(pats):
    for p in pats:
        for c in g.columns:
            if nt(c)==nt(p) or nt(p) in nt(c): return c
    return None
dc=pick(['date','day']); th=pick(['THREATS_GPR_AI','GPR_THREATS']); ac=pick(['ACTS_GPR_AI','GPR_ACTS'])
g=g.rename(columns={dc:'date',th:'Threat',ac:'Acts'}); g.date=pd.to_datetime(g.date,errors='coerce'); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].copy(); g['period_q']=g.date.dt.to_period('Q').astype(str)
qg=g.groupby('period_q').agg(Threat=('Threat','mean'),Acts=('Acts','mean')).reset_index().sort_values('period_q')
for c in ['Threat','Acts']:
    qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)
    qg[c+'_lag1']=qg[c+'_z'].shift(1)
    qg[c+'_lag2']=qg[c+'_z'].shift(2)

panel=wide[(wide.year>=2020)&(wide.year<=2025)].merge(stat,on='symbol',how='inner').merge(qg,on='period_q',how='left')
cols=['symbol','pidx','cfo_assets','log_assets','leverage','HL67','Threat_z','Threat_lag1','Threat_lag2','Acts_z','Acts_lag1','Acts_lag2']
x=panel[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
for nm in ['Threat_z','Threat_lag1','Threat_lag2','Acts_z','Acts_lag1','Acts_lag2']:
    x[nm+'_x_HL']=x[nm]*x['HL67']
inter=[c+'_x_HL' for c in ['Threat_z','Threat_lag1','Threat_lag2','Acts_z','Acts_lag1','Acts_lag2']]
x=x.set_index(['symbol','pidx']).sort_index()
exog=inter+['log_assets','leverage']
mod=PanelOLS(x.cfo_assets,x[exog],entity_effects=True,time_effects=True,drop_absorbed=True)
r=mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)

rows=[]
for nm in inter:
    rows.append({'term':nm,'beta':float(r.params[nm]),'se':float(r.std_errors[nm]),'p':float(r.pvalues[nm])})
pd.DataFrame(rows).to_csv(OUT/'joint_lag_coefficients.csv',index=False)

# Wald helper using covariance matrix
params=r.params
cov=r.cov
names=list(params.index)
def wald_linear(weights, value=0.0):
    w=np.zeros(len(names))
    for k,v in weights.items(): w[names.index(k)]=v
    est=float(w@params.values - value)
    var=float(w@cov.values@w)
    se=np.sqrt(var) if var>=0 else np.nan
    z=est/se if se and np.isfinite(se) and se>0 else np.nan
    p=float(chi2.sf(z*z,1)) if np.isfinite(z) else np.nan
    return {'estimate':est,'se':float(se),'z':float(z),'p':p}

T0='Threat_z_x_HL'; T1='Threat_lag1_x_HL'; T2='Threat_lag2_x_HL'
A0='Acts_z_x_HL'; A1='Acts_lag1_x_HL'; A2='Acts_lag2_x_HL'
# cumulative 0-2 quarter association for each component
cumT=wald_linear({T0:1,T1:1,T2:1})
cumA=wald_linear({A0:1,A1:1,A2:1})
cumDiff=wald_linear({T0:1,T1:1,T2:1,A0:-1,A1:-1,A2:-1})
# persistence only lags 1+2
persT=wald_linear({T1:1,T2:1})
persA=wald_linear({A1:1,A2:1})
persDiff=wald_linear({T1:1,T2:1,A1:-1,A2:-1})
# joint significance of lag coefficients for each component: 2-df Wald
from numpy.linalg import pinv

def joint_two(a,b):
    R=np.zeros((2,len(names))); R[0,names.index(a)]=1; R[1,names.index(b)]=1
    q=R@params.values
    V=R@cov.values@R.T
    stat=float(q.T@pinv(V)@q)
    p=float(chi2.sf(stat,2))
    return {'chi2':stat,'df':2,'p':p}

jointLagT=joint_two(T1,T2)
jointLagA=joint_two(A1,A2)

summary={
 'sample_n':int(len(x)),
 'sample_firms':int(x.index.get_level_values(0).nunique()),
 'highliab_cut':float(cut),
 'coefficients':rows,
 'cumulative_0_2q':{'Threat':cumT,'Acts':cumA,'Threat_minus_Acts':cumDiff},
 'persistence_lags_1_2':{'Threat':persT,'Acts':persA,'Threat_minus_Acts':persDiff},
 'joint_significance_lags_1_2':{'Threat':jointLagT,'Acts':jointLagA},
 'interpretation':'All six interaction terms are estimated simultaneously. Cumulative tests compare summed contemporaneous and lagged conditional associations. Persistence tests use only lag 1 and lag 2 terms. Difference tests assess whether the cumulative/persistent association differs between Threat and Acts.',
 'caution':'With only 24 quarters, inference based on time clustering can be fragile. Treat these tests as mechanism evidence, not clean causal identification.'
}
(OUT/'assessment.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
