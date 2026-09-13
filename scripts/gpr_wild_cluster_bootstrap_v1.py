from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS

NEW=Path('data/gpr_bronze_refresh/fundamental_shards')
OUT=Path('data/gpr_wild_cluster_bootstrap_v1'); OUT.mkdir(parents=True,exist_ok=True)
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
if any(mapping[x] is None for x in mapping): raise RuntimeError(mapping)
use={v:k for k,v in mapping.items()}
z=new[new.item_id.isin(use)].copy(); z['metric']=z.item_id.map(use)
wide=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index()
wide[['year','q']]=wide.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); wide=wide.sort_values(['symbol','year','q']); wide['pidx']=wide.year*4+wide.q
wide['avg_assets']=(wide.assets+wide.groupby('symbol').assets.shift(1))/2
wide['cfo_assets']=wide.cfo/wide.avg_assets; wide['leverage']=wide.liabilities/wide.assets; wide['log_assets']=np.log(wide.assets.where(wide.assets>0))
pre=wide[(wide.year>=2017)&(wide.year<=2019)].copy(); pre['liab_ratio']=pre.current_liab/pre.assets
stat=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),liab_ratio=('liab_ratio','median')).reset_index(); stat=stat[stat.pre_quarters>=2].copy(); cut=stat.liab_ratio.quantile(2/3); stat['HL67']=(stat.liab_ratio>=cut).astype(int)

g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv')
def nt(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(pats):
    for p in pats:
        for c in g.columns:
            if nt(c)==nt(p) or nt(p) in nt(c): return c
    return None
dc=pick(['date','day']); th=pick(['THREATS_GPR_AI','GPR_THREATS']); ac=pick(['ACTS_GPR_AI','GPR_ACTS'])
if not all([dc,th,ac]): raise RuntimeError(list(g.columns))
g=g.rename(columns={dc:'date',th:'Threat',ac:'Acts'}); g.date=pd.to_datetime(g.date,errors='coerce'); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].copy(); g['period_q']=g.date.dt.to_period('Q').astype(str)
qg=g.groupby('period_q').agg(Threat=('Threat','mean'),Acts=('Acts','mean')).reset_index()
for c in ['Threat','Acts']:
    qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)
panel=wide[(wide.year>=2020)&(wide.year<=2025)].merge(stat[['symbol','HL67']],on='symbol',how='inner').merge(qg,on='period_q',how='left')

# Time-cluster wild bootstrap-t using Rademacher weights on quarter clusters.
# Restricted bootstrap under H0: beta_interaction=0. Other regressors retained.
def prepare(shock):
    cols=['symbol','pidx','period_q','cfo_assets','log_assets','leverage','HL67',shock]
    x=panel[cols].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['inter']=x[shock]*x.HL67
    x=x.set_index(['symbol','pidx']).sort_index()
    return x

def fit_full(x,y=None):
    yy=x.cfo_assets if y is None else y
    mod=PanelOLS(yy,x[['inter','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True)
    return mod.fit(cov_type='clustered',cluster_time=True)

def fit_restricted(x):
    mod=PanelOLS(x.cfo_assets,x[['log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True)
    return mod.fit(cov_type='clustered',cluster_time=True)

def wild_boot(shock,B=9999,seed=20260914):
    x=prepare(shock); full=fit_full(x); beta=float(full.params['inter']); se=float(full.std_errors['inter']); t_obs=beta/se
    rest=fit_restricted(x); fitted=rest.predict().fitted_values.iloc[:,0]; resid=rest.resids
    # align time cluster labels
    tmp=x.reset_index()[['symbol','pidx','period_q']]
    idx_df=x.reset_index()[['symbol','pidx']]
    # x index is sorted identically to reset index order
    periods=tmp['period_q'].to_numpy(); uniq=np.array(sorted(pd.unique(periods)))
    rng=np.random.default_rng(seed); tstars=[]; failures=0
    for b in range(B):
        wdict={u:rng.choice([-1.0,1.0]) for u in uniq}
        w=np.array([wdict[p] for p in periods])
        ystar=pd.Series(fitted.to_numpy()+resid.to_numpy()*w,index=x.index,name='ystar')
        try:
            rb=fit_full(x,ystar); seb=float(rb.std_errors['inter'])
            if np.isfinite(seb) and seb>0:
                tstars.append(float(rb.params['inter']/seb))
            else: failures+=1
        except Exception:
            failures+=1
    tstars=np.asarray(tstars)
    p=(1+np.sum(np.abs(tstars)>=abs(t_obs)))/(1+len(tstars))
    return {'shock':shock,'n':int(len(x)),'firms':int(x.index.get_level_values(0).nunique()),'quarters':int(len(uniq)),'beta':beta,'cluster_time_se':se,'t_obs':float(t_obs),'B_requested':B,'B_success':int(len(tstars)),'failures':int(failures),'wild_cluster_p_two_sided':float(p)}

res=[wild_boot('Threat_z'),wild_boot('Acts_z')]
summary={'method':'Restricted wild cluster bootstrap-t with Rademacher weights at quarter level; null imposes interaction coefficient = 0; full/restricted models include firm and quarter fixed effects plus size and leverage.','results':res,'interpretation_rule':'Use this as small-time-cluster inference robustness only. It does not convert the fixed-effects association into causal identification.'}
(OUT/'assessment.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
pd.DataFrame(res).to_csv(OUT/'wild_cluster_results.csv',index=False)
print(json.dumps(summary,indent=2))
