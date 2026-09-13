from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS

NEW=Path('data/gpr_bronze_refresh/fundamental_shards')
OUT=Path('data/gpr_wild_cluster_bootstrap_v2'); OUT.mkdir(parents=True,exist_ok=True)
raw=pd.concat([pd.read_parquet(f) for f in sorted(NEW.glob('fundamental_shard_*.parquet'))],ignore_index=True)
raw['symbol']=raw.symbol.astype(str).str.upper().str.strip(); raw['period_q']=raw.period_q.astype(str); raw['value']=pd.to_numeric(raw.value,errors='coerce')
raw=raw.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
ids=set(raw.item_id.dropna().astype(str))
def resolve(pref,tokens):
    for x in pref:
        if x in ids:return x
    c=sorted([x for x in ids if all(t.upper() in x.upper() for t in tokens)])
    return c[0] if len(c)==1 else None
mp={'assets':resolve(['BS_TOTAL_ASSETS'],['TOTAL','ASSET']), 'liabilities':resolve(['BS_TOTAL_LIABILITIES'],['TOTAL','LIABIL']), 'current_liab':resolve(['BS_SHORT_TERM_LIABILITIES'],['SHORT','LIABIL']), 'cfo':resolve(['CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'],['OPERATING','CASH'])}
if any(v is None for v in mp.values()): raise RuntimeError(mp)
rev={v:k for k,v in mp.items()}; z=raw[raw.item_id.isin(rev)].copy(); z['metric']=z.item_id.map(rev)
w=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index()
w[['year','q']]=w.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); w=w.sort_values(['symbol','year','q']); w['pidx']=w.year*4+w.q
w['avg_assets']=(w.assets+w.groupby('symbol').assets.shift(1))/2; w['cfo_assets']=w.cfo/w.avg_assets; w['leverage']=w.liabilities/w.assets; w['log_assets']=np.log(w.assets.where(w.assets>0))
pre=w[(w.year>=2017)&(w.year<=2019)].copy(); pre['lr']=pre.current_liab/pre.assets
st=pre.groupby('symbol').agg(n=('lr','count'),lr=('lr','median')).reset_index(); st=st[st.n>=2].copy(); cut=st.lr.quantile(2/3); st['HL67']=(st.lr>=cut).astype(int)

g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv')
def nt(s):return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(pats):
    for p in pats:
        for c in g.columns:
            if nt(c)==nt(p) or nt(p) in nt(c):return c
    return None
dc=pick(['date','day']); th=pick(['THREATS_GPR_AI','GPR_THREATS']); ac=pick(['ACTS_GPR_AI','GPR_ACTS'])
if not all([dc,th,ac]): raise RuntimeError(list(g.columns))
g=g.rename(columns={dc:'date',th:'Threat',ac:'Acts'}); g.date=pd.to_datetime(g.date,errors='coerce'); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].copy(); g['period_q']=g.date.dt.to_period('Q').astype(str)
qg=g.groupby('period_q').agg(Threat=('Threat','mean'),Acts=('Acts','mean')).reset_index()
for c in ['Threat','Acts']:qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)
panel=w[(w.year>=2020)&(w.year<=2025)].merge(st[['symbol','HL67']],on='symbol',how='inner').merge(qg,on='period_q',how='left')

def prepare(shock):
    x=panel[['symbol','pidx','period_q','cfo_assets','log_assets','leverage','HL67',shock]].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['inter']=x[shock]*x.HL67
    return x.set_index(['symbol','pidx']).sort_index()

def fit_full(x,y=None):
    yy=x.cfo_assets if y is None else y
    return PanelOLS(yy,x[['inter','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_time=True)

def fit_restricted(x):
    return PanelOLS(x.cfo_assets,x[['log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_time=True)

def vec(obj):
    a=np.asarray(obj)
    return a.reshape(-1)

def wild(shock,B=999,seed=20260914):
    x=prepare(shock); full=fit_full(x); beta=float(full.params['inter']); se=float(full.std_errors['inter']); t_obs=beta/se
    r0=fit_restricted(x); fitted=vec(r0.fitted_values); resid=vec(r0.resids)
    periods=x['period_q'].astype(str).to_numpy(); uq=np.array(sorted(pd.unique(periods))); rng=np.random.default_rng(seed); ts=[]; fail=0
    for _ in range(B):
        signs=dict(zip(uq,rng.choice([-1.0,1.0],size=len(uq)))); ww=np.array([signs[p] for p in periods])
        ys=pd.Series(fitted+resid*ww,index=x.index,name='ystar')
        try:
            rb=fit_full(x,ys); sb=float(rb.std_errors['inter'])
            if np.isfinite(sb) and sb>0:ts.append(float(rb.params['inter']/sb))
            else:fail+=1
        except Exception:fail+=1
    ts=np.asarray(ts); p=(1+int(np.sum(np.abs(ts)>=abs(t_obs))))/(1+len(ts))
    return {'shock':shock,'n':len(x),'firms':x.index.get_level_values(0).nunique(),'quarters':len(uq),'beta':beta,'cluster_time_se':se,'t_obs':t_obs,'B_requested':B,'B_success':len(ts),'failures':fail,'wild_cluster_p_two_sided':p}

res=[wild('Threat_z'),wild('Acts_z')]
out={'method':'Restricted wild cluster bootstrap-t; Rademacher quarter-level weights; H0 sets the interaction coefficient to zero. Firm FE, quarter FE, size and leverage retained.','results':res,'decision_rule':'Robust at 5% if wild-cluster p < 0.05; supportive at 10% if p < 0.10. This is an inference robustness test, not causal identification.'}
(OUT/'assessment.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); pd.DataFrame(res).to_csv(OUT/'wild_cluster_results.csv',index=False); print(json.dumps(out,indent=2))
