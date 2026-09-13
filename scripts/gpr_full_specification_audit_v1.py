from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

NEW=Path('data/gpr_bronze_refresh/fundamental_shards')
OUT=Path('data/gpr_full_specification_audit_v1'); OUT.mkdir(parents=True,exist_ok=True)
new=pd.concat([pd.read_parquet(f) for f in sorted(NEW.glob('fundamental_shard_*.parquet'))],ignore_index=True)
new['symbol']=new.symbol.astype(str).str.upper().str.strip(); new['period_q']=new.period_q.astype(str); new['value']=pd.to_numeric(new.value,errors='coerce')
new=new.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
allids=set(new.item_id.dropna().astype(str))
def resolve(preferred,tokens):
    for x in preferred:
        if x in allids:return x
    cand=sorted([x for x in allids if all(t.upper() in x.upper() for t in tokens)])
    return cand[0] if len(cand)==1 else None
mapping={
 'assets':resolve(['BS_TOTAL_ASSETS'],['TOTAL','ASSET']),
 'liabilities':resolve(['BS_TOTAL_LIABILITIES'],['TOTAL','LIABIL']),
 'current_liab':resolve(['BS_SHORT_TERM_LIABILITIES'],['SHORT','LIABIL']),
 'cfo':resolve(['CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'],['OPERATING','CASH'])}
use={v:k for k,v in mapping.items()}; z=new[new.item_id.isin(use)].copy(); z['metric']=z.item_id.map(use)
wide=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index()
wide[['year','q']]=wide.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); wide=wide.sort_values(['symbol','year','q']); wide['pidx']=wide.year*4+wide.q
wide['avg_assets']=(wide.assets+wide.groupby('symbol').assets.shift(1))/2
wide['cfo_assets']=wide.cfo/wide.avg_assets; wide['leverage']=wide.liabilities/wide.assets; wide['log_assets']=np.log(wide.assets.where(wide.assets>0))
wide['liab_ratio']=wide.current_liab/wide.assets

# AI-GPR
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
for c in ['Threat','Acts']: qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)
for c in ['Threat_z','Acts_z']:
    qg[c+'_lag1']=qg[c].shift(1); qg[c+'_lag2']=qg[c].shift(2)

# baseline definitions
pre=wide[(wide.year>=2017)&(wide.year<=2019)].copy()
def baseline_stat(mask,agg='median',snapshot=False):
    d=pre[mask(pre)].copy()
    if snapshot:
        s=d[['symbol','liab_ratio']].dropna().drop_duplicates('symbol',keep='last').rename(columns={'liab_ratio':'base'})
        s['n_pre']=1
    else:
        gg=d.groupby('symbol').liab_ratio
        s=(gg.mean() if agg=='mean' else gg.median()).rename('base').reset_index(); n=gg.count().rename('n_pre').reset_index(); s=s.merge(n,on='symbol')
    return s
specs={
 'median_2017_2019':baseline_stat(lambda d:(d.year>=2017)&(d.year<=2019),'median'),
 'median_2018_2019':baseline_stat(lambda d:(d.year>=2018)&(d.year<=2019),'median'),
 'median_2019':baseline_stat(lambda d:d.year==2019,'median'),
 'mean_2017_2019':baseline_stat(lambda d:(d.year>=2017)&(d.year<=2019),'mean'),
 'snapshot_2019Q4':baseline_stat(lambda d:(d.year==2019)&(d.q==4),'median',snapshot=True),
}

panel0=wide[(wide.year>=2020)&(wide.year<=2025)].merge(qg,on='period_q',how='left')

def fit_binary(base,shock):
    b=base[base.n_pre>= (1 if len(base) and base.n_pre.max()==1 else 2)].copy(); cut=b.base.quantile(2/3); b['HL']=(b.base>=cut).astype(int)
    d=panel0.merge(b[['symbol','HL']],on='symbol',how='inner')
    cols=['symbol','pidx','cfo_assets',shock,'HL','log_assets','leverage']; x=d[cols].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['inter']=x[shock]*x.HL; x=x.set_index(['symbol','pidx']).sort_index()
    r=PanelOLS(x.cfo_assets,x[['inter','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    return {'n':len(x),'firms':x.index.get_level_values(0).nunique(),'cut':float(cut),'beta':float(r.params.inter),'se':float(r.std_errors.inter),'p':float(r.pvalues.inter)}

def fit_continuous(base,shock):
    b=base[base.n_pre>=2].copy(); b['base_z']=(b.base-b.base.mean())/b.base.std(ddof=0)
    d=panel0.merge(b[['symbol','base_z']],on='symbol',how='inner'); cols=['symbol','pidx','cfo_assets',shock,'base_z','log_assets','leverage']; x=d[cols].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['inter']=x[shock]*x.base_z; x=x.set_index(['symbol','pidx']).sort_index()
    r=PanelOLS(x.cfo_assets,x[['inter','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    return {'n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params.inter),'se':float(r.std_errors.inter),'p':float(r.pvalues.inter)}

baseline_rows=[]
for name,b in specs.items():
    for shock in ['Threat_z','Acts_z']:
        try: baseline_rows.append({'definition':name,'shock':shock,**fit_binary(b,shock)})
        except Exception as e: baseline_rows.append({'definition':name,'shock':shock,'error':repr(e)})
# continuous main baseline
for shock in ['Threat_z','Acts_z']:
    baseline_rows.append({'definition':'continuous_median_2017_2019','shock':shock,**fit_continuous(specs['median_2017_2019'],shock)})
pd.DataFrame(baseline_rows).to_csv(OUT/'baseline_definition_robustness.csv',index=False)

# Main baseline group for lag/collinearity audit
b=specs['median_2017_2019']; b=b[b.n_pre>=2].copy(); cut=b.base.quantile(2/3); b['HL']=(b.base>=cut).astype(int)
panel=panel0.merge(b[['symbol','HL']],on='symbol',how='inner')
lagvars=['Threat_z','Threat_z_lag1','Threat_z_lag2','Acts_z','Acts_z_lag1','Acts_z_lag2']
# quarter-level shock correlation and VIF/condition no
qmat=qg[lagvars].dropna().copy(); corr=qmat.corr(); corr.to_csv(OUT/'shock_lag_correlation.csv')
X=(qmat-qmat.mean())/qmat.std(ddof=0)
vifs={c:float(variance_inflation_factor(X.values,i)) for i,c in enumerate(X.columns)}
condition=float(np.linalg.cond(X.values))

# Separate distributed lag models and joint distributed lag model
def fit_lag(which):
    shocks=['Threat_z','Threat_z_lag1','Threat_z_lag2'] if which=='Threat' else ['Acts_z','Acts_z_lag1','Acts_z_lag2'] if which=='Acts' else lagvars
    cols=['symbol','pidx','cfo_assets','HL','log_assets','leverage']+shocks; x=panel[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
    inter=[]
    for s in shocks:
        nm=s+'_x_HL'; x[nm]=x[s]*x.HL; inter.append(nm)
    x=x.set_index(['symbol','pidx']).sort_index(); r=PanelOLS(x.cfo_assets,x[inter+['log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    rows=[]
    for nm in inter: rows.append({'model':which,'term':nm,'beta':float(r.params[nm]),'se':float(r.std_errors[nm]),'p':float(r.pvalues[nm]),'n':len(x)})
    return rows,r
lagrows=[]
for wh in ['Threat','Acts','Joint']:
    try:
        rr,_=fit_lag(wh); lagrows+=rr
    except Exception as e: lagrows.append({'model':wh,'error':repr(e)})
pd.DataFrame(lagrows).to_csv(OUT/'distributed_lag_models.csv',index=False)

# Time-only clustering sensitivity for primary separate models + lag difference
covrows=[]
def fit_primary(shock,cov):
    d=panel[['symbol','pidx','cfo_assets','HL','log_assets','leverage',shock]].replace([np.inf,-np.inf],np.nan).dropna().copy(); d['inter']=d[shock]*d.HL; d=d.set_index(['symbol','pidx']).sort_index(); mod=PanelOLS(d.cfo_assets,d[['inter','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True)
    if cov=='time':r=mod.fit(cov_type='clustered',cluster_time=True)
    elif cov=='twoway':r=mod.fit(cov_type='clustered',cluster_time=True,cluster_entity=True)
    elif cov=='dk':r=mod.fit(cov_type='kernel',kernel='bartlett',bandwidth=3)
    return {'shock':shock,'cov':cov,'beta':float(r.params.inter),'se':float(r.std_errors.inter),'p':float(r.pvalues.inter)}
for shock in ['Threat_z','Acts_z']:
    for cov in ['twoway','time','dk']:
        covrows.append(fit_primary(shock,cov))
pd.DataFrame(covrows).to_csv(OUT/'primary_covariance_sensitivity.csv',index=False)

summary={
 'baseline_robustness':baseline_rows,
 'collinearity':{'pairwise_correlation':corr.to_dict(),'vif':vifs,'condition_number':condition},
 'notes':[
   'Baseline robustness changes only the pre-period definition; the 2020-2025 outcome/shock sample and model are otherwise unchanged.',
   'High VIF or condition number in the six-term joint distributed-lag model would indicate unstable individual lag coefficients and sign reversals should not be over-interpreted.',
   'Separate distributed-lag models are reported to distinguish genuine timing patterns from multicollinearity induced by jointly entering Threat and Acts lags.',
   'No specification is selected based on p-values.'
 ]
}
(OUT/'assessment.json').write_text(json.dumps(summary,indent=2,allow_nan=True),encoding='utf-8')
print(json.dumps({'condition_number':condition,'vif':vifs,'baseline_rows':baseline_rows},indent=2))
