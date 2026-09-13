from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS

NEW=Path('data/gpr_bronze_refresh/fundamental_shards')
OUT=Path('data/gpr_pa2_final_gate_v1'); OUT.mkdir(parents=True,exist_ok=True)
new=pd.concat([pd.read_parquet(f) for f in sorted(NEW.glob('fundamental_shard_*.parquet'))],ignore_index=True)
new['symbol']=new.symbol.astype(str).str.upper().str.strip(); new['period_q']=new.period_q.astype(str); new['value']=pd.to_numeric(new.value,errors='coerce')
new=new.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
allids=set(new.item_id.dropna().astype(str))
def resolve(preferred,tokens):
    for x in preferred:
        if x in allids:return x
    cand=sorted([x for x in allids if all(t.upper() in x.upper() for t in tokens)])
    return cand[0] if len(cand)==1 else None
mapping={'assets':resolve(['BS_TOTAL_ASSETS'],['TOTAL','ASSET']),'liabilities':resolve(['BS_TOTAL_LIABILITIES'],['TOTAL','LIABIL']),'current_liab':resolve(['BS_SHORT_TERM_LIABILITIES'],['SHORT','LIABIL']),'cfo':resolve(['CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'],['OPERATING','CASH'])}
if any(mapping[x] is None for x in mapping):raise RuntimeError(mapping)
use={v:k for k,v in mapping.items()}; z=new[new.item_id.isin(use)].copy(); z['metric']=z.item_id.map(use)
wide=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index()
wide[['year','q']]=wide.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); wide=wide.sort_values(['symbol','year','q']); wide['pidx']=wide.year*4+wide.q
wide['avg_assets']=(wide.assets+wide.groupby('symbol').assets.shift(1))/2; wide['cfo_assets']=wide.cfo/wide.avg_assets; wide['leverage']=wide.liabilities/wide.assets; wide['log_assets']=np.log(wide.assets.where(wide.assets>0))
pre=wide[(wide.year>=2017)&(wide.year<=2019)].copy(); pre['liab_ratio']=pre.current_liab/pre.assets
stat=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),liab_ratio=('liab_ratio','median')).reset_index(); stat=stat[stat.pre_quarters>=2].copy()
for pct,name in [(0.60,'HL60'),(2/3,'HL67'),(0.75,'HL75')]:
    cut=stat.liab_ratio.quantile(pct); stat[name]=(stat.liab_ratio>=cut).astype(int); stat[name+'_cut']=cut

g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv')
def nt(s):return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(pats):
    for p in pats:
        for c in g.columns:
            if nt(c)==nt(p) or nt(p) in nt(c):return c
    return None
dc=pick(['date','day']); ac=pick(['ACTS_GPR_AI','GPR_ACTS']); th=pick(['THREATS_GPR_AI','GPR_THREATS'])
if not all([dc,ac,th]):raise RuntimeError(list(g.columns))
g=g.rename(columns={dc:'date',ac:'Acts',th:'Threat'}); g.date=pd.to_datetime(g.date,errors='coerce'); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].copy(); g['period_q']=g.date.dt.to_period('Q').astype(str)
qg=g.groupby('period_q').agg(Acts_mean=('Acts','mean'),Acts_max=('Acts','max'),Acts_p90=('Acts',lambda s:s.quantile(.90)),Threat_mean=('Threat','mean')).reset_index()
a90=float(g.Acts.quantile(.90)); spike=g.assign(spike=(g.Acts>=a90).astype(float)).groupby('period_q').spike.mean().reset_index(name='Acts_spikeshare'); qg=qg.merge(spike,on='period_q')
for c in ['Acts_mean','Acts_max','Acts_p90','Acts_spikeshare','Threat_mean']:
    qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)
qg=qg.sort_values('period_q'); qg['Acts_mean_z_lead1']=qg.Acts_mean_z.shift(-1); qg['Acts_mean_z_lag1']=qg.Acts_mean_z.shift(1); qg['Acts_mean_z_lag2']=qg.Acts_mean_z.shift(2)
panel=wide[(wide.year>=2020)&(wide.year<=2025)].merge(stat,on='symbol',how='inner').merge(qg,on='period_q',how='left'); panel['Post2022']=(panel.year>=2022).astype(int); panel['FakePost2021']=(panel.year>=2021).astype(int)
lo,hi=panel.cfo_assets.quantile([.01,.99]); panel['cfo_assets_w']=panel.cfo_assets.clip(lo,hi)

def fit(data,shock='Acts_mean_z',grp='HL67',outcome='cfo_assets',cov='twoway',extra_controls=True):
    cols=['symbol','pidx',outcome,shock,grp]+(['log_assets','leverage'] if extra_controls else [])
    x=data[cols].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['inter']=x[shock]*x[grp]; x=x.set_index(['symbol','pidx']).sort_index(); ex=['inter']+(['log_assets','leverage'] if extra_controls else [])
    mod=PanelOLS(x[outcome],x[ex],entity_effects=True,time_effects=True,drop_absorbed=True)
    if cov=='twoway':r=mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    elif cov=='entity':r=mod.fit(cov_type='clustered',cluster_entity=True)
    elif cov=='dk':r=mod.fit(cov_type='kernel',kernel='bartlett',bandwidth=3)
    elif cov=='robust':r=mod.fit(cov_type='robust')
    else:raise ValueError(cov)
    return {'n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params['inter']),'se':float(r.std_errors['inter']),'p':float(r.pvalues['inter']),'r2_within':float(r.rsquared_within)}
rows=[]
def add(test,label,data=None,**kw):
    try:
        rr=fit(panel if data is None else data,**kw); rows.append({'test':test,'label':label,**rr})
    except Exception as e:rows.append({'test':test,'label':label,'error':repr(e)})
add('primary','Acts_mean_z x HL67')
for sh in ['Acts_max_z','Acts_p90_z','Acts_spikeshare_z']:add('alt_acts_measure',sh,shock=sh)
for sh in ['Acts_mean_z_lead1','Acts_mean_z_lag1','Acts_mean_z_lag2']:add('timing',sh,shock=sh)
for grp in ['HL60','HL67','HL75']:add('cutoff',grp,grp=grp)
for cov in ['twoway','entity','dk','robust']:add('se_robustness',cov,cov=cov)
add('winsor','cfo_assets_w',outcome='cfo_assets_w'); add('controls','without_controls',extra_controls=False)
add('exclusion','exclude_2022H1',data=panel[~panel.period_q.isin(['2022Q1','2022Q2'])])
for y in range(2020,2026):add('leave_one_year_out',str(y),data=panel[panel.year!=y])
for pq in sorted(panel.period_q.dropna().unique()):add('leave_one_quarter_out',pq,data=panel[panel.period_q!=pq])
res=pd.DataFrame(rows); res.to_csv(OUT/'pa2_final_gate_results.csv',index=False)

def fit_did(post_col):
    x=panel[['symbol','pidx','cfo_assets','log_assets','leverage',post_col,'HL67']].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['did']=x[post_col]*x.HL67; x=x.set_index(['symbol','pidx']).sort_index(); r=PanelOLS(x.cfo_assets,x[['did','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True); return {'beta':float(r.params.did),'se':float(r.std_errors.did),'p':float(r.pvalues.did),'n':len(x),'firms':x.index.get_level_values(0).nunique()}
did={'Post2022':fit_did('Post2022'),'FakePost2021':fit_did('FakePost2021')}
# event-time diagnostic around 2021Q4, for comparability with PA1
panel['k']=panel.pidx-(2021*4+4); ev=panel[['symbol','pidx','cfo_assets','log_assets','leverage','HL67','k']].replace([np.inf,-np.inf],np.nan).dropna().copy(); ks=[int(k) for k in sorted(ev.k.unique()) if k!=-1]
for k in ks:ev[f'k_{k}']=((ev.k==k)&(ev.HL67==1)).astype(int)
ex=[f'k_{k}' for k in ks]+['log_assets','leverage']; ev=ev.set_index(['symbol','pidx']).sort_index(); dyn=[]
try:
    r=PanelOLS(ev.cfo_assets,ev[ex],entity_effects=True,time_effects=True,drop_absorbed=True,check_rank=False).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    for k in ks:
        nm=f'k_{k}'
        if nm in r.params.index:dyn.append({'k':k,'beta':float(r.params[nm]),'se':float(r.std_errors.get(nm,np.nan)),'p':float(r.pvalues.get(nm,np.nan))})
except Exception as e:dyn=[{'error':repr(e)}]
pd.DataFrame(dyn).to_csv(OUT/'dynamic_did.csv',index=False)
valid=res[res.error.isna()] if 'error' in res.columns else res; primary=valid[valid.test=='primary'].iloc[0]; alt=valid[valid.test=='alt_acts_measure']; loo=valid[valid.test=='leave_one_year_out']; loq=valid[valid.test=='leave_one_quarter_out']; sev=valid[valid.test=='se_robustness']; timing=valid[valid.test=='timing']; lead=timing[timing.label=='Acts_mean_z_lead1'].iloc[0]; excl=valid[valid.test=='exclusion'].iloc[0]
summary={'sample_firms':int(primary.firms),'sample_n':int(primary.n),'primary_beta':float(primary.beta),'primary_p':float(primary.p),'primary_negative_p_lt_05':bool(primary.beta<0 and primary.p<.05),'alternative_acts_negative_share':float((alt.beta<0).mean()),'alternative_acts_p_lt_10_share':float((alt.p<.10).mean()),'lead_placebo_clean_p_gt_10':bool(lead.p>.10),'lead_beta':float(lead.beta),'lead_p':float(lead.p),'leave_one_year_negative_share':float((loo.beta<0).mean()),'leave_one_year_p_lt_10_share':float((loo.p<.10).mean()),'leave_one_quarter_negative_share':float((loq.beta<0).mean()),'leave_one_quarter_p_lt_10_share':float((loq.p<.10).mean()),'se_all_negative':bool((sev.beta<0).all()),'se_p_lt_10_share':float((sev.p<.10).mean()),'exclude_2022H1_beta':float(excl.beta),'exclude_2022H1_p':float(excl.p),'did':did,'caution':'Post2022 DID is a broad macro-regime diagnostic, not a uniquely identified geopolitical Acts shock.'}
summary['decision']='PASS_TO_THESIS_CORE' if (summary['primary_negative_p_lt_05'] and summary['lead_placebo_clean_p_gt_10'] and summary['alternative_acts_negative_share']>=2/3 and summary['leave_one_year_negative_share']==1.0 and summary['se_all_negative']) else 'HOLD_OR_EXTENSION'
(OUT/'assessment.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
