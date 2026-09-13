from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm

NEW=Path('data/gpr_bronze_refresh/fundamental_shards')
OUT=Path('data/gpr_joint_econsig_v1'); OUT.mkdir(parents=True,exist_ok=True)

# Rebuild the same locked panel used by PA1/PA2 gates
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
stat=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),liab_ratio=('liab_ratio','median')).reset_index(); stat=stat[stat.pre_quarters>=2].copy()
cut=stat.liab_ratio.quantile(2/3); stat['HL67']=(stat.liab_ratio>=cut).astype(int); stat['HL67_cut']=cut

# Official AI-GPR daily data -> quarterly standardized Threat / Acts means
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
qg=g.groupby('period_q').agg(Threat_mean=('Threat','mean'),Acts_mean=('Acts','mean')).reset_index()
for c in ['Threat_mean','Acts_mean']:
    qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)

panel=wide[(wide.year>=2020)&(wide.year<=2025)].merge(stat,on='symbol',how='inner').merge(qg,on='period_q',how='left')
cols=['symbol','pidx','period_q','year','cfo_assets','log_assets','leverage','HL67','Threat_mean_z','Acts_mean_z']
x=panel[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
x['Threat_HL']=x.Threat_mean_z*x.HL67
x['Acts_HL']=x.Acts_mean_z*x.HL67
x=x.set_index(['symbol','pidx']).sort_index()

# Joint model: both interactions in the same regression
ex=['Threat_HL','Acts_HL','log_assets','leverage']
mod=PanelOLS(x.cfo_assets,x[ex],entity_effects=True,time_effects=True,drop_absorbed=True)
r=mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)

bT=float(r.params['Threat_HL']); bA=float(r.params['Acts_HL'])
seT=float(r.std_errors['Threat_HL']); seA=float(r.std_errors['Acts_HL'])
pT=float(r.pvalues['Threat_HL']); pA=float(r.pvalues['Acts_HL'])
cov=r.cov
# Direct coefficient equality test H0: beta_Threat = beta_Acts
var_diff=float(cov.loc['Threat_HL','Threat_HL']+cov.loc['Acts_HL','Acts_HL']-2*cov.loc['Threat_HL','Acts_HL'])
se_diff=float(np.sqrt(max(var_diff,0)))
diff=float(bT-bA)
z_diff=float(diff/se_diff) if se_diff>0 else np.nan
p_diff=float(2*norm.sf(abs(z_diff))) if np.isfinite(z_diff) else np.nan

joint=pd.DataFrame([
 {'term':'Threat x HighLiab','beta':bT,'se':seT,'p':pT},
 {'term':'Acts x HighLiab','beta':bA,'se':seA,'p':pA},
 {'term':'Threat minus Acts','beta':diff,'se':se_diff,'p':p_diff}
])
joint.to_csv(OUT/'joint_model_results.csv',index=False)

# Outcome distribution and economic significance. Since shocks are z-scored, beta = differential OCFA change for +1 SD shock.
raw=x.reset_index()
summary_rows=[]
for label,d in [('All',raw),('HighLiab=1',raw[raw.HL67==1]),('HighLiab=0',raw[raw.HL67==0])]:
    y=d.cfo_assets
    summary_rows.append({'group':label,'n':len(d),'firms':d.symbol.nunique(),'mean_ocfa':float(y.mean()),'median_ocfa':float(y.median()),'sd_ocfa':float(y.std(ddof=1)),'p25_ocfa':float(y.quantile(.25)),'p75_ocfa':float(y.quantile(.75))})
desc=pd.DataFrame(summary_rows); desc.to_csv(OUT/'ocfa_descriptive.csv',index=False)
all_mean=float(desc.loc[desc.group=='All','mean_ocfa'].iloc[0]); all_sd=float(desc.loc[desc.group=='All','sd_ocfa'].iloc[0]); high_mean=float(desc.loc[desc.group=='HighLiab=1','mean_ocfa'].iloc[0]); high_sd=float(desc.loc[desc.group=='HighLiab=1','sd_ocfa'].iloc[0])

def econ(name,beta):
    return {
      'shock':name,
      'beta_per_1sd_shock':float(beta),
      'percentage_point_change_ocfa':float(beta*100),
      'beta_as_share_of_overall_mean':float(beta/all_mean) if all_mean!=0 else None,
      'beta_as_pct_of_overall_mean':float(beta/all_mean*100) if all_mean!=0 else None,
      'beta_in_overall_outcome_sd':float(beta/all_sd) if all_sd!=0 else None,
      'beta_as_share_of_highliab_mean':float(beta/high_mean) if high_mean!=0 else None,
      'beta_as_pct_of_highliab_mean':float(beta/high_mean*100) if high_mean!=0 else None,
      'beta_in_highliab_outcome_sd':float(beta/high_sd) if high_sd!=0 else None,
    }
econ_df=pd.DataFrame([econ('Threat',bT),econ('Acts',bA)])
econ_df.to_csv(OUT/'economic_significance.csv',index=False)

assessment={
 'sample_n':int(r.nobs),
 'sample_firms':int(raw.symbol.nunique()),
 'highliab_cut':float(cut),
 'joint_model':{
   'Threat_beta':bT,'Threat_se':seT,'Threat_p':pT,
   'Acts_beta':bA,'Acts_se':seA,'Acts_p':pA,
   'Threat_minus_Acts':diff,'difference_se':se_diff,'difference_z':z_diff,'difference_p':p_diff,
   'equal_coefficients_rejected_5pct':bool(p_diff<.05) if np.isfinite(p_diff) else None
 },
 'outcome_distribution':{
   'overall_mean':all_mean,'overall_sd':all_sd,
   'highliab_mean':high_mean,'highliab_sd':high_sd
 },
 'economic_significance':econ_df.to_dict(orient='records'),
 'interpretation_rule':'Threat and Acts are standardized. Each interaction beta is the differential change in OCFA for HighLiab firms relative to other firms associated with a one-standard-deviation increase in the corresponding geopolitical-risk component, conditional on firm FE, quarter FE, size, and leverage.',
 'caution':'A statistically significant difference test compares the two conditional association coefficients. It does not by itself establish a causal difference between Threat and Acts.'
}
(OUT/'assessment.json').write_text(json.dumps(assessment,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(assessment,indent=2,ensure_ascii=False))
