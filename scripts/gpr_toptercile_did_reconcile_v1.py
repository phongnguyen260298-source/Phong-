from __future__ import annotations
import json,re,warnings
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import chi2
warnings.filterwarnings('ignore')
PANEL=Path('output/xy_screen/vci42_analysis_panel_2020_2025.csv'); EXPO=Path('data/research_cohorts/vci578_preexposure.csv'); OUT=Path('data/gpr_toptercile_did_reconcile_v1'); OUT.mkdir(parents=True,exist_ok=True)
p=pd.read_csv(PANEL); p['symbol']=p.symbol.astype(str).str.upper().str.strip(); p['period_q']=p.period_q.astype(str); p['pidx']=p.year.astype(int)*4+p.q.astype(int)
e=pd.read_csv(EXPO); e['symbol']=e.symbol.astype(str).str.upper().str.strip(); e=e[['symbol','liab_exp']].drop_duplicates('symbol'); cut_all=float(e.liab_exp.quantile(2/3)); e['HighLiab_TopTercile']=(e.liab_exp>=cut_all).astype(int)
# GPR
u='https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv'; g=pd.read_csv(u)
def n(s): return re.sub(r'[^a-z0-9]+','',str(s).lower())
def pick(pats):
  for pat in pats:
    q=n(pat)
    for c in g.columns:
      if n(c)==q or q in n(c): return c
  return None
dc=pick(['date','day']); ac=pick(['ACTS_GPR_AI','GPR_ACTS','acts']); th=pick(['THREATS_GPR_AI','GPR_THREATS','threats'])
g=g.rename(columns={dc:'date',ac:'Acts',th:'Threat'}); g.date=pd.to_datetime(g.date,errors='coerce'); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')]; g['period_q']=g.date.dt.to_period('Q').astype(str); qg=g.groupby('period_q')[['Acts','Threat']].mean().reset_index()
for c in ['Acts','Threat']: qg[c]=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)
d=p.merge(e,on='symbol',how='inner',validate='many_to_one').merge(qg,on='period_q',how='left',validate='many_to_one').sort_values(['symbol','pidx']); d['post2022']=(d.period_q>='2022Q1').astype(int); d['fakepost2021']=(d.period_q>='2021Q1').astype(int)
# diagnostic overlap-recomputed tercile only, never primary
firm=d[['symbol','liab_exp']].drop_duplicates(); cut_overlap=float(firm.liab_exp.quantile(2/3)); d['HighLiab_TopTercile_overlap']=(d.liab_exp>=cut_overlap).astype(int)

def fit(data,term,group,controls=('log_assets',),cov='twoway'):
  cols=['symbol','pidx','period_q','cfo_assets',term,group]+list(controls); x=data[cols].replace([np.inf,-np.inf],np.nan).dropna().copy(); k=f'{term}_X_{group}'; x[k]=x[term]*x[group]; X=[k]+list(controls); x=x.set_index(['symbol','pidx']).sort_index(); m=PanelOLS(x.cfo_assets,x[X],entity_effects=True,time_effects=True,drop_absorbed=True)
  if cov=='twoway': r=m.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
  elif cov=='dk': r=m.fit(cov_type='kernel',kernel='bartlett',bandwidth=4)
  else: r=m.fit(cov_type='clustered',cluster_entity=True)
  return {'n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params[k]),'se':float(r.std_errors[k]),'p':float(r.pvalues[k]),'r2_within':float(r.rsquared_within),'cov':cov}
rows=[]
for grp,label in [('HighLiab_TopTercile','locked_all378'),('HighLiab_TopTercile_overlap','diagnostic_overlap')]:
  for shock in ['Acts','Threat']:
    for cov in ['twoway','dk','entity']:
      z=fit(d,shock,grp,('log_assets',),cov); z.update(spec=f'{shock}_{label}',group=grp); rows.append(z)
  for cov in ['twoway','dk','entity']:
    z=fit(d,'post2022',grp,('log_assets',),cov); z.update(spec=f'DID_Post2022_{label}',group=grp); rows.append(z)
# placebo pre-2022
pre=d[d.period_q<'2022Q1'].copy()
z=fit(pre,'fakepost2021','HighLiab_TopTercile',('log_assets',),'twoway'); z.update(spec='PLACEBO_Fake2021_locked_all378',group='HighLiab_TopTercile'); rows.append(z)
res=pd.DataFrame(rows); res.to_csv(OUT/'results.csv',index=False)
# Dynamic DID primary top-tercile. Avoid the previous singular joint-Wald bug: report individual pre coefficients and a conservative max-pre p flag; also compute covariance pseudo-inverse Wald when estimable.
base=2022*4+1; ed=d[['symbol','pidx','cfo_assets','log_assets','HighLiab_TopTercile']].dropna().copy(); ed['rel']=(ed.pidx-base).clip(-7,12); ks=[k for k in sorted(ed.rel.unique()) if k!=-1]; terms=[]; pairs=[]
for k in ks:
  nm=('e_m'+str(abs(int(k))) if k<0 else 'e_p'+str(int(k))); ed[nm]=(ed.rel.eq(k).astype(int)*ed.HighLiab_TopTercile); terms.append(nm); pairs.append((int(k),nm))
x=ed[['symbol','pidx','cfo_assets','log_assets']+terms].set_index(['symbol','pidx']).sort_index(); r=PanelOLS(x.cfo_assets,x[terms+['log_assets']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
ev=[]
for k,nm in pairs:
  ev.append({'rel_q':k,'term':nm,'beta':float(r.params.get(nm,np.nan)),'se':float(r.std_errors.get(nm,np.nan)),'p':float(r.pvalues.get(nm,np.nan))})
ev=pd.DataFrame(ev); ev.to_csv(OUT/'eventstudy.csv',index=False)
pre_names=[nm for k,nm in pairs if k<-1 and nm in r.params.index and np.isfinite(r.std_errors.get(nm,np.nan))]
pre_individual_sig=ev[(ev.rel_q<-1)&(ev.p<.05)].to_dict('records')
wald_p=np.nan; wald_stat=np.nan; rank=0
if pre_names:
  idx=[list(r.params.index).index(nm) for nm in pre_names]; b=r.params.values[idx]; V=r.cov.values[np.ix_(idx,idx)]; rank=int(np.linalg.matrix_rank(V));
  if rank>0:
    Vinv=np.linalg.pinv(V); wald_stat=float(b.T@Vinv@b); wald_p=float(1-chi2.cdf(wald_stat,rank))
qa={'panel_symbols':int(p.symbol.nunique()),'exposure_symbols':int(e.symbol.nunique()),'overlap_symbols':int(d.symbol.nunique()),'panel_to_exposure_coverage':float(d.symbol.nunique()/p.symbol.nunique()),'cut_toptercile_all378':cut_all,'cut_toptercile_overlap325':cut_overlap,'primary_treated_firms':int(d.loc[d.HighLiab_TopTercile==1,'symbol'].nunique()),'primary_control_firms':int(d.loc[d.HighLiab_TopTercile==0,'symbol'].nunique()),'overlap_treated_firms':int(d.loc[d.HighLiab_TopTercile_overlap==1,'symbol'].nunique()),'duplicates':int(d.duplicated(['symbol','period_q']).sum()),'period_min':d.period_q.min(),'period_max':d.period_q.max(),'pretrend_valid_terms':pre_names,'pretrend_individual_sig_5pct':pre_individual_sig,'pretrend_joint_wald_stat_pinv':wald_stat,'pretrend_joint_df_rank':rank,'pretrend_joint_p_pinv':wald_p}
# reconciliation against old locked file values
a=res[(res.spec=='Acts_locked_all378')&(res['cov']=='twoway')].iloc[0].to_dict(); did=res[(res.spec=='DID_Post2022_locked_all378')&(res['cov']=='twoway')].iloc[0].to_dict(); placebo=res[res.spec=='PLACEBO_Fake2021_locked_all378'].iloc[0].to_dict()
assessment={'qa':qa,'acts_toptercile_current_exactpanel':a,'did_toptercile_current_exactpanel':did,'placebo_fake2021':placebo,'old_locked_reference':{'acts_beta':-0.004472,'acts_p':0.000392,'did_p':0.001569,'pretrend_p':0.680634,'group_label':'HighLiab_TopTercile'},'replication_gap':{'acts_beta_difference':float(a['beta']-(-0.004472)),'acts_p_current':float(a['p']),'did_p_current':float(did['p'])},'pretrend_pass':bool(len(pre_individual_sig)==0 and (np.isnan(wald_p) or wald_p>.05)),'interpretation':'If current exact-panel TopTercile fails to reproduce old locked results, old corporate evidence is not portable to the refreshed exact-item-ID panel and must not anchor the thesis without lineage reconciliation.'}
(OUT/'qa_summary.json').write_text(json.dumps(qa,indent=2,allow_nan=True),encoding='utf-8'); (OUT/'assessment.json').write_text(json.dumps(assessment,indent=2,allow_nan=True),encoding='utf-8'); print(json.dumps(assessment,indent=2,allow_nan=True)); print(res.to_string(index=False)); print(ev.to_string(index=False))