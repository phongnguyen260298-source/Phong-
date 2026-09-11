from __future__ import annotations
import json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

PANEL=Path('output/xy_screen/vci42_analysis_panel_2020_2025.csv')
EXPO=Path('data/research_cohorts/vci578_preexposure.csv')
OUT=Path('data/gpr_full_model_did_audit_v1'); OUT.mkdir(parents=True,exist_ok=True)

p=pd.read_csv(PANEL)
p['symbol']=p['symbol'].astype(str).str.upper().str.strip()
p['period_q']=p['period_q'].astype(str)
p['year']=pd.to_numeric(p['year'],errors='coerce').astype('Int64')
p['q']=pd.to_numeric(p['q'],errors='coerce').astype('Int64')
p['pidx']=p['year'].astype(int)*4+p['q'].astype(int)
# locked exposure
e=pd.read_csv(EXPO)
e['symbol']=e['symbol'].astype(str).str.upper().str.strip()
need=['symbol','liab_exp','liab_exp_high']
miss=[c for c in need if c not in e.columns]
if miss: raise RuntimeError(f'missing exposure cols {miss}')
e=e[need].drop_duplicates('symbol')

# GPR daily -> quarterly means, standardized across 2020-2025
url='https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv'
g=pd.read_csv(url)
def normtxt(s): return re.sub(r'[^a-z0-9]+','',str(s).lower())
def pick(cols,pats):
    for pat in pats:
        q=normtxt(pat)
        for c in cols:
            z=normtxt(c)
            if z==q or q in z: return c
    return None

dc=pick(g.columns,['date','day']); th=pick(g.columns,['THREATS_GPR_AI','GPR_THREATS','threats']); ac=pick(g.columns,['ACTS_GPR_AI','GPR_ACTS','acts'])
if not all([dc,th,ac]): raise RuntimeError(f'GPR cols unresolved: {list(g.columns)}')
g=g.rename(columns={dc:'date',th:'Threat_raw',ac:'Acts_raw'})
g['date']=pd.to_datetime(g['date'],errors='coerce')
g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].copy()
g['period_q']=g.date.dt.to_period('Q').astype(str)
qg=g.groupby('period_q').agg(Threat_mean=('Threat_raw','mean'),Acts_mean=('Acts_raw','mean'),Acts_max=('Acts_raw','max'),Threat_max=('Threat_raw','max')).reset_index()
for c in ['Threat_mean','Acts_mean','Acts_max','Threat_max']:
    qg[c]=pd.to_numeric(qg[c],errors='coerce')
    qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)
# high-Acts regime prespecified from quarterly Acts_mean distribution
acts_p90=float(qg['Acts_mean'].quantile(.90)); qg['Acts_high_q']=(qg['Acts_mean']>=acts_p90).astype(int)

# Build merged panel on locked exposure
d=p.merge(e,on='symbol',how='left',validate='many_to_one').merge(qg,on='period_q',how='left',validate='many_to_one').sort_values(['symbol','pidx'])
# Reconstruct an independent pre-2022 treatment from panel itself, using only 2020Q1-2021Q4 leverage
pre=p[p['period_q']<'2022Q1'].copy()
prelev=(pre.groupby('symbol').agg(prelev=('leverage','mean'),pre_n=('period_q','nunique')).reset_index())
elig=prelev[prelev.pre_n>=4].copy(); med=float(elig.prelev.median()); elig['recon_highliab']=(elig.prelev>=med).astype(int)
d=d.merge(elig[['symbol','prelev','pre_n','recon_highliab']],on='symbol',how='left',validate='many_to_one')
d['post2022']=(d['period_q']>='2022Q1').astype(int)
# alternative post cutoffs are sensitivity only, never winner-selection
for cut in ['2022Q2','2022Q3']:
    d['post_'+cut]=(d['period_q']>=cut).astype(int)
# fake pre-period placebo break
# only interpreted on pre-2022 subsample
d['fakepost2021']=(d['period_q']>='2021Q1').astype(int)

# QA and selection diagnostics
panel_syms=set(p.symbol.unique()); expo_syms=set(e.symbol.unique()); overlap=panel_syms & expo_syms
locked=d[d.liab_exp_high.notna()].copy()
qa={
 'panel_rows':int(len(p)),'panel_symbols':int(p.symbol.nunique()),
 'exposure_rows':int(len(e)),'exposure_symbols':int(e.symbol.nunique()),
 'overlap_symbols':int(len(overlap)),'exposure_overlap_rate':float(len(overlap)/max(1,len(expo_syms))),
 'panel_to_exposure_coverage':float(len(overlap)/max(1,len(panel_syms))),
 'locked_analysis_symbols':int(locked.symbol.nunique()),
 'duplicate_panel_symbol_quarter':int(p.duplicated(['symbol','period_q']).sum()),
 'duplicate_locked_symbol_quarter':int(locked.duplicated(['symbol','period_q']).sum()),
 'period_min':str(p.period_q.min()),'period_max':str(p.period_q.max()),
 'locked_treated_firms':int(locked.loc[locked.liab_exp_high==1,'symbol'].nunique()),
 'locked_control_firms':int(locked.loc[locked.liab_exp_high==0,'symbol'].nunique()),
 'recon_eligible_firms':int(elig.symbol.nunique()),'recon_median_prelev':med,
 'recon_treated_firms':int(elig.loc[elig.recon_highliab==1,'symbol'].nunique()),'recon_control_firms':int(elig.loc[elig.recon_highliab==0,'symbol'].nunique()),
 'missing_locked':{c:float(locked[c].isna().mean()) for c in ['cfo_assets','log_assets','leverage','liab_exp','liab_exp_high','Acts_mean_z','Threat_mean_z']},
 'acts_quarterly_p90_threshold_raw':acts_p90,
 'high_acts_quarters':qg.loc[qg.Acts_high_q==1,'period_q'].tolist()
}
# Compare included vs excluded panel firms using pre-2022 firm means
pre_char=pre.groupby('symbol').agg(cfo_pre=('cfo_assets','mean'),size_pre=('log_assets','mean'),lev_pre=('leverage','mean')).reset_index()
pre_char['in_locked_exposure']=pre_char.symbol.isin(expo_syms).astype(int)
sel={}
for c in ['cfo_pre','size_pre','lev_pre']:
    a=pre_char.loc[pre_char.in_locked_exposure==1,c].dropna(); b=pre_char.loc[pre_char.in_locked_exposure==0,c].dropna()
    pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/max(1,len(a)+len(b)-2))
    smd=float((a.mean()-b.mean())/pooled) if pooled and np.isfinite(pooled) else np.nan
    sel[c]={'included_mean':float(a.mean()),'excluded_mean':float(b.mean()),'smd':smd,'n_in':int(len(a)),'n_out':int(len(b))}
qa['selection_diagnostics_pre2022']=sel

# Helpers
def fit_panel(data,y,terms,controls=('log_assets','leverage'),cov='twoway',winsor=False):
    components=[]
    for t in terms:
        components.extend(t.split('*'))
    cols=['symbol','pidx','period_q',y]+list(controls)+components
    cols=list(dict.fromkeys(cols))
    x=data[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
    X=[]
    for t in terms:
        if '*' in t:
            a,b=t.split('*'); nm=f'{a}_X_{b}'; x[nm]=pd.to_numeric(x[a],errors='coerce')*pd.to_numeric(x[b],errors='coerce'); X.append(nm)
        else: X.append(t)
    X += list(controls)
    if winsor:
        for c in [y]+X:
            if c in x and pd.api.types.is_numeric_dtype(x[c]):
                lo,hi=x[c].quantile([.01,.99]); x[c]=x[c].clip(lo,hi)
    x=x.set_index(['symbol','pidx']).sort_index()
    mod=PanelOLS(x[y],x[X],entity_effects=True,time_effects=True,drop_absorbed=True)
    if cov=='twoway': r=mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    elif cov=='entity': r=mod.fit(cov_type='clustered',cluster_entity=True)
    elif cov=='dk': r=mod.fit(cov_type='kernel',kernel='bartlett',bandwidth=4)
    else: r=mod.fit(cov_type='robust')
    return x,r

def extract(spec,data,terms,key,controls=('log_assets','leverage'),cov='twoway',winsor=False,family='core'):
    try:
        x,r=fit_panel(data,'cfo_assets',terms,controls,cov,winsor)
        return {'family':family,'spec':spec,'n':int(len(x)),'firms':int(x.index.get_level_values(0).nunique()),'beta':float(r.params.get(key,np.nan)),'se':float(r.std_errors.get(key,np.nan)),'p':float(r.pvalues.get(key,np.nan)),'r2_within':float(r.rsquared_within),'cov':cov,'controls':'+'.join(controls) if controls else 'none'}
    except Exception as ex:
        return {'family':family,'spec':spec,'beta':np.nan,'se':np.nan,'p':np.nan,'error':f'{type(ex).__name__}: {ex}','cov':cov,'controls':'+'.join(controls) if controls else 'none'}

rows=[]
# A. SAME-SAMPLE model audit, locked treatment
for controls,label in [((), 'FE_only'),(('log_assets',),'plus_size'),(('log_assets','leverage'),'plus_size_leverage')]:
    rows.append(extract(f'Acts_locked_{label}',locked,['Acts_mean_z*liab_exp_high'],'Acts_mean_z_X_liab_exp_high',controls=controls,family='acts_locked'))
# SE sensitivity on canonical +size (avoid potential post-treatment leverage over-control)
for cov in ['twoway','entity','dk','robust']:
    rows.append(extract(f'Acts_locked_SE_{cov}',locked,['Acts_mean_z*liab_exp_high'],'Acts_mean_z_X_liab_exp_high',controls=('log_assets',),cov=cov,family='se_sensitivity'))
# continuous exposure and Threat diagnostic
rows.append(extract('Acts_continuous_lockedsample',locked,['Acts_mean_z*liab_exp'],'Acts_mean_z_X_liab_exp',controls=('log_assets',),family='exposure_form'))
rows.append(extract('Threat_locked_highliab',locked,['Threat_mean_z*liab_exp_high'],'Threat_mean_z_X_liab_exp_high',controls=('log_assets',),family='threat_diag'))
# joint Threat/Acts
try:
    x,r=fit_panel(locked,'cfo_assets',['Acts_mean_z*liab_exp_high','Threat_mean_z*liab_exp_high'],('log_assets',),'twoway')
    for shock in ['Acts_mean_z','Threat_mean_z']:
        k=f'{shock}_X_liab_exp_high'; rows.append({'family':'joint','spec':f'joint_{shock}','n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params[k]),'se':float(r.std_errors[k]),'p':float(r.pvalues[k]),'r2_within':float(r.rsquared_within),'cov':'twoway','controls':'log_assets'})
except Exception as ex:
    rows.append({'family':'joint','spec':'joint_failed','p':np.nan,'error':str(ex)})
# winsor and exclude 2022H1
rows.append(extract('Acts_locked_winsor',locked,['Acts_mean_z*liab_exp_high'],'Acts_mean_z_X_liab_exp_high',controls=('log_assets',),winsor=True,family='robust'))
rows.append(extract('Acts_locked_excl2022H1',locked[~locked.period_q.isin(['2022Q1','2022Q2'])],['Acts_mean_z*liab_exp_high'],'Acts_mean_z_X_liab_exp_high',controls=('log_assets',),family='robust'))
# reconstructed pre-treatment group on wider sample
recon=d[d.recon_highliab.notna()].copy()
rows.append(extract('Acts_reconstructed_pre2022_highliab',recon,['Acts_mean_z*recon_highliab'],'Acts_mean_z_X_recon_highliab',controls=('log_assets',),family='reconstructed'))

# B. DID main and sensitivities
rows.append(extract('DID_Post2022_locked',locked,['post2022*liab_exp_high'],'post2022_X_liab_exp_high',controls=('log_assets',),family='did'))
rows.append(extract('DID_Post2022_locked_plus_leverage',locked,['post2022*liab_exp_high'],'post2022_X_liab_exp_high',controls=('log_assets','leverage'),family='did'))
for cov in ['twoway','entity','dk']:
    rows.append(extract(f'DID_Post2022_locked_SE_{cov}',locked,['post2022*liab_exp_high'],'post2022_X_liab_exp_high',controls=('log_assets',),cov=cov,family='did_se'))
for cut in ['2022Q2','2022Q3']:
    rows.append(extract(f'DID_Post_{cut}_sensitivity',locked,[f'post_{cut}*liab_exp_high'],f'post_{cut}_X_liab_exp_high',controls=('log_assets',),family='did_cutoff'))
rows.append(extract('DID_Post2022_reconstructed',recon,['post2022*recon_highliab'],'post2022_X_recon_highliab',controls=('log_assets',),family='did_reconstructed'))
# fake timing on only pre-2022 data
prelocked=locked[locked.period_q<'2022Q1'].copy()
rows.append(extract('PLACEBO_fake_Post2021_locked',prelocked,['fakepost2021*liab_exp_high'],'fakepost2021_X_liab_exp_high',controls=('log_assets',),family='placebo'))
# High-Acts regime DID-like exposure test
rows.append(extract('ShockDID_HighActsQ_locked',locked,['Acts_high_q*liab_exp_high'],'Acts_high_q_X_liab_exp_high',controls=('log_assets',),family='shock_did'))

res=pd.DataFrame(rows)
# FDR within families where >1 tests are confirmatory/sensitivity; placebo kept separate
res['q_fdr_family']=np.nan
for fam,idx in res.groupby('family').groups.items():
    ii=[i for i in idx if pd.notna(res.loc[i,'p'])]
    if len(ii):
        res.loc[ii,'q_fdr_family']=multipletests(res.loc[ii,'p'].astype(float).values,method='fdr_bh')[1]

# C. Dynamic DID for locked treatment, reference quarter 2021Q4
# quarter relative to 2022Q1: 2022Q1=0, 2021Q4=-1 reference
base_idx=2022*4+1
ed=locked[['symbol','pidx','period_q','cfo_assets','log_assets','liab_exp_high']].replace([np.inf,-np.inf],np.nan).dropna().copy()
ed['rel']=ed.pidx-base_idx
# bin extremes to reduce sparse tails
ed['relbin']=ed['rel'].clip(lower=-7,upper=12)
ks=sorted(k for k in ed.relbin.unique() if k!=-1)
terms=[]; names=[]
for k in ks:
    nm=('evt_m'+str(abs(int(k))) if k<0 else 'evt_p'+str(int(k)))
    ed[nm]=((ed.relbin==k).astype(int)*ed.liab_exp_high.astype(int)); terms.append(nm); names.append((k,nm))
x=ed[['symbol','pidx','cfo_assets','log_assets']+terms].dropna().set_index(['symbol','pidx']).sort_index()
mod=PanelOLS(x.cfo_assets,x[terms+['log_assets']],entity_effects=True,time_effects=True,drop_absorbed=True)
er=mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
ev=[]
for k,nm in names:
    ev.append({'rel_q':int(k),'term':nm,'beta':float(er.params.get(nm,np.nan)),'se':float(er.std_errors.get(nm,np.nan)),'p':float(er.pvalues.get(nm,np.nan))})
event=pd.DataFrame(ev)
# joint pretrend test excluding reference -1; all available k< -1
preterms=[nm for k,nm in names if k<-1 and nm in er.params.index]
pretrend_p=np.nan
if preterms:
    R=np.zeros((len(preterms),len(er.params)))
    for i,nm in enumerate(preterms): R[i,list(er.params.index).index(nm)]=1
    try: pretrend_p=float(er.wald_test(R).pval)
    except Exception: pretrend_p=np.nan

# Assessment: holes are evidence-driven, not significance hunting
holes=[]
if qa['panel_to_exposure_coverage']<0.8:
    holes.append({'severity':'HIGH','issue':'Exposure cohort covers less than 80% of CFO panel','evidence':qa['panel_to_exposure_coverage'],'implication':'Selection/external-validity risk; locked exposure sample is not the 578-firm panel.'})
for c,v in sel.items():
    if np.isfinite(v['smd']) and abs(v['smd'])>=0.2:
        holes.append({'severity':'HIGH' if abs(v['smd'])>=0.5 else 'MEDIUM','issue':f'Exposure-sample selection imbalance in {c}','evidence':v['smd'],'implication':'Included firms differ materially from excluded firms before treatment.'})
# helper values
by={r['spec']:r for r in rows}
canon=by.get('Acts_locked_plus_size',{})
canonlev=by.get('Acts_locked_plus_size_leverage',{})
reconr=by.get('Acts_reconstructed_pre2022_highliab',{})
did=by.get('DID_Post2022_locked',{})
placebo=by.get('PLACEBO_fake_Post2021_locked',{})
if pd.notna(canon.get('p',np.nan)) and canon.get('p',1)>0.05:
    holes.append({'severity':'CRITICAL','issue':'Acts x locked pre-treatment HighLiab does not replicate at 5% in exact locked sample','evidence':{'beta':canon.get('beta'),'p':canon.get('p')},'implication':'Previously locked main corporate claim requires reconciliation before thesis use.'})
if pd.notna(canon.get('beta',np.nan)) and pd.notna(reconr.get('beta',np.nan)) and np.sign(canon['beta'])!=np.sign(reconr['beta']):
    holes.append({'severity':'HIGH','issue':'Acts interaction changes sign under independently reconstructed pre-2022 HighLiab','evidence':{'locked':canon['beta'],'reconstructed':reconr['beta']},'implication':'Result is treatment-definition dependent.'})
if pd.notna(canon.get('p',np.nan)) and pd.notna(canonlev.get('p',np.nan)) and ((canon['p']<=.05)!=(canonlev['p']<=.05)):
    holes.append({'severity':'HIGH','issue':'Inference changes when contemporaneous leverage is controlled','evidence':{'size_only_p':canon['p'],'plus_leverage_p':canonlev['p']},'implication':'Potential over-control/mediator sensitivity.'})
if pd.notna(placebo.get('p',np.nan)) and placebo['p']<=.05:
    holes.append({'severity':'HIGH','issue':'Fake-2021 DID placebo is significant','evidence':{'beta':placebo.get('beta'),'p':placebo.get('p')},'implication':'Post-2022 DID may capture pre-existing differential dynamics.'})
if pd.notna(pretrend_p) and pretrend_p<=.05:
    holes.append({'severity':'CRITICAL','issue':'Dynamic DID rejects joint pretrend','evidence':pretrend_p,'implication':'Parallel-trends support fails for Post-2022 DID.'})
# universal identification caveat
holes.append({'severity':'HIGH','issue':'Post-2022 is a broad macro regime, not a uniquely exogenous GPR treatment','evidence':'2022 coincides with multiple macro shocks','implication':'Even a significant DID is supportive differential evidence, not clean causal GPR identification.'})
if len(qg)!=24:
    holes.append({'severity':'MEDIUM','issue':'Unexpected GPR quarterly coverage','evidence':len(qg),'implication':'Check daily-to-quarter aggregation.'})
holes.append({'severity':'MEDIUM','issue':'Only 24 quarter-level shock observations underpin common GPR variation','evidence':24,'implication':'Time-cluster asymptotics are limited; report two-way cluster plus Driscoll-Kraay sensitivity.'})

assessment={
 'qa':qa,
 'main_model':canon,
 'did_main':did,
 'dynamic_did_pretrend_joint_p':pretrend_p,
 'holes':holes,
 'readiness':'NEEDS_REVISION' if any(h['severity'] in ['CRITICAL','HIGH'] for h in holes) else 'READY_WITH_CAVEATS',
 'interpretation_lock':'Do not make causal GPR claims from Post-2022 DID alone. Main Acts claim is usable only if same-sample locked-treatment replication and inference sensitivities are satisfactory.',
 'design_note':'Primary audit uses locked pre-treatment HighLiab and size control. Contemporaneous leverage is treated as a sensitivity because it may be an over-control/mediator.'
}
res.to_csv(OUT/'model_did_results.csv',index=False)
event.to_csv(OUT/'dynamic_did_eventstudy.csv',index=False)
(OUT/'qa_summary.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
print(json.dumps(assessment,ensure_ascii=False,indent=2,allow_nan=True))
print(res.to_string(index=False))
print(event.to_string(index=False))
