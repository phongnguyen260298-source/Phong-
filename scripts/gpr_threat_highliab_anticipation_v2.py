from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

PANEL=Path('output/xy_screen/vci42_analysis_panel_2020_2025.csv')
EXPO=Path('data/research_cohorts/vci578_preexposure.csv')
OUT=Path('data/gpr_threat_highliab_anticipation_v2'); OUT.mkdir(parents=True,exist_ok=True)

p=pd.read_csv(PANEL)
p['symbol']=p.symbol.astype(str).str.upper().str.strip(); p['period_q']=p.period_q.astype(str); p['pidx']=p.year.astype(int)*4+p.q.astype(int)
e=pd.read_csv(EXPO); e['symbol']=e.symbol.astype(str).str.upper().str.strip(); e=e[['symbol','liab_exp']].drop_duplicates('symbol')

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
if not all([dc,th,ac]): raise RuntimeError(f'GPR columns unresolved: {list(g.columns)}')
g=g.rename(columns={dc:'date',th:'Threat',ac:'Acts'}); g['date']=pd.to_datetime(g.date,errors='coerce')
g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].copy(); g['period_q']=g.date.dt.to_period('Q').astype(str)
for c in ['Threat','Acts']: g[c]=pd.to_numeric(g[c],errors='coerce')

# Pre-specified quarterly shock summaries. Mean is primary; max and p90 are robustness only.
qg=g.groupby('period_q').agg(Threat_mean=('Threat','mean'),Threat_max=('Threat','max'),Threat_p90=('Threat',lambda x:x.quantile(.90)),Acts_mean=('Acts','mean')).reset_index()
for c in ['Threat_mean','Threat_max','Threat_p90','Acts_mean']:
    qg[c]=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)

d=p.merge(e,on='symbol',how='inner',validate='many_to_one').merge(qg,on='period_q',how='left',validate='many_to_one').sort_values(['symbol','pidx'])
firmexpo=d[['symbol','liab_exp']].drop_duplicates()
for qq,name in [(0.50,'high50'),(2/3,'high33'),(0.75,'high25')]:
    cut=firmexpo.liab_exp.quantile(qq); d[name]=(d.liab_exp>=cut).astype(int)

ordered=list(qg.period_q)
for s in ['Threat_mean','Threat_max','Threat_p90','Acts_mean']:
    mp=qg.set_index('period_q')[s]
    d[f'{s}_lag1']=d.period_q.map({ordered[i]:(mp.loc[ordered[i-1]] if i>0 else np.nan) for i in range(len(ordered))})
    d[f'{s}_lead1']=d.period_q.map({ordered[i]:(mp.loc[ordered[i+1]] if i<len(ordered)-1 else np.nan) for i in range(len(ordered))})

qa={
 'panel_rows':int(len(p)),'panel_symbols':int(p.symbol.nunique()),'exposure_symbols':int(e.symbol.nunique()),
 'analysis_symbols':int(d.symbol.nunique()),'analysis_rows':int(len(d)),
 'duplicate_symbol_quarter':int(d.duplicated(['symbol','period_q']).sum()),
 'period_min':str(d.period_q.min()),'period_max':str(d.period_q.max()),
 'join_coverage_panel_to_exposure':float(d.symbol.nunique()/p.symbol.nunique()) if p.symbol.nunique() else None,
 'missing':{c:float(d[c].isna().mean()) for c in ['cfo_assets','log_assets','leverage','liab_exp','Threat_mean','Acts_mean']},
 'binary_groups':{k:{'treated_firms':int(d.loc[d[k]==1,'symbol'].nunique()),'control_firms':int(d.loc[d[k]==0,'symbol'].nunique())} for k in ['high50','high33','high25']}
}

def fit(data,terms=(),exclude_2022h1=False,winsor=False):
    components=[]
    for t in terms: components.extend(t.split('*'))
    cols=list(dict.fromkeys(['symbol','pidx','period_q','cfo_assets','log_assets','leverage']+components))
    x=data[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if exclude_2022h1: x=x[~x.period_q.isin(['2022Q1','2022Q2'])]
    X=[]
    for t in terms:
        if '*' in t:
            a,b=t.split('*'); nm=f'{a}_X_{b}'; x[nm]=x[a]*x[b]; X.append(nm)
        else: X.append(t)
    X += ['log_assets','leverage']
    if winsor:
        for c in ['cfo_assets']+X:
            lo,hi=x[c].quantile([.01,.99]); x[c]=x[c].clip(lo,hi)
    x=x.set_index(['symbol','pidx']).sort_index()
    r=PanelOLS(x['cfo_assets'],x[X],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    return x,r

rows=[]
def add(spec,shock,expo,terms,key=None,**kwargs):
    try:
        x,r=fit(d,terms=terms,**kwargs); key=key or f'{shock}_X_{expo}'
        rows.append({'spec':spec,'shock':shock,'exposure':expo,'n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params.get(key,np.nan)),'se':float(r.std_errors.get(key,np.nan)),'p':float(r.pvalues.get(key,np.nan)),'r2_within':float(r.rsquared_within)})
    except Exception as ex:
        rows.append({'spec':spec,'shock':shock,'exposure':expo,'beta':np.nan,'se':np.nan,'p':np.nan,'error':f'{type(ex).__name__}: {ex}'})

# Primary specification and prespecified robustness checks.
add('primary_mean_continuous','Threat_mean','liab_exp',['Threat_mean*liab_exp'])
add('winsor_1_99','Threat_mean','liab_exp',['Threat_mean*liab_exp'],winsor=True)
add('exclude_2022H1','Threat_mean','liab_exp',['Threat_mean*liab_exp'],exclude_2022h1=True)
add('lag1_anticipation_persistence','Threat_mean','liab_exp',['Threat_mean_lag1*liab_exp'],key='Threat_mean_lag1_X_liab_exp')
add('lead1_placebo','Threat_mean','liab_exp',['Threat_mean_lead1*liab_exp'],key='Threat_mean_lead1_X_liab_exp')
for grp in ['high50','high33','high25']:
    add('binary_threshold','Threat_mean',grp,[f'Threat_mean*{grp}'])
for shock in ['Threat_max','Threat_p90']:
    add('alternative_shock_aggregation',shock,'liab_exp',[f'{shock}*liab_exp'])
# Joint Threat/Acts model tests whether threat association survives realized acts.
try:
    x,r=fit(d,terms=['Threat_mean*liab_exp','Acts_mean*liab_exp'])
    for s in ['Threat_mean','Acts_mean']:
        k=f'{s}_X_liab_exp'; rows.append({'spec':'joint_threat_acts','shock':s,'exposure':'liab_exp','n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params[k]),'se':float(r.std_errors[k]),'p':float(r.pvalues[k]),'r2_within':float(r.rsquared_within)})
except Exception as ex:
    rows.append({'spec':'joint_threat_acts','shock':'joint','exposure':'liab_exp','beta':np.nan,'p':np.nan,'error':f'{type(ex).__name__}: {ex}'})

res=pd.DataFrame(rows)
# BH-FDR across the full prespecified family, excluding benchmark Acts coefficient in joint model.
mask=(res.shock!='Acts_mean') & res.p.notna(); pv=res.loc[mask,'p'].to_numpy(); q=np.full(len(pv),np.nan)
if len(pv):
    order=np.argsort(pv); run=1.0; n=len(pv)
    for j in range(n-1,-1,-1):
        k=order[j]; run=min(run,pv[k]*n/(j+1)); q[k]=min(1.0,run)
    res.loc[mask,'q_fdr_prespecified']=q
res.to_csv(OUT/'threat_highliab_results.csv',index=False)

primary=res[res.spec=='primary_mean_continuous'].iloc[0].to_dict()
lead=res[res.spec=='lead1_placebo'].iloc[0].to_dict()
joint=res[(res.spec=='joint_threat_acts')&(res.shock=='Threat_mean')]
assessment={
 'direction':'Threat x HighLiab anticipation',
 'primary':primary,
 'lead_placebo':lead,
 'joint_threat_given_acts':joint.iloc[0].to_dict() if len(joint) else None,
 'qa':qa,
 'decision_rule':'Upgrade only if primary sign is negative, q_FDR<=0.05, lead placebo p>0.10, and sign is stable across winsor/exclude-2022H1 plus at least one alternative aggregation. Joint Threat|Acts is supportive but not mandatory.',
 'causal_language':'Associational/differential exposure language only unless a stronger exogenous identification strategy is added.',
 'selection_note':'vci578_preexposure.csv is an older 378-firm pre-treatment-exposure subset; it must not be described as a 578-firm sample.'
}
(OUT/'qa_summary.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps(assessment,ensure_ascii=False,indent=2,default=str))
