from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from linearmodels.panel import PanelOLS

ROOT=Path('.')
PANEL=Path('output/xy_screen/vci42_analysis_panel_2020_2025.csv')
EXPO=Path('data/research_cohorts/vci578_preexposure.csv')
OUT=Path('data/gpr_cfo_core_robustness_v1'); OUT.mkdir(parents=True,exist_ok=True)

# Locked CFO panel rebuilt from exact VCI item_ids.
p=pd.read_csv(PANEL)
p['symbol']=p.symbol.astype(str).str.upper().str.strip()
p['period_q']=p.period_q.astype(str)
p['pidx']=p.year.astype(int)*4+p.q.astype(int)

# Pre-treatment exposure cohort; no post-2020 accounting variables are used to define exposure.
e=pd.read_csv(EXPO)
e['symbol']=e.symbol.astype(str).str.upper().str.strip()
e=e[['symbol','liab_exp']].drop_duplicates('symbol')

# Official AI-GPR daily data; aggregate Threats/Acts to quarter, then z-score over 2020-2025.
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
g=g.rename(columns={dc:'date',th:'Threat',ac:'Acts'})
g['date']=pd.to_datetime(g.date,errors='coerce')
g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].copy(); g['period_q']=g.date.dt.to_period('Q').astype(str)
qg=g.groupby('period_q')[['Threat','Acts']].mean().reset_index()
for c in ['Threat','Acts']:
    qg[c]=pd.to_numeric(qg[c],errors='coerce')
    qg[c]=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)

d=p.merge(e,on='symbol',how='inner',validate='many_to_one').merge(qg,on='period_q',how='left',validate='many_to_one')
d=d.sort_values(['symbol','pidx']).copy()
# liability exposure file is already standardized in upstream locked pipeline; preserve it as-is.
# Alternative binary definitions are pre-declared quantile cuts, not outcome-selected.
for qq,name in [(0.50,'high50'),(2/3,'high33'),(0.75,'high25')]:
    cut=d[['symbol','liab_exp']].drop_duplicates().liab_exp.quantile(qq)
    d[name]=(d.liab_exp>=cut).astype(int)

# Leads/lags of shocks only. Exposure remains fixed pre-treatment.
for s in ['Threat','Acts']:
    d[f'{s}_lag1']=d[s].shift(1)
    d[f'{s}_lead1']=d[s].shift(-1)
# shift above is global; correct at quarter level by mapping instead of firm order.
for s in ['Threat','Acts']:
    mp=qg.set_index('period_q')[s]
    ordered=list(qg.period_q)
    lag={ordered[i]: (mp.loc[ordered[i-1]] if i>0 else np.nan) for i in range(len(ordered))}
    lead={ordered[i]: (mp.loc[ordered[i+1]] if i<len(ordered)-1 else np.nan) for i in range(len(ordered))}
    d[f'{s}_lag1']=d.period_q.map(lag); d[f'{s}_lead1']=d.period_q.map(lead)

qa={
 'panel_rows':int(len(p)),'panel_symbols':int(p.symbol.nunique()),
 'exposure_symbols':int(e.symbol.nunique()),'analysis_symbols':int(d.symbol.nunique()),
 'analysis_rows':int(len(d)),'duplicate_symbol_quarter':int(d.duplicated(['symbol','period_q']).sum()),
 'period_min':str(d.period_q.min()),'period_max':str(d.period_q.max()),
 'missing':{c:float(d[c].isna().mean()) for c in ['cfo_assets','log_assets','leverage','liab_exp','Threat','Acts']},
 'binary_groups':{k:{'treated_firms':int(d.loc[d[k]==1,'symbol'].nunique()),'control_firms':int(d.loc[d[k]==0,'symbol'].nunique())} for k in ['high50','high33','high25']}
}

def fit(data,y='cfo_assets',terms=(),exclude_2022h1=False,winsor=False):
    cols=['symbol','pidx','period_q',y,'log_assets','leverage','liab_exp']+list(terms)
    x=data[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if exclude_2022h1: x=x[~x.period_q.isin(['2022Q1','2022Q2'])]
    X=[]
    for t in terms:
        if '*' in t:
            a,b=t.split('*'); nm=f'{a}_X_{b}'; x[nm]=x[a]*x[b]; X.append(nm)
        else: X.append(t)
    X += ['log_assets','leverage']
    if winsor:
        for c in [y]+X:
            lo,hi=x[c].quantile([.01,.99]); x[c]=x[c].clip(lo,hi)
    x=x.set_index(['symbol','pidx']).sort_index()
    r=PanelOLS(x[y],x[X],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    return x,r

rows=[]
def add(spec,shock,expo,terms,**kwargs):
    try:
        x,r=fit(d,terms=terms,**kwargs)
        key=f'{shock}_X_{expo}'
        rows.append({'spec':spec,'shock':shock,'exposure':expo,'n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params.get(key,np.nan)),'se':float(r.std_errors.get(key,np.nan)),'p':float(r.pvalues.get(key,np.nan)),'r2_within':float(r.rsquared_within)})
    except Exception as ex:
        rows.append({'spec':spec,'shock':shock,'exposure':expo,'error':f'{type(ex).__name__}: {ex}'})

# Main continuous exposure models + robustness.
for s in ['Threat','Acts']:
    add('continuous_main',s,'liab_exp',[f'{s}*liab_exp'])
    add('continuous_excl_2022H1',s,'liab_exp',[f'{s}*liab_exp'],exclude_2022h1=True)
    add('continuous_winsor_1_99',s,'liab_exp',[f'{s}*liab_exp'],winsor=True)
    add('lag1',s,'liab_exp',[f'{s}_lag1*liab_exp'])
    add('lead1_placebo',s,'liab_exp',[f'{s}_lead1*liab_exp'])
# Binary sensitivity thresholds.
for s in ['Threat','Acts']:
    for grp in ['high50','high33','high25']:
        add('binary_threshold',s,grp,[f'{s}*{grp}'])
# Joint Threat and Acts: independent explanatory content.
try:
    x,r=fit(d,terms=['Threat*liab_exp','Acts*liab_exp'])
    for s in ['Threat','Acts']:
        k=f'{s}_X_liab_exp'
        rows.append({'spec':'joint_threat_acts','shock':s,'exposure':'liab_exp','n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params[k]),'se':float(r.std_errors[k]),'p':float(r.pvalues[k]),'r2_within':float(r.rsquared_within)})
except Exception as ex:
    rows.append({'spec':'joint_threat_acts','error':f'{type(ex).__name__}: {ex}'})

res=pd.DataFrame(rows)
# BH-FDR by declared family to avoid pooling structurally different robustness checks.
res['q_fdr_family']=np.nan
for fam,idx in res.groupby('spec').groups.items():
    ii=[i for i in idx if pd.notna(res.loc[i,'p'])]
    if not ii: continue
    pv=res.loc[ii,'p'].to_numpy(); order=np.argsort(pv); n=len(pv); q=np.empty(n); run=1.0
    for j in range(n-1,-1,-1):
        k=order[j]; run=min(run,pv[k]*n/(j+1)); q[k]=min(1.0,run)
    res.loc[ii,'q_fdr_family']=q
res.to_csv(OUT/'core_robustness_results.csv',index=False)
(OUT/'qa_summary.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
summary={'qa':qa,'results':res.replace({np.nan:None}).to_dict('records'),'causal_note':'TWFE exposure interactions are interpreted as differential associations unless stronger identification supports causal language.'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))