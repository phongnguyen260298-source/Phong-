from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

PANEL=Path('output/xy_screen/vci42_analysis_panel_2020_2025.csv')
EXPO=Path('data/research_cohorts/vci578_preexposure.csv')
OUT=Path('data/gpr_threat_highliab_rescue_v3'); OUT.mkdir(parents=True,exist_ok=True)

p=pd.read_csv(PANEL)
p['symbol']=p.symbol.astype(str).str.upper().str.strip(); p['period_q']=p.period_q.astype(str); p['pidx']=p.year.astype(int)*4+p.q.astype(int)
e=pd.read_csv(EXPO); e['symbol']=e.symbol.astype(str).str.upper().str.strip()
e=e[['symbol','liab_exp','liab_exp_high']].drop_duplicates('symbol')

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
# Pre-specified quarterly measures aligned to anticipation theory
qg=g.groupby('period_q').agg(Threat_mean=('Threat','mean'), Threat_p90=('Threat',lambda x: x.quantile(.9)), Threat_max=('Threat','max'), Acts_mean=('Acts','mean')).reset_index()
# Extreme-threat day share, threshold fixed on full 2020-2025 daily distribution (not sample outcomes)
thr90=float(g['Threat'].quantile(.90)); spike=g.assign(threat_spike=(g['Threat']>=thr90).astype(int)).groupby('period_q')['threat_spike'].mean().rename('Threat_spike_share').reset_index(); qg=qg.merge(spike,on='period_q',how='left')
for c in ['Threat_mean','Threat_p90','Threat_max','Threat_spike_share','Acts_mean']:
    qg[c]=pd.to_numeric(qg[c],errors='coerce'); sd=qg[c].std(ddof=0); qg[c]=(qg[c]-qg[c].mean())/sd if sd and np.isfinite(sd) else qg[c]-qg[c].mean()

d=p.merge(e,on='symbol',how='inner',validate='many_to_one').merge(qg,on='period_q',how='left',validate='many_to_one').sort_values(['symbol','pidx'])
# Concordance diagnostic only: compare locked pre-treatment group vs sample-recomputed median group
cut=d[['symbol','liab_exp']].drop_duplicates().liab_exp.median(); d['high50_recomputed']=(d.liab_exp>=cut).astype(int)
firmcmp=d[['symbol','liab_exp_high','high50_recomputed']].drop_duplicates('symbol')
concord=float((firmcmp.liab_exp_high==firmcmp.high50_recomputed).mean())

ordered=list(qg.period_q); mp=qg.set_index('period_q')['Threat_mean']
lag={ordered[i]:(mp.loc[ordered[i-1]] if i>0 else np.nan) for i in range(len(ordered))}; lead={ordered[i]:(mp.loc[ordered[i+1]] if i<len(ordered)-1 else np.nan) for i in range(len(ordered))}
d['Threat_lag1']=d.period_q.map(lag); d['Threat_lead1']=d.period_q.map(lead)

def fit(data, shock, expo='liab_exp_high', winsor=False, exclude_2022h1=False, joint_acts=False):
    cols=['symbol','pidx','period_q','cfo_assets','log_assets','leverage',expo,shock]
    if joint_acts: cols += ['Acts_mean']
    x=data[list(dict.fromkeys(cols))].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if exclude_2022h1: x=x[~x.period_q.isin(['2022Q1','2022Q2'])]
    k=f'{shock}_X_{expo}'; x[k]=x[shock]*x[expo]; X=[k,'log_assets','leverage']
    if joint_acts:
        ka=f'Acts_mean_X_{expo}'; x[ka]=x['Acts_mean']*x[expo]; X=[k,ka,'log_assets','leverage']
    if winsor:
        for c in ['cfo_assets']+X:
            lo,hi=x[c].quantile([.01,.99]); x[c]=x[c].clip(lo,hi)
    x=x.set_index(['symbol','pidx']).sort_index()
    r=PanelOLS(x['cfo_assets'],x[X],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    return x,r,k

rows=[]
def add(spec,shock,**kw):
    try:
        x,r,k=fit(d,shock,**kw); rows.append({'spec':spec,'shock':shock,'n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params[k]),'se':float(r.std_errors[k]),'p':float(r.pvalues[k]),'r2_within':float(r.rsquared_within)})
    except Exception as ex:
        rows.append({'spec':spec,'shock':shock,'beta':np.nan,'se':np.nan,'p':np.nan,'error':f'{type(ex).__name__}: {ex}'})

# Locked family: exact treatment definition first; alternatives are theory-driven and declared here before execution.
add('PRIMARY_locked_highliab','Threat_mean')
add('winsor_1_99','Threat_mean',winsor=True)
add('exclude_2022H1','Threat_mean',exclude_2022h1=True)
add('lag1_persistence','Threat_lag1')
add('LEAD1_PLACEBO','Threat_lead1')
add('alt_threat_p90','Threat_p90')
add('alt_threat_max','Threat_max')
add('alt_threat_spike_share','Threat_spike_share')
add('joint_threat_acts','Threat_mean',joint_acts=True)

res=pd.DataFrame(rows)
# BH-FDR across prespecified non-placebo signal family; placebo is reported separately, not used to find significance.
signal_idx=res.index[~res['spec'].eq('LEAD1_PLACEBO') & res['p'].notna()].tolist(); res['q_fdr_prespecified']=np.nan
if signal_idx:
    pv=res.loc[signal_idx,'p'].to_numpy(); order=np.argsort(pv); n=len(pv); q=np.empty(n); run=1.0
    for j in range(n-1,-1,-1):
        k=order[j]; run=min(run,pv[k]*n/(j+1)); q[k]=min(1.0,run)
    res.loc[signal_idx,'q_fdr_prespecified']=q

qa={'panel_rows':int(len(p)),'panel_symbols':int(p.symbol.nunique()),'exposure_symbols':int(e.symbol.nunique()),'analysis_symbols':int(d.symbol.nunique()),'analysis_rows':int(len(d)),'duplicate_symbol_quarter':int(d.duplicated(['symbol','period_q']).sum()),'period_min':str(d.period_q.min()),'period_max':str(d.period_q.max()),'join_coverage_panel_to_exposure':float(d.symbol.nunique()/p.symbol.nunique()),'locked_highliab_treated_firms':int(d.loc[d.liab_exp_high==1,'symbol'].nunique()),'locked_highliab_control_firms':int(d.loc[d.liab_exp_high==0,'symbol'].nunique()),'locked_vs_recomputed_high50_concordance':concord,'locked_vs_recomputed_disagreements':int((firmcmp.liab_exp_high!=firmcmp.high50_recomputed).sum()),'missing':{c:float(d[c].isna().mean()) for c in ['cfo_assets','log_assets','leverage','liab_exp','liab_exp_high','Threat_mean','Acts_mean']},'daily_threat_90pct_threshold':thr90}

primary=res[res.spec=='PRIMARY_locked_highliab'].iloc[0].to_dict(); placebo=res[res.spec=='LEAD1_PLACEBO'].iloc[0].to_dict()
rob=res[res.spec.isin(['winsor_1_99','exclude_2022H1','alt_threat_p90','alt_threat_max','alt_threat_spike_share'])].copy()
upgrade=bool(pd.notna(primary.get('q_fdr_prespecified')) and primary['beta']<0 and primary['q_fdr_prespecified']<=0.05 and placebo['p']>0.10 and (rob['beta']<0).sum()>=3)
summary={'direction':'Threat x locked pre-treatment HighLiab rescue v3','primary':primary,'lead_placebo':placebo,'qa':qa,'upgrade_gate_pass':upgrade,'decision_rule':'Upgrade only if PRIMARY locked HighLiab beta<0 and q_FDR<=0.05; lead placebo p>0.10; and at least 3/5 prespecified robustness/alternative shock specifications retain negative sign. No model may be redefined after seeing outcomes.','causal_language':'Differential association / anticipation evidence only unless stronger exogenous identification is added.'}
res.to_csv(OUT/'results.csv',index=False); (OUT/'qa_summary.json').write_text(json.dumps(qa,indent=2),encoding='utf-8'); (OUT/'assessment.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))