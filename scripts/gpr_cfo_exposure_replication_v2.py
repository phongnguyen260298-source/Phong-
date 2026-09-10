from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT=Path('input/vci42')
PANEL=Path('output/xy_screen/vci42_analysis_panel_2020_2025.csv')
LEGACY=Path('data/research_cohorts/vci578_preexposure.csv')
OUT=Path('data/gpr_cfo_exposure_replication_v2'); OUT.mkdir(parents=True,exist_ok=True)

def pq(v):
    m=re.search(r'(20\d{2}|19\d{2})\D*Q\D*([1-4])',str(v),re.I)
    return f'{m.group(1)}Q{m.group(2)}' if m else None

# Independently rebuild pre-treatment current-liability exposure from exact VCI item ids.
rows=[]
for f in sorted(ROOT.glob('chunk_*/fundamental_quarter/balance_sheet/*.parquet')):
    x=pd.read_parquet(f,columns=['period','item_id','value'])
    x=x[x.item_id.isin(['BS_TOTAL_ASSETS','BS_SHORT_TERM_LIABILITIES'])].copy()
    if x.empty: continue
    x['symbol']=f.stem.upper().strip(); x['period_q']=x.period.map(pq); x['year']=pd.to_numeric(x.period_q.str[:4],errors='coerce')
    x=x[x.year<2020]
    if x.empty: continue
    x['metric']=x.item_id.map({'BS_TOTAL_ASSETS':'assets','BS_SHORT_TERM_LIABILITIES':'current_liab'}); x['value']=pd.to_numeric(x.value,errors='coerce')
    rows.append(x[['symbol','period_q','metric','value']])
prelong=pd.concat(rows,ignore_index=True).drop_duplicates(['symbol','period_q','metric'],keep='last')
pre=prelong.pivot(index=['symbol','period_q'],columns='metric',values='value').reset_index()
pre['liab_ratio']=pre.current_liab/pre.assets.replace(0,np.nan)
pre=pre.replace([np.inf,-np.inf],np.nan)
# Require at least 2 distinct pre-2020 quarters and use the within-firm median to reduce one-quarter noise.
stat=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),liab_raw=('liab_ratio','median')).reset_index()
stat=stat[(stat.pre_quarters>=2)&stat.liab_raw.notna()].copy()
lo,hi=stat.liab_raw.quantile([.01,.99]); stat['liab_clip']=stat.liab_raw.clip(lo,hi); stat['liab_exp_rederived']=(stat.liab_clip-stat.liab_clip.mean())/stat.liab_clip.std(ddof=0)
cut=stat.liab_exp_rederived.quantile(2/3); stat['HighLiab_rederived']=(stat.liab_exp_rederived>=cut).astype(int)

# Compare with legacy exposure where possible.
legacy=pd.read_csv(LEGACY,usecols=['symbol','liab_exp']); legacy['symbol']=legacy.symbol.astype(str).str.upper().str.strip(); legacy=legacy.drop_duplicates('symbol')
cmp=stat.merge(legacy,on='symbol',how='inner')
corr=float(cmp[['liab_exp_rederived','liab_exp']].corr().iloc[0,1]) if len(cmp)>2 else np.nan

# Official AI-GPR daily -> quarterly Threat/Acts z scores.
g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv')
def n(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(pats):
    for p0 in pats:
        q=n(p0)
        for c in g.columns:
            if n(c)==q or q in n(c): return c
    return None
dc=pick(['date','day']); th=pick(['THREATS_GPR_AI','GPR_THREATS','threats']); ac=pick(['ACTS_GPR_AI','GPR_ACTS','acts'])
if not all([dc,th,ac]): raise RuntimeError(f'GPR schema unresolved: {list(g.columns)}')
g=g.rename(columns={dc:'date',th:'Threat',ac:'Acts'}); g['date']=pd.to_datetime(g.date,errors='coerce'); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')]; g['period_q']=g.date.dt.to_period('Q').astype(str)
qg=g.groupby('period_q')[['Threat','Acts']].mean().reset_index()
for c in ['Threat','Acts']:
    qg[c]=pd.to_numeric(qg[c],errors='coerce'); qg[c]=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)

p=pd.read_csv(PANEL); p['symbol']=p.symbol.astype(str).str.upper().str.strip(); p['period_q']=p.period_q.astype(str); p['pidx']=p.year.astype(int)*4+p.q.astype(int)
d=p.merge(stat[['symbol','liab_exp_rederived','HighLiab_rederived','pre_quarters']],on='symbol',how='inner',validate='many_to_one').merge(qg,on='period_q',how='left',validate='many_to_one')

# Exact TWFE model, firm and quarter clustered SE.
def fit(data,shock,expo,exclude=False,wins=False,joint=False):
    cols=['symbol','pidx','period_q','cfo_assets','log_assets','leverage',expo,'Threat','Acts']
    x=data[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if exclude: x=x[~x.period_q.isin(['2022Q1','2022Q2'])]
    x[f'{shock}_X_exp']=x[shock]*x[expo]
    X=[f'{shock}_X_exp','log_assets','leverage']
    if joint:
        other='Acts' if shock=='Threat' else 'Threat'; x[f'{other}_X_exp']=x[other]*x[expo]; X=[f'{shock}_X_exp',f'{other}_X_exp','log_assets','leverage']
    if wins:
        for c in ['cfo_assets']+X:
            a,b=x[c].quantile([.01,.99]); x[c]=x[c].clip(a,b)
    x=x.set_index(['symbol','pidx']).sort_index(); r=PanelOLS(x.cfo_assets,x[X],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    key=f'{shock}_X_exp'; return {'n':len(x),'firms':x.index.get_level_values(0).nunique(),'beta':float(r.params[key]),'se':float(r.std_errors[key]),'p':float(r.pvalues[key]),'r2_within':float(r.rsquared_within)}

rows=[]
for shock in ['Threat','Acts']:
    for expo,ename in [('liab_exp_rederived','continuous'),('HighLiab_rederived','top_tercile')]:
        for spec,kw in [('main',{}),('exclude_2022H1',{'exclude':True}),('winsor_1_99',{'wins':True})]:
            rec={'shock':shock,'exposure':ename,'spec':spec}; rec.update(fit(d,shock,expo,**kw)); rows.append(rec)
    rec={'shock':shock,'exposure':'continuous','spec':'joint_threat_acts'}; rec.update(fit(d,shock,'liab_exp_rederived',joint=True)); rows.append(rec)
res=pd.DataFrame(rows)
# family BH-FDR by exposure/spec family
res['q_fdr_family']=np.nan
for key,idx in res.groupby(['exposure','spec']).groups.items():
    ii=list(idx); pv=res.loc[ii,'p'].to_numpy(); order=np.argsort(pv); m=len(pv); q=np.empty(m); run=1.
    for j in range(m-1,-1,-1):
        k=order[j]; run=min(run,pv[k]*m/(j+1)); q[k]=min(1.,run)
    res.loc[ii,'q_fdr_family']=q
res.to_csv(OUT/'replication_results.csv',index=False)
stat.to_csv(OUT/'rederived_preexposure.csv',index=False)
qa={'pre_rows':int(len(pre)),'pre_symbols_any':int(pre.symbol.nunique()),'pre_symbols_2plus':int(len(stat)),'legacy_exposure_symbols':int(legacy.symbol.nunique()),'overlap_rederived_legacy':int(len(cmp)),'corr_rederived_legacy':corr,'analysis_symbols':int(d.symbol.nunique()),'analysis_rows':int(len(d)),'duplicate_symbol_quarter':int(d.duplicated(['symbol','period_q']).sum()),'period_min':str(d.period_q.min()),'period_max':str(d.period_q.max()),'missing_cfo_assets':float(d.cfo_assets.isna().mean()),'treated_rederived':int(stat.HighLiab_rederived.sum()),'control_rederived':int((stat.HighLiab_rederived==0).sum())}
(OUT/'qa_summary.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps({'qa':qa,'results':res.to_dict('records'),'note':'Independent replication using exact VCI pre-2020 current-liabilities and assets; differential association wording only.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'qa':qa,'results':res.to_dict('records')},ensure_ascii=False,indent=2))