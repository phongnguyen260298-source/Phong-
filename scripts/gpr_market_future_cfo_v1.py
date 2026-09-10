from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm

ROOT=Path('.')
MKT=Path('data/gpr_market_top5_v4')
CFO=Path('output/xy_screen/vci42_analysis_panel_2020_2025.csv')
OUT=Path('data/gpr_market_future_cfo_v1'); OUT.mkdir(parents=True,exist_ok=True)

GROUPS=['HighLiab_TopTercile','CashConversionTrap','DoubleExposure_TopTercile','FragileLowMarginFunding','AssetCommitmentMismatch']
DIRECTIONS=[
    ('1 Threat×HighLiab anticipation','HighLiab_TopTercile','THREATS_z'),
    ('2 Acts×HighLiab realization','HighLiab_TopTercile','ACTS_z'),
    ('3 Acts×CashConversionTrap','CashConversionTrap','ACTS_z'),
    ('4 Threat×DoubleExposure','DoubleExposure_TopTercile','THREATS_z'),
    ('5a Threat×AssetCommitmentMismatch','AssetCommitmentMismatch','THREATS_z'),
    ('5b Threat×FragileLowMarginFunding','FragileLowMarginFunding','THREATS_z'),
]

px=pd.read_csv(MKT/'daily_price_panel.csv.gz',parse_dates=['date'])
px['symbol']=px.symbol.astype(str).str.upper().str.strip(); px['date']=pd.to_datetime(px.date).dt.normalize()
groups=pd.read_csv('data/research_cohorts/vci578_behavioral_groups.csv')
groups['symbol']=groups.symbol.astype(str).str.upper().str.strip()
g=pd.read_csv(MKT/'ai_gpr_data_daily.csv')

# Resolve the same GPR columns used by the market analyzer.
def pick(cols,pats):
    import re
    normed={re.sub(r'[^a-z0-9]','',c.lower()):c for c in cols}
    for p0 in pats:
        p=re.sub(r'[^a-z0-9]','',p0.lower())
        for n,c in normed.items():
            if n==p or p in n: return c
    return None

dc=pick(g.columns,['date','day']); thr=pick(g.columns,['THREATS_GPR_AI','GPR_THREATS','threats']); act=pick(g.columns,['ACTS_GPR_AI','GPR_ACTS','acts'])
if not all([dc,thr,act]): raise RuntimeError(f'GPR schema unresolved: {list(g.columns)}')
g=g.rename(columns={dc:'date',thr:'THREATS',act:'ACTS'}); g['date']=pd.to_datetime(g.date,errors='coerce').dt.normalize()
g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')].sort_values('date').copy()
for c in ['THREATS','ACTS']:
    g[c]=pd.to_numeric(g[c],errors='coerce')
    innov=np.log1p(g[c].clip(lower=0)).diff()
    g[c+'_z']=(innov-innov.mean())/innov.std(ddof=0)

# Calendar shocks are mapped to the next actual trading date, so weekend/non-trading shocks are not silently discarded.
trading=pd.DatetimeIndex(sorted(px.date.dropna().unique()))
map_rows=[]
for _,r in g[['date','THREATS_z','ACTS_z']].dropna(how='all',subset=['THREATS_z','ACTS_z']).iterrows():
    i=trading.searchsorted(r.date)
    if i<len(trading): map_rows.append({'trade_date':trading[i],'THREATS_z':r.THREATS_z,'ACTS_z':r.ACTS_z})
mg=pd.DataFrame(map_rows).groupby('trade_date',as_index=False)[['THREATS_z','ACTS_z']].sum(min_count=1)
d=px.merge(mg,left_on='date',right_on='trade_date',how='left').drop(columns=['trade_date'])
d['period_q']=d.date.dt.to_period('Q').astype(str)
for shock in ['THREATS_z','ACTS_z']:
    d[shock]=d[shock].fillna(0.0)
    d['prod_'+shock]=d['ar']*d[shock]

# Primary firm-quarter GPR-linked market-reaction measure: shock-weighted abnormal return.
q=d.groupby(['symbol','period_q']).agg(
    n_trade=('date','size'),
    mr_threat_num=('prod_THREATS_z','sum'), mr_threat_den=('THREATS_z',lambda x:np.abs(x).sum()),
    mr_acts_num=('prod_ACTS_z','sum'), mr_acts_den=('ACTS_z',lambda x:np.abs(x).sum()),
).reset_index()
q['THREATS_z'] = q.mr_threat_num/q.mr_threat_den.replace(0,np.nan)
q['ACTS_z'] = q.mr_acts_num/q.mr_acts_den.replace(0,np.nan)
q=q[['symbol','period_q','n_trade','THREATS_z','ACTS_z']]

# Event CAR5 robustness: use the already pre-specified top event dates, aligned to next trading day.
ev=pd.read_csv(MKT/'top_event_dates.csv',parse_dates=['event_date'])
ev['event_date']=pd.to_datetime(ev.event_date).dt.normalize()
erows=[]
for _,r in ev.iterrows():
    i=trading.searchsorted(r.event_date)
    if i>=len(trading): continue
    td=trading[i]; win=trading[i:min(i+5,len(trading))]
    z=px[px.date.isin(win)].groupby('symbol',as_index=False)['ar'].sum().rename(columns={'ar':'car5'})
    z['shock']=r.shock; z['trade_date']=td; z['period_q']=pd.Timestamp(td).to_period('Q').strftime('%YQ%q') if False else str(pd.Timestamp(td).to_period('Q'))
    erows.append(z)
efirm=pd.concat(erows,ignore_index=True) if erows else pd.DataFrame(columns=['symbol','car5','shock','trade_date','period_q'])
eq=efirm.groupby(['symbol','period_q','shock'],as_index=False)['car5'].mean().pivot(index=['symbol','period_q'],columns='shock',values='car5').reset_index()
eq=eq.rename(columns={'THREATS_z':'event5_THREATS_z','ACTS_z':'event5_ACTS_z'})
q=q.merge(eq,on=['symbol','period_q'],how='left')

cfo=pd.read_csv(CFO)
cfo['symbol']=cfo.symbol.astype(str).str.upper().str.strip(); cfo['period_q']=cfo.period_q.astype(str)
cfo=cfo[(cfo.year>=2020)&(cfo.year<=2025)].copy()
cfo['pidx']=cfo.year.astype(int)*4+cfo.q.astype(int)
nexty=cfo[['symbol','pidx','cfo_assets']].rename(columns={'pidx':'next_pidx','cfo_assets':'cfo_next'})
cfo['next_pidx']=cfo.pidx+1
cfo=cfo.merge(nexty,on=['symbol','next_pidx'],how='left')
base=cfo[['symbol','period_q','pidx','cfo_assets','cfo_next','log_assets','leverage']].merge(groups[['symbol']+GROUPS],on='symbol',how='inner').merge(q,on=['symbol','period_q'],how='inner')
base.to_csv(OUT/'linkage_panel.csv.gz',index=False,compression='gzip')

# QA before modeling.
qa={
    'market_symbols':int(px.symbol.nunique()),
    'group_file_symbols':int(groups.symbol.nunique()),
    'cfo_symbols_2020_2025':int(cfo.symbol.nunique()),
    'linkage_symbols':int(base.symbol.nunique()),
    'linkage_rows':int(len(base)),
    'duplicate_symbol_quarter':int(base.duplicated(['symbol','period_q']).sum()),
    'market_start':str(px.date.min().date()),'market_end':str(px.date.max().date()),
    'cfo_period_min':str(cfo.period_q.min()),'cfo_period_max':str(cfo.period_q.max()),
    'cfo_next_nonmissing':int(base.cfo_next.notna().sum()),
    'missing_rate':{c:float(base[c].isna().mean()) for c in ['cfo_assets','cfo_next','log_assets','leverage','THREATS_z','ACTS_z','event5_THREATS_z','event5_ACTS_z']},
    'groups':{}
}
for grp in GROUPS:
    x=pd.to_numeric(base[grp],errors='coerce')
    qa['groups'][grp]={'classified_symbols':int(base.loc[x.isin([0,1]),'symbol'].nunique()),'treated_symbols':int(base.loc[x==1,'symbol'].nunique()),'control_symbols':int(base.loc[x==0,'symbol'].nunique()),'missing_symbols':int(base.loc[x.isna(),'symbol'].nunique())}

# Winsor helper for robustness only; main model remains raw after plausibility cleaning inherited from CFO panel.
def wins(s):
    lo,hi=s.quantile([.01,.99]); return s.clip(lo,hi)

def fit_model(data, shock, grp, event=False, winsor=False, reverse=False):
    mr=('event5_'+shock) if event else shock
    cols=['symbol','period_q','pidx','cfo_assets','cfo_next','log_assets','leverage',mr,grp]
    x=data[cols].copy(); x[grp]=pd.to_numeric(x[grp],errors='coerce'); x=x[x[grp].isin([0,1])]
    if reverse:
        x=x.sort_values(['symbol','pidx']); x['mr_use']=x.groupby('symbol')[mr].shift(-1); y='cfo_assets'
    else:
        x['mr_use']=x[mr]; y='cfo_next'
    x['mr_group']=x['mr_use']*x[grp]
    x=x.replace([np.inf,-np.inf],np.nan).dropna(subset=[y,'cfo_assets','log_assets','leverage','mr_use','mr_group'])
    if winsor:
        for c in [y,'cfo_assets','log_assets','leverage','mr_use','mr_group']: x[c]=wins(x[c])
    if len(x)<200 or x.symbol.nunique()<20 or x.period_q.nunique()<6: return None
    x=x.set_index(['symbol','period_q']).sort_index()
    X=x[['mr_use','mr_group','cfo_assets','log_assets','leverage']]
    mod=PanelOLS(x[y],X,entity_effects=True,time_effects=True,drop_absorbed=True)
    r=mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    b0=float(r.params.get('mr_use',np.nan)); b1=float(r.params.get('mr_group',np.nan));
    cov=r.cov
    total=b0+b1
    var=float(cov.loc['mr_use','mr_use']+cov.loc['mr_group','mr_group']+2*cov.loc['mr_use','mr_group'])
    se_total=np.sqrt(var) if var>=0 else np.nan
    p_total=float(2*norm.sf(abs(total/se_total))) if np.isfinite(se_total) and se_total>0 else np.nan
    return {'n':int(len(x)),'firms':int(x.index.get_level_values(0).nunique()),'quarters':int(x.index.get_level_values(1).nunique()),
            'beta_control_mr':b0,'p_control_mr':float(r.pvalues.get('mr_use',np.nan)),
            'beta_interaction':b1,'p_interaction':float(r.pvalues.get('mr_group',np.nan)),
            'beta_treated_total':total,'se_treated_total':se_total,'p_treated_total':p_total,
            'r2_within':float(r.rsquared_within)}

rows=[]
for name,grp,shock in DIRECTIONS:
    for spec,event,winsor,reverse in [('primary',False,False,False),('winsor_1_99',False,True,False),('event_car5',True,False,False),('reverse_timing_placebo',False,False,True)]:
        z=fit_model(base,shock,grp,event=event,winsor=winsor,reverse=reverse)
        rec={'direction':name,'group':grp,'shock':shock,'spec':spec}
        if z: rec.update(z)
        else: rec['status']='INSUFFICIENT'
        rows.append(rec)
res=pd.DataFrame(rows)
# BH-FDR across the six pre-specified primary differential tests (5a/5b both retained; no winner chosen post hoc).
res['q_fdr_interaction']=np.nan
m=(res.spec=='primary')&res.p_interaction.notna()
if m.any():
    pv=res.loc[m,'p_interaction'].to_numpy(); order=np.argsort(pv); n=len(pv); qv=np.empty(n); run=1.0
    for j in range(n-1,-1,-1):
        idx=order[j]; run=min(run,pv[idx]*n/(j+1)); qv[idx]=min(1.0,run)
    res.loc[m,'q_fdr_interaction']=qv
res.to_csv(OUT/'market_future_cfo_results.csv',index=False)
(OUT/'qa_summary.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
summary={'interpretation':'Predictive/associational linkage only. Positive treated-total beta means a more negative GPR-linked abnormal return is associated with lower CFO next quarter. Causal market-feedback/self-fulfilling language is not permitted without additional identification.',
         'qa':qa,'results':res.replace({np.nan:None}).to_dict(orient='records')}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))