from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

MKT=Path('data/gpr_market_top5_v4')
CFO=Path('output/xy_screen/vci42_analysis_panel_2020_2025.csv')
GROUP=Path('data/research_cohorts/vci578_behavioral_groups.csv')
OUT=Path('data/gpr_threat_doubleexposure_validation_v1'); OUT.mkdir(parents=True,exist_ok=True)

px=pd.read_csv(MKT/'daily_price_panel.csv.gz',parse_dates=['date'])
px['symbol']=px.symbol.astype(str).str.upper().str.strip(); px['date']=pd.to_datetime(px.date).dt.normalize()
groups=pd.read_csv(GROUP); groups['symbol']=groups.symbol.astype(str).str.upper().str.strip()
if 'DoubleExposure_TopTercile' not in groups.columns: raise RuntimeError('DoubleExposure_TopTercile missing')

ev=pd.read_csv(MKT/'top_event_dates.csv',parse_dates=['event_date'])
ev['event_date']=pd.to_datetime(ev.event_date).dt.normalize()
# Restrict to Threat events only using the exact shock label produced by the locked market pipeline.
th=ev[ev['shock'].astype(str).str.contains('THREAT',case=False,na=False)].copy().reset_index(drop=True)
if th.empty: raise RuntimeError(f'No Threat events found; shock labels={ev.shock.unique().tolist()}')
th['event_id']=np.arange(len(th))

trading=pd.DatetimeIndex(sorted(px.date.dropna().unique()))
rows=[]
for _,r in th.iterrows():
    i=trading.searchsorted(r.event_date)
    if i>=len(trading): continue
    td=trading[i]
    specs={
        'car1': trading[i:min(i+1,len(trading))],
        'car5': trading[i:min(i+5,len(trading))],
        'car20': trading[i:min(i+20,len(trading))],
        'pre5': trading[max(0,i-5):i],
    }
    for spec,win in specs.items():
        if len(win)==0: continue
        z=px[px.date.isin(win)].groupby('symbol',as_index=False)['ar'].sum().rename(columns={'ar':'car'})
        z['spec']=spec; z['event_id']=int(r.event_id); z['event_date']=r.event_date; z['period_q']=str(pd.Timestamp(td).to_period('Q'))
        rows.append(z)
evfirm=pd.concat(rows,ignore_index=True)

# CFO panel and next-quarter outcome.
cfo=pd.read_csv(CFO); cfo['symbol']=cfo.symbol.astype(str).str.upper().str.strip(); cfo['period_q']=cfo.period_q.astype(str)
cfo=cfo[(cfo.year>=2020)&(cfo.year<=2025)].copy(); cfo['pidx']=cfo.year.astype(int)*4+cfo.q.astype(int); cfo=cfo.sort_values(['symbol','pidx'])
cfo['cfo_lag']=cfo.groupby('symbol')['cfo_assets'].shift(1)
nexty=cfo[['symbol','pidx','cfo_assets']].rename(columns={'pidx':'next_pidx','cfo_assets':'cfo_next'}); cfo['next_pidx']=cfo.pidx+1; cfo=cfo.merge(nexty,on=['symbol','next_pidx'],how='left')
base=cfo[['symbol','period_q','pidx','cfo_lag','cfo_assets','cfo_next','log_assets','leverage']].merge(groups[['symbol','DoubleExposure_TopTercile']],on='symbol',how='inner')
base['DoubleExposure_TopTercile']=pd.to_numeric(base['DoubleExposure_TopTercile'],errors='coerce')

# Aggregate each event-window measure to symbol-quarter exactly as the original linkage did.
def make_q(spec,drop_event=None):
    z=evfirm[evfirm.spec.eq(spec)].copy()
    if drop_event is not None: z=z[z.event_id.ne(drop_event)]
    q=z.groupby(['symbol','period_q'],as_index=False)['car'].mean().rename(columns={'car':'mr'})
    return base.merge(q,on=['symbol','period_q'],how='left')

def fit(data,reverse=False):
    x=data[['symbol','period_q','pidx','cfo_lag','cfo_assets','cfo_next','log_assets','leverage','DoubleExposure_TopTercile','mr']].copy()
    x=x[x.DoubleExposure_TopTercile.isin([0,1])].sort_values(['symbol','pidx'])
    if reverse:
        x['mr_use']=x.groupby('symbol')['mr'].shift(-1); y='cfo_assets'; baseline='cfo_lag'
    else:
        x['mr_use']=x['mr']; y='cfo_next'; baseline='cfo_assets'
    x['mr_group']=x.mr_use*x.DoubleExposure_TopTercile
    req=[y,baseline,'log_assets','leverage','mr_use','mr_group']; x=x.replace([np.inf,-np.inf],np.nan).dropna(subset=req)
    if len(x)<200 or x.symbol.nunique()<20 or x.pidx.nunique()<6: return None
    x=x.set_index(['symbol','pidx']).sort_index()
    r=PanelOLS(x[y],x[['mr_use','mr_group',baseline,'log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    return {'n':int(len(x)),'firms':int(x.index.get_level_values(0).nunique()),'quarters':int(x.index.get_level_values(1).nunique()),'beta_interaction':float(r.params['mr_group']),'se_interaction':float(r.std_errors['mr_group']),'p_interaction':float(r.pvalues['mr_group']),'r2_within':float(r.rsquared_within)}

out=[]
for spec in ['car1','car5','car20','pre5']:
    z=fit(make_q(spec)); rec={'test':spec,'type':'event_window'}; rec.update(z or {'status':'INSUFFICIENT'}); out.append(rec)
# Reverse-timing placebo for the focal CAR5 measure.
z=fit(make_q('car5'),reverse=True); rec={'test':'reverse_timing_car5','type':'placebo'}; rec.update(z or {'status':'INSUFFICIENT'}); out.append(rec)
# Leave-one-Threat-event-out robustness for CAR5.
for eid in sorted(th.event_id.unique()):
    z=fit(make_q('car5',drop_event=int(eid))); rec={'test':f'car5_drop_event_{int(eid)}','type':'leave_one_event_out','dropped_event':int(eid)}; rec.update(z or {'status':'INSUFFICIENT'}); out.append(rec)
res=pd.DataFrame(out)
# BH-FDR only across the three prespecified post-event windows; placebos and LOEO are diagnostic gates.
res['q_fdr_post_windows']=np.nan
m=res.test.isin(['car1','car5','car20']) & res.p_interaction.notna()
if m.any():
    pv=res.loc[m,'p_interaction'].to_numpy(); order=np.argsort(pv); n=len(pv); q=np.empty(n); run=1.0
    for j in range(n-1,-1,-1):
        k=order[j]; run=min(run,pv[k]*n/(j+1)); q[k]=min(1.0,run)
    res.loc[m,'q_fdr_post_windows']=q
res.to_csv(OUT/'results.csv',index=False)

qa={
    'price_symbols':int(px.symbol.nunique()),'price_start':str(px.date.min().date()),'price_end':str(px.date.max().date()),
    'threat_events':int(len(th)),'event_dates':[str(x.date()) for x in th.event_date],
    'cfo_symbols':int(cfo.symbol.nunique()),'group_symbols':int(groups.symbol.nunique()),
    'analysis_group_classified_symbols':int(base.loc[base.DoubleExposure_TopTercile.isin([0,1]),'symbol'].nunique()),
    'treated_symbols':int(base.loc[base.DoubleExposure_TopTercile.eq(1),'symbol'].nunique()),
    'control_symbols':int(base.loc[base.DoubleExposure_TopTercile.eq(0),'symbol'].nunique()),
    'duplicate_symbol_quarter_cfo':int(base.duplicated(['symbol','period_q']).sum()),
    'cfo_period_min':str(base.period_q.min()),'cfo_period_max':str(base.period_q.max()),
    'missing_core':{c:float(base[c].isna().mean()) for c in ['cfo_assets','cfo_next','log_assets','leverage','DoubleExposure_TopTercile']}
}
(OUT/'qa_summary.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')

get=lambda name: res.loc[res.test.eq(name)].iloc[0] if res.test.eq(name).any() else None
car5=get('car5'); pre5=get('pre5'); rev=get('reverse_timing_car5'); lo=res[res.type.eq('leave_one_event_out')]
lo_negative=bool((lo.beta_interaction<0).all()) if len(lo) and lo.beta_interaction.notna().all() else False
# Prespecified upgrade gate: focal CAR5 negative and FDR<=5%; pre5 and reverse placebo p>10%; all LOEO signs negative.
gate=bool(car5 is not None and car5.beta_interaction<0 and car5.q_fdr_post_windows<=0.05 and pre5 is not None and pre5.p_interaction>0.10 and rev is not None and rev.p_interaction>0.10 and lo_negative)
assessment={'direction':'Threat x DoubleExposure validation v1','upgrade_gate_pass':gate,'focal_car5':None if car5 is None else car5.replace({np.nan:None}).to_dict(),'pre5_placebo':None if pre5 is None else pre5.replace({np.nan:None}).to_dict(),'reverse_timing_placebo':None if rev is None else rev.replace({np.nan:None}).to_dict(),'loeo_all_negative':lo_negative,'decision_rule':'Upgrade only if CAR5 interaction is negative with BH-FDR q<=0.05 across CAR1/CAR5/CAR20; pre5 placebo p>0.10; reverse-timing CAR5 placebo p>0.10; and every leave-one-Threat-event-out CAR5 interaction retains a negative sign.','causal_language':'Predictive/event-specific association only; do not call market reaction causal without an exogenous identification strategy.'}
(OUT/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'qa':qa,'assessment':assessment,'results':res.replace({np.nan:None}).to_dict('records')},ensure_ascii=False,indent=2))
