from __future__ import annotations
import time, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import t as tdist
from vnstock_data import Market

START='2020-01-01'; END='2025-12-31'
OUT=Path('data/gpr_market_top5_v2'); OUT.mkdir(parents=True, exist_ok=True)
exp=pd.read_csv('data/research_cohorts/vci578_preexposure.csv')
bg=pd.read_csv('data/research_cohorts/vci578_behavioral_groups.csv')
exp['symbol']=exp['symbol'].astype(str).str.upper(); bg['symbol']=bg['symbol'].astype(str).str.upper()
base=exp.merge(bg,on='symbol',how='left')
symbols=base.symbol.dropna().drop_duplicates().tolist()

def fetch_one(sym):
    last=None
    for k in range(3):
        try:
            df=Market().equity(sym).ohlcv(start=START,end=END,interval='1D')
            if df is None or df.empty: return sym,None,'EMPTY'
            tc=next((c for c in ['time','date','trading_date'] if c in df.columns),None)
            cc=next((c for c in ['close','Close','close_price'] if c in df.columns),None)
            if tc is None or cc is None: return sym,None,f'schema={list(df.columns)}'
            z=df[[tc,cc]].rename(columns={tc:'date',cc:'close'}).copy()
            z['date']=pd.to_datetime(z['date'],errors='coerce'); z['close']=pd.to_numeric(z['close'],errors='coerce')
            z=z.dropna().sort_values('date'); z['symbol']=sym
            return sym,z,None
        except Exception as e:
            last=e; time.sleep(1.2*(k+1))
    return sym,None,f'{type(last).__name__}:{last}'

frames=[]; errors=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    futs={pool.submit(fetch_one,s):s for s in symbols}
    for i,f in enumerate(as_completed(futs),1):
        sym,z,e=f.result()
        if z is not None: frames.append(z)
        else: errors.append({'symbol':sym,'error':e})
        if i%50==0: print('done',i,'/',len(symbols),flush=True)
pd.DataFrame(errors).to_csv(OUT/'price_errors.csv',index=False)
if not frames: raise RuntimeError('No price data')
px=pd.concat(frames,ignore_index=True).sort_values(['symbol','date'])
px['ret']=px.groupby('symbol')['close'].pct_change()
mkt=px.groupby('date')['ret'].median().rename('mkt_ret')
px=px.merge(mkt,on='date',how='left'); px['ar']=px['ret']-px['mkt_ret']
px[['symbol','date','close','ret','mkt_ret','ar']].to_csv(OUT/'daily_price_panel.csv.gz',index=False,compression='gzip')

url='https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv'
r=requests.get(url,timeout=60); r.raise_for_status(); (OUT/'ai_gpr_data_daily.csv').write_bytes(r.content)
g=pd.read_csv(OUT/'ai_gpr_data_daily.csv')
def pick(cols,pats):
    norm={re.sub(r'[^a-z0-9]','',c.lower()):c for c in cols}
    for p in pats:
        pp=re.sub(r'[^a-z0-9]','',p.lower())
        for n,c in norm.items():
            if n==pp or pp in n: return c
    return None

dc=pick(g.columns,['date','day']); ai=pick(g.columns,['GPR_AI','AI_GPR']); orig=pick(g.columns,['GPR_AER','original']); thr=pick(g.columns,['GPR_THREATS','threats']); act=pick(g.columns,['GPR_ACTS','acts'])
if not all([dc,ai,thr,act]): raise RuntimeError(f'GPR schema unresolved: {list(g.columns)}')
ren={dc:'date',ai:'AI_GPR',thr:'THREATS',act:'ACTS'}
if orig: ren[orig]='GPR_ORIG'
g=g.rename(columns=ren); g['date']=pd.to_datetime(g['date'],errors='coerce'); g=g[(g.date>=START)&(g.date<=END)].sort_values('date').copy()
shock_cols=['AI_GPR','THREATS','ACTS']+(['GPR_ORIG'] if 'GPR_ORIG' in g.columns else [])
for c in shock_cols:
    g[c]=pd.to_numeric(g[c],errors='coerce'); innov=np.log1p(g[c].clip(lower=0)).diff(); g[c+'_z']=(innov-innov.mean())/innov.std(ddof=0)

# Five research directions/groupings requested by the thesis screen
GROUPS=['HighLiab_TopTercile','CashConversionTrap','DoubleExposure_TopTercile','FragileLowMarginFunding','AssetCommitmentMismatch']
p=px.merge(base[['symbol']+GROUPS],on='symbol',how='inner')

# daily treated-minus-control abnormal-return spreads
spread_rows=[]
for grp in GROUPS:
    d=p.dropna(subset=['ar',grp]).copy(); d[grp]=pd.to_numeric(d[grp],errors='coerce')
    tmp=d.groupby(['date',grp])['ar'].mean().unstack()
    if 0 in tmp.columns and 1 in tmp.columns:
        s=(tmp[1]-tmp[0]).rename('spread').reset_index(); s['group']=grp; spread_rows.append(s)
spreads=pd.concat(spread_rows,ignore_index=True) if spread_rows else pd.DataFrame(columns=['date','spread','group'])

rows=[]
for grp in GROUPS:
    ss=spreads[spreads.group==grp]
    for shock in ['THREATS_z','ACTS_z','AI_GPR_z']+(['GPR_ORIG_z'] if 'GPR_ORIG_z' in g.columns else []):
        s=ss.merge(g[['date',shock]],on='date',how='inner').replace([np.inf,-np.inf],np.nan).dropna(subset=['spread',shock])
        if len(s)<30 or s[shock].nunique()<3:
            rows.append({'group':grp,'shock':shock,'beta':np.nan,'p':np.nan,'n':len(s),'status':'INSUFFICIENT'})
            continue
        X=sm.add_constant(s[[shock]],has_constant='add'); fit=sm.OLS(s['spread'],X).fit(cov_type='HAC',cov_kwds={'maxlags':5})
        rows.append({'group':grp,'shock':shock,'beta':float(fit.params[shock]),'p':float(fit.pvalues[shock]),'n':len(s),'status':'OK'})
reg=pd.DataFrame(rows); reg.to_csv(OUT/'group_spread_regressions.csv',index=False)

# Top 15 threat-innovation dates, separated by >=10 calendar days
gg=g.dropna(subset=['THREATS_z']).sort_values('THREATS_z',ascending=False); events=[]
for _,r0 in gg.iterrows():
    d=r0['date']
    if all(abs((d-e).days)>=10 for e in events): events.append(d)
    if len(events)>=15: break
pd.DataFrame({'event_date':events}).to_csv(OUT/'top15_threat_event_dates.csv',index=False)

trading=pd.DatetimeIndex(sorted(px.date.dropna().unique())); event_rows=[]
for ed in events:
    idx=trading.searchsorted(ed)
    if idx>=len(trading): continue
    d0=trading[idx]
    for h in [1,5,20]:
        window=trading[idx:min(idx+h,len(trading))]
        z=p[p.date.isin(window)].groupby('symbol',as_index=False)['ar'].sum().rename(columns={'ar':'car'}).merge(base[['symbol']+GROUPS],on='symbol',how='left')
        for grp in GROUPS:
            a=z.loc[z[grp]==1,'car'].dropna(); b=z.loc[z[grp]==0,'car'].dropna()
            if len(a)<=5 or len(b)<=5: continue
            va=a.var(ddof=1); vb=b.var(ddof=1); diff=a.mean()-b.mean(); se=np.sqrt(va/len(a)+vb/len(b))
            if not np.isfinite(se) or se<=0: pv=np.nan
            else:
                tt=diff/se; den=(va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1); dfw=((va/len(a)+vb/len(b))**2/den) if den>0 else np.nan; pv=2*tdist.sf(abs(tt),dfw) if np.isfinite(dfw) else np.nan
            event_rows.append({'event_date':ed,'trade_date':d0,'h':h,'group':grp,'treated_car':a.mean(),'control_car':b.mean(),'diff':diff,'p':pv,'n_t':len(a),'n_c':len(b)})
erc=pd.DataFrame(event_rows); erc.to_csv(OUT/'event_car_by_group.csv',index=False)

pool=[]
if not erc.empty:
    for (grp,h),z in erc.groupby(['group','h']):
        pool.append({'group':grp,'h':h,'mean_diff':z['diff'].mean(),'median_diff':z['diff'].median(),'events':len(z),'share_negative':(z['diff']<0).mean(),'mean_p':z['p'].mean()})
pool=pd.DataFrame(pool); pool.to_csv(OUT/'event_car_summary.csv',index=False)

# Compare directions 1-3 first; 4-5 remain in same output for final table
DIRECTIONS=[
 ('1 Threat→HighLiab market anticipation','HighLiab_TopTercile','THREATS_z'),
 ('2 Acts→HighLiab realization','HighLiab_TopTercile','ACTS_z'),
 ('3 Acts→CashConversionTrap','CashConversionTrap','ACTS_z'),
 ('4 Threat→DoubleExposure','DoubleExposure_TopTercile','THREATS_z'),
 ('5 Threat→AssetCommitmentMismatch','AssetCommitmentMismatch','THREATS_z'),
]
rank=[]
for name,grp,shock in DIRECTIONS:
    rr=reg[(reg.group==grp)&(reg.shock==shock)&(reg.status=='OK')]
    if rr.empty: rank.append({'direction':name,'group':grp,'shock':shock,'beta_daily_spread':np.nan,'p_hac':np.nan,'event5_mean_diff':np.nan,'event5_share_negative':np.nan,'empirical_score':0}); continue
    rr=rr.iloc[0]; es=pool[(pool.group==grp)&(pool.h==5)] if not pool.empty else pd.DataFrame()
    md=float(es.mean_diff.iloc[0]) if not es.empty else np.nan; neg=float(es.share_negative.iloc[0]) if not es.empty else np.nan
    score=0
    if rr.beta<0: score+=30
    if rr.p<0.01: score+=30
    elif rr.p<0.05: score+=22
    elif rr.p<0.10: score+=12
    if np.isfinite(md) and md<0: score+=25
    if np.isfinite(neg): score+=15*neg
    rank.append({'direction':name,'group':grp,'shock':shock,'beta_daily_spread':rr.beta,'p_hac':rr.p,'event5_mean_diff':md,'event5_share_negative':neg,'empirical_score':score})
rank=pd.DataFrame(rank).sort_values(['empirical_score','p_hac'],ascending=[False,True]); rank.to_csv(OUT/'TOP5_market_ranking.csv',index=False)
summary={'symbols_target':len(symbols),'symbols_price_ok':int(px.symbol.nunique()),'price_rows':int(len(px)),'errors':len(errors),'gpr_cols':{'ai':ai,'original':orig,'threats':thr,'acts':act},'top5':rank.replace({np.nan:None}).to_dict(orient='records')}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
