from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import t as tdist

START='2020-01-01'; END='2025-12-31'
OUT=Path('data/gpr_market_top5_v4'); OUT.mkdir(parents=True, exist_ok=True)
SHARDS=OUT/'shards'
exp=pd.read_csv('data/research_cohorts/vci578_preexposure.csv')
bg=pd.read_csv('data/research_cohorts/vci578_behavioral_groups.csv')
exp['symbol']=exp['symbol'].astype(str).str.upper(); bg['symbol']=bg['symbol'].astype(str).str.upper()
base=exp.merge(bg,on='symbol',how='left')
files=sorted(SHARDS.glob('price_shard_*.csv.gz'))
if not files: raise RuntimeError('No shard price files')
px=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
px['symbol']=px['symbol'].astype(str).str.upper(); px['date']=pd.to_datetime(px['date'],errors='coerce').dt.normalize(); px['close']=pd.to_numeric(px['close'],errors='coerce')
raw_rows=len(px); dup_rows=int(px.duplicated(['symbol','date']).sum())
px=px.dropna(subset=['symbol','date','close']).drop_duplicates(['symbol','date']).sort_values(['symbol','date'])
px['ret']=px.groupby('symbol')['close'].pct_change()
mkt=px.groupby('date')['ret'].median().rename('mkt_ret')
px=px.merge(mkt,on='date',how='left'); px['ar']=px['ret']-px['mkt_ret']
px[['symbol','date','close','ret','mkt_ret','ar']].to_csv(OUT/'daily_price_panel.csv.gz',index=False,compression='gzip')
coverage_sym=px.groupby('symbol').agg(price_n=('date','size'),price_start=('date','min'),price_end=('date','max')).reset_index()
coverage_sym.to_csv(OUT/'price_coverage_by_symbol.csv',index=False)

url='https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv'
r=requests.get(url,timeout=60); r.raise_for_status(); (OUT/'ai_gpr_data_daily.csv').write_bytes(r.content)
g=pd.read_csv(OUT/'ai_gpr_data_daily.csv')
def pick(cols,pats):
    norm={re.sub(r'[^a-z0-9]','',c.lower()):c for c in cols}
    for p0 in pats:
        p=re.sub(r'[^a-z0-9]','',p0.lower())
        for n,c in norm.items():
            if n==p or p in n: return c
    return None

dc=pick(g.columns,['date','day']); ai=pick(g.columns,['GPR_AI','AI_GPR']); orig=pick(g.columns,['GPR_AER','original']); thr=pick(g.columns,['THREATS_GPR_AI','GPR_THREATS','threats']); act=pick(g.columns,['ACTS_GPR_AI','GPR_ACTS','acts'])
if not all([dc,ai,thr,act]): raise RuntimeError(f'GPR schema unresolved: {list(g.columns)}')
ren={dc:'date',ai:'AI_GPR',thr:'THREATS',act:'ACTS'}
if orig: ren[orig]='GPR_ORIG'
g=g.rename(columns=ren); g['date']=pd.to_datetime(g['date'],errors='coerce').dt.normalize(); g=g[(g.date>=START)&(g.date<=END)].sort_values('date').copy()
shock_cols=['AI_GPR','THREATS','ACTS']+(['GPR_ORIG'] if 'GPR_ORIG' in g.columns else [])
for c in shock_cols:
    g[c]=pd.to_numeric(g[c],errors='coerce'); innov=np.log1p(g[c].clip(lower=0)).diff(); g[c+'_z']=(innov-innov.mean())/innov.std(ddof=0)

GROUPS=['HighLiab_TopTercile','CashConversionTrap','DoubleExposure_TopTercile','FragileLowMarginFunding','AssetCommitmentMismatch']
p=px.merge(base[['symbol']+GROUPS],on='symbol',how='inner')
coverage=[]
for grp in GROUPS:
    valid=base[['symbol',grp]].dropna().copy(); valid[grp]=pd.to_numeric(valid[grp],errors='coerce')
    cov=p[['symbol']].drop_duplicates().merge(valid,on='symbol',how='left')
    coverage.append({'group':grp,'firms_base':int(base.symbol.nunique()),'firms_price':int(p.symbol.nunique()),'treated_price':int(cov.loc[cov[grp]==1,'symbol'].nunique()),'control_price':int(cov.loc[cov[grp]==0,'symbol'].nunique())})
pd.DataFrame(coverage).to_csv(OUT/'group_coverage.csv',index=False)

# Daily treated-minus-control abnormal-return spread.
spread_rows=[]
for grp in GROUPS:
    d=p.dropna(subset=['ar',grp]).copy(); d[grp]=pd.to_numeric(d[grp],errors='coerce')
    tmp=d.groupby(['date',grp])['ar'].mean().unstack()
    if 0 in tmp.columns and 1 in tmp.columns:
        s=(tmp[1]-tmp[0]).rename('spread').reset_index(); s['group']=grp; spread_rows.append(s)
spreads=pd.concat(spread_rows,ignore_index=True) if spread_rows else pd.DataFrame(columns=['date','spread','group'])

rows=[]
shock_z=['THREATS_z','ACTS_z','AI_GPR_z']+(['GPR_ORIG_z'] if 'GPR_ORIG_z' in g.columns else [])
for grp in GROUPS:
    ss=spreads[spreads.group==grp]
    for shock in shock_z:
        s=ss.merge(g[['date',shock]],on='date',how='inner').replace([np.inf,-np.inf],np.nan).dropna(subset=['spread',shock])
        if len(s)<30 or s[shock].nunique()<3:
            rows.append({'group':grp,'shock':shock,'beta':np.nan,'p':np.nan,'n':len(s),'status':'INSUFFICIENT'}); continue
        X=sm.add_constant(s[[shock]],has_constant='add'); fit=sm.OLS(s['spread'],X).fit(cov_type='HAC',cov_kwds={'maxlags':5})
        rows.append({'group':grp,'shock':shock,'beta':float(fit.params[shock]),'p':float(fit.pvalues[shock]),'n':len(s),'status':'OK'})
reg=pd.DataFrame(rows)
# Benjamini-Hochberg FDR across all valid group×shock daily screens.
reg['q_fdr']=np.nan
ok=reg['p'].notna()
if ok.any():
    pv=reg.loc[ok,'p'].values; order=np.argsort(pv); m=len(pv); q=np.empty(m); running=1.0
    for rank_idx in range(m-1,-1,-1):
        j=order[rank_idx]; val=pv[j]*m/(rank_idx+1); running=min(running,val); q[j]=min(1.0,running)
    reg.loc[ok,'q_fdr']=q
reg.to_csv(OUT/'group_spread_regressions.csv',index=False)

# Top shock innovation event dates, separated by >=10 calendar days, for Threats and Acts separately.
def top_events(shock,n=15):
    gg=g.dropna(subset=[shock]).sort_values(shock,ascending=False); out=[]
    for _,r0 in gg.iterrows():
        d=r0['date']
        if all(abs((d-e).days)>=10 for e in out): out.append(d)
        if len(out)>=n: break
    return out
event_sets={'THREATS_z':top_events('THREATS_z'),'ACTS_z':top_events('ACTS_z')}
pd.DataFrame([(k,d) for k,v in event_sets.items() for d in v],columns=['shock','event_date']).to_csv(OUT/'top_event_dates.csv',index=False)

trading=pd.DatetimeIndex(sorted(px.date.dropna().unique())); event_rows=[]
for shock,events in event_sets.items():
    for ed in events:
        idx=trading.searchsorted(ed)
        if idx>=len(trading): continue
        d0=trading[idx]
        windows={'post1':trading[idx:min(idx+1,len(trading))], 'post5':trading[idx:min(idx+5,len(trading))], 'post20':trading[idx:min(idx+20,len(trading))], 'pre5':trading[max(0,idx-5):idx]}
        for horizon,window in windows.items():
            if len(window)==0: continue
            z=p[p.date.isin(window)].groupby('symbol',as_index=False)['ar'].sum().rename(columns={'ar':'car'}).merge(base[['symbol']+GROUPS],on='symbol',how='left')
            for grp in GROUPS:
                a=z.loc[z[grp]==1,'car'].dropna(); b=z.loc[z[grp]==0,'car'].dropna()
                if len(a)<=5 or len(b)<=5: continue
                va=a.var(ddof=1); vb=b.var(ddof=1); diff=a.mean()-b.mean(); se=np.sqrt(va/len(a)+vb/len(b))
                if not np.isfinite(se) or se<=0: pv=np.nan
                else:
                    tt=diff/se; den=(va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1); dfw=((va/len(a)+vb/len(b))**2/den) if den>0 else np.nan; pv=2*tdist.sf(abs(tt),dfw) if np.isfinite(dfw) else np.nan
                event_rows.append({'shock':shock,'event_date':ed,'trade_date':d0,'horizon':horizon,'group':grp,'treated_car':a.mean(),'control_car':b.mean(),'diff':diff,'p':pv,'n_t':len(a),'n_c':len(b)})
erc=pd.DataFrame(event_rows); erc.to_csv(OUT/'event_car_by_group.csv',index=False)
pool=[]
if not erc.empty:
    for (shock,grp,horizon),z in erc.groupby(['shock','group','horizon']):
        pool.append({'shock':shock,'group':grp,'horizon':horizon,'mean_diff':z['diff'].mean(),'median_diff':z['diff'].median(),'events':len(z),'share_negative':(z['diff']<0).mean(),'share_event_p_lt_05':(z['p']<.05).mean()})
pool=pd.DataFrame(pool); pool.to_csv(OUT/'event_car_summary.csv',index=False)

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
    if rr.empty:
        rank.append({'direction':name,'group':grp,'shock':shock,'beta_daily_spread':np.nan,'p_hac':np.nan,'q_fdr':np.nan,'event5_mean_diff':np.nan,'event5_share_negative':np.nan,'pre5_mean_diff':np.nan,'empirical_score':0}); continue
    rr=rr.iloc[0]
    es=pool[(pool.group==grp)&(pool.shock==shock)&(pool.horizon=='post5')] if not pool.empty else pd.DataFrame()
    pre=pool[(pool.group==grp)&(pool.shock==shock)&(pool.horizon=='pre5')] if not pool.empty else pd.DataFrame()
    md=float(es.mean_diff.iloc[0]) if not es.empty else np.nan; neg=float(es.share_negative.iloc[0]) if not es.empty else np.nan; premd=float(pre.mean_diff.iloc[0]) if not pre.empty else np.nan
    score=0
    if rr.beta<0: score+=25
    if rr.p<0.01 and rr.q_fdr<0.05: score+=25
    elif rr.p<0.05 and rr.q_fdr<0.10: score+=18
    elif rr.p<0.10: score+=10
    if np.isfinite(md) and md<0: score+=20
    if np.isfinite(neg): score+=15*neg
    if np.isfinite(premd) and abs(premd)<abs(md): score+=15
    rank.append({'direction':name,'group':grp,'shock':shock,'beta_daily_spread':rr.beta,'p_hac':rr.p,'q_fdr':rr.q_fdr,'event5_mean_diff':md,'event5_share_negative':neg,'pre5_mean_diff':premd,'empirical_score':score})
rank=pd.DataFrame(rank).sort_values(['empirical_score','p_hac'],ascending=[False,True]); rank.to_csv(OUT/'TOP5_market_ranking.csv',index=False)
err_files=sorted(SHARDS.glob('errors_shard_*.csv')); errn=0
for f in err_files:
    try:
        x=pd.read_csv(f); errn+=len(x)
    except: pass
summary={'price_shards':len(files),'symbols_target':int(base.symbol.nunique()),'symbols_price_ok':int(px.symbol.nunique()),'price_rows_raw':int(raw_rows),'price_rows_unique':int(len(px)),'duplicate_symbol_date_rows':dup_rows,'price_start':str(px.date.min().date()),'price_end':str(px.date.max().date()),'median_bars_per_symbol':float(coverage_sym.price_n.median()),'errors':errn,'gpr_start':str(g.date.min().date()),'gpr_end':str(g.date.max().date()),'gpr_cols':{'ai':ai,'original':orig,'threats':thr,'acts':act},'top5':rank.replace({np.nan:None}).to_dict(orient='records')}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
