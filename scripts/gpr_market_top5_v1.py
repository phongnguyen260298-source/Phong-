from __future__ import annotations
import time, json, re, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import t as tdist
from vnstock_data import Market

START='2020-01-01'
END='2025-12-31'
OUT=Path('data/gpr_market_top5')
OUT.mkdir(parents=True, exist_ok=True)
GROUP_FILE='data/research_cohorts/vci578_behavioral_groups.csv'

groups_df=pd.read_csv(GROUP_FILE)
group_cols=['HighLiab_TopTercile','CashConversionTrap','DoubleExposure_TopTercile','FragileLowMarginFunding','AssetCommitmentMismatch']
for c in group_cols:
    groups_df[c]=pd.to_numeric(groups_df[c], errors='coerce').fillna(0).astype(int)
symbols=groups_df['symbol'].dropna().astype(str).str.upper().drop_duplicates().tolist()

# ---------- price retrieval ----------
def fetch_one(sym):
    last=None
    for k in range(4):
        try:
            df=Market().equity(sym).ohlcv(start=START,end=END,interval='1D')
            if df is None or df.empty:
                return sym,None,'EMPTY'
            df=df.copy()
            tc=next((c for c in ['time','date','trading_date'] if c in df.columns),None)
            cc=next((c for c in ['close','Close','close_price'] if c in df.columns),None)
            vc=next((c for c in ['volume','Volume','match_volume','total_volume'] if c in df.columns),None)
            if tc is None or cc is None:
                return sym,None,f'schema={list(df.columns)}'
            keep=[tc,cc]+([vc] if vc else [])
            z=df[keep].copy().rename(columns={tc:'date',cc:'close', **({vc:'volume'} if vc else {})})
            z['date']=pd.to_datetime(z['date'],errors='coerce').dt.tz_localize(None)
            z['close']=pd.to_numeric(z['close'],errors='coerce')
            if 'volume' in z:
                z['volume']=pd.to_numeric(z['volume'],errors='coerce')
            else:
                z['volume']=np.nan
            z=z.dropna(subset=['date','close']).sort_values('date')
            z=z[(z.date>=pd.Timestamp(START))&(z.date<=pd.Timestamp(END))]
            if z.empty: return sym,None,'EMPTY_AFTER_FILTER'
            z['symbol']=sym
            return sym,z[['symbol','date','close','volume']],None
        except BaseException as e:
            last=e
            time.sleep(min(8,1.5*(k+1)))
    return sym,None,f'{type(last).__name__}:{last}'

frames=[]; errors=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    futs={pool.submit(fetch_one,s):s for s in symbols}
    for i,f in enumerate(as_completed(futs),1):
        sym,z,e=f.result()
        if z is not None: frames.append(z)
        else: errors.append({'symbol':sym,'error':e})
        if i%50==0: print('done',i,'/',len(symbols),flush=True)
        time.sleep(0.08)

pd.DataFrame(errors).to_csv(OUT/'price_errors.csv',index=False)
if not frames:
    raise RuntimeError('No price data')
px=pd.concat(frames,ignore_index=True).sort_values(['symbol','date'])
px['ret']=px.groupby('symbol')['close'].pct_change()
mkt=px.groupby('date')['ret'].median().rename('mkt_ret')
px=px.merge(mkt,on='date',how='left')
px['ar']=px['ret']-px['mkt_ret']
px['logvol']=np.log1p(px['volume'])
px['logvol_med60']=px.groupby('symbol')['logvol'].transform(lambda s:s.shift(1).rolling(60,min_periods=20).median())
px['abn_logvol']=px['logvol']-px['logvol_med60']
px[['symbol','date','close','volume','ret','mkt_ret','ar','abn_logvol']].to_parquet(OUT/'price_daily_2020_2025.parquet',index=False,compression='zstd')

# ---------- AI-GPR daily ----------
url='https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv'
r=requests.get(url,timeout=90); r.raise_for_status()
(OUT/'ai_gpr_data_daily.csv').write_bytes(r.content)
g=pd.read_csv(OUT/'ai_gpr_data_daily.csv')

def pick(cols,pats):
    norm={re.sub(r'[^a-z0-9]','',str(c).lower()):c for c in cols}
    for p in pats:
        pp=re.sub(r'[^a-z0-9]','',p.lower())
        for n,c in norm.items():
            if n==pp or pp in n:
                return c
    return None

dc=pick(g.columns,['date','day','time'])
ai=pick(g.columns,['GPR_AI','AI_GPR','aigpr'])
orig=pick(g.columns,['GPR_AER','GPR Original','original'])
thr=pick(g.columns,['GPR_THREATS','Threats GPR','threats','gprt'])
act=pick(g.columns,['GPR_ACTS','Acts GPR','acts','gpra'])
(OUT/'gpr_schema.json').write_text(json.dumps({'columns':list(map(str,g.columns)),'date':str(dc),'ai':str(ai),'original':str(orig),'threats':str(thr),'acts':str(act)},indent=2),encoding='utf-8')
if not all([dc,ai,thr,act]):
    raise RuntimeError(f'GPR schema unresolved: {list(g.columns)}')
g=g.rename(columns={dc:'date',ai:'AI_GPR',thr:'THREATS',act:'ACTS', **({orig:'GPR_ORIG'} if orig else {})})
g['date']=pd.to_datetime(g['date'],errors='coerce').dt.tz_localize(None)
g=g[(g.date>=pd.Timestamp(START)-pd.Timedelta(days=10))&(g.date<=pd.Timestamp(END))].sort_values('date').copy()
shock_names=['AI_GPR','THREATS','ACTS']+(['GPR_ORIG'] if 'GPR_ORIG' in g else [])
for c in shock_names:
    g[c]=pd.to_numeric(g[c],errors='coerce')
    g[c+'_innov']=np.log1p(g[c].clip(lower=0)).diff()

trading=pd.DatetimeIndex(sorted(px['date'].dropna().unique()))
map_rows=[]
for _,rr in g.dropna(subset=['date']).iterrows():
    j=trading.searchsorted(rr['date'])
    if j>=len(trading):
        continue
    td=trading[j]
    row={'trade_date':td}
    for c in shock_names:
        row[c+'_innov']=rr.get(c+'_innov',np.nan)
        row[c+'_level']=rr.get(c,np.nan)
    map_rows.append(row)
mg=pd.DataFrame(map_rows)
agg={}
for c in shock_names:
    agg[c+'_innov']='sum'
    agg[c+'_level']='last'
mg=mg.groupby('trade_date',as_index=False).agg(agg).sort_values('trade_date')
for c in shock_names:
    s=mg[c+'_innov'].replace([np.inf,-np.inf],np.nan)
    sd=s.std(ddof=0)
    mg[c+'_z']=(s-s.mean())/sd if np.isfinite(sd) and sd>0 else np.nan
mg.to_csv(OUT/'gpr_mapped_trading_daily.csv',index=False)

# ---------- group daily spread regressions ----------
p=px.merge(groups_df[['symbol']+group_cols],on='symbol',how='inner')
spread_rows=[]
for grp in group_cols:
    d=p.dropna(subset=['ar',grp])
    tmp=d.groupby(['date',grp])['ar'].mean().unstack()
    if 0 in tmp.columns and 1 in tmp.columns:
        ss=(tmp[1]-tmp[0]).rename('spread').reset_index()
        ss['group']=grp
        spread_rows.append(ss)
spreads=pd.concat(spread_rows,ignore_index=True) if spread_rows else pd.DataFrame(columns=['date','spread','group'])
spreads.to_csv(OUT/'daily_group_spreads.csv',index=False)

reg_rows=[]
shock_cols=[c+'_z' for c in shock_names]
for grp in group_cols:
    ss=spreads[spreads.group==grp][['date','spread']].merge(mg.rename(columns={'trade_date':'date'})[['date']+shock_cols],on='date',how='inner')
    for shock in shock_cols:
        z=ss[['spread',shock]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)<60 or z[shock].std(ddof=0)==0:
            reg_rows.append({'group':grp,'shock':shock,'beta':np.nan,'p':np.nan,'n':len(z),'status':'INSUFFICIENT'})
            continue
        X=sm.add_constant(z[[shock]],has_constant='add')
        fit=sm.OLS(z['spread'],X).fit(cov_type='HAC',cov_kwds={'maxlags':5})
        reg_rows.append({'group':grp,'shock':shock,'beta':float(fit.params[shock]),'p':float(fit.pvalues[shock]),'n':len(z),'status':'OK'})
reg=pd.DataFrame(reg_rows)
reg.to_csv(OUT/'group_spread_regressions.csv',index=False)

# ---------- event dates ----------
thcol='THREATS_z'
gg=mg.dropna(subset=[thcol]).sort_values(thcol,ascending=False)
events=[]
for _,r0 in gg.iterrows():
    d=pd.Timestamp(r0['trade_date'])
    if all(abs((d-e).days)>=10 for e in events):
        events.append(d)
    if len(events)>=20: break
pd.DataFrame({'event_trade_date':events}).to_csv(OUT/'top20_threat_event_dates.csv',index=False)

# ---------- firm-level event metrics ----------
event_firm=[]
for ed in events:
    idx=trading.get_indexer([ed])[0]
    if idx<0: continue
    pre_win=trading[max(0,idx-5):idx]
    for sym,zs in p.groupby('symbol'):
        zs=zs.set_index('date').sort_index()
        row={'symbol':sym,'event_trade_date':ed}
        ok=False
        for h in [1,5,20]:
            win=trading[idx:min(len(trading),idx+h)]
            zz=zs.reindex(win)
            car=zz['ar'].sum(min_count=1)
            row[f'car_{h}']=car
            if np.isfinite(car): ok=True
        pre=zs.reindex(pre_win)['ar'].sum(min_count=1) if len(pre_win) else np.nan
        row['pre_car_5']=pre
        win5=trading[idx:min(len(trading),idx+5)]
        vv=zs.reindex(win5)['abn_logvol']
        row['abn_volume_5']=vv.mean() if vv.notna().any() else np.nan
        row['belief_pressure_5']=max(0.0,-row.get('car_5',np.nan))*max(0.0,row['abn_volume_5']) if np.isfinite(row.get('car_5',np.nan)) and np.isfinite(row['abn_volume_5']) else np.nan
        for c in group_cols:
            vals=groups_df.loc[groups_df.symbol.eq(sym),c]
            row[c]=int(vals.iloc[0]) if len(vals) else 0
        if ok: event_firm.append(row)
ef=pd.DataFrame(event_firm)
ef.to_csv(OUT/'event_firm_metrics.csv',index=False)

summ=[]
for grp in group_cols:
    for metric in ['pre_car_5','car_1','car_5','car_20','abn_volume_5','belief_pressure_5']:
        z=ef.dropna(subset=[grp,metric])
        for ed,ze in z.groupby('event_trade_date'):
            a=ze.loc[ze[grp]==1,metric]; b=ze.loc[ze[grp]==0,metric]
            if len(a)<8 or len(b)<20: continue
            va=a.var(ddof=1); vb=b.var(ddof=1)
            se=np.sqrt(va/len(a)+vb/len(b)) if np.isfinite(va) and np.isfinite(vb) else np.nan
            diff=a.mean()-b.mean()
            if np.isfinite(se) and se>0:
                tt=diff/se
                denom=((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
                dfw=(va/len(a)+vb/len(b))**2/denom if denom>0 else np.nan
                pv=2*tdist.sf(abs(tt),dfw) if np.isfinite(dfw) else np.nan
            else: pv=np.nan
            summ.append({'group':grp,'metric':metric,'event_trade_date':ed,'treated_mean':a.mean(),'control_mean':b.mean(),'diff':diff,'p':pv,'n_t':len(a),'n_c':len(b)})
event_group=pd.DataFrame(summ)
event_group.to_csv(OUT/'event_group_metrics.csv',index=False)

pool=[]
for (grp,metric),z in event_group.groupby(['group','metric']):
    pool.append({'group':grp,'metric':metric,'mean_diff':z['diff'].mean(),'median_diff':z['diff'].median(),'events':len(z),'share_negative':(z['diff']<0).mean(),'fisher_p':float(1.0)})
from scipy.stats import chi2
for row in pool:
    z=event_group[(event_group.group==row['group'])&(event_group.metric==row['metric'])]['p'].dropna().clip(lower=1e-12,upper=1)
    if len(z):
        stat=-2*np.log(z).sum(); row['fisher_p']=float(chi2.sf(stat,2*len(z)))
pool=pd.DataFrame(pool)
pool.to_csv(OUT/'event_group_summary.csv',index=False)

cands=[
    ('D1_Threat_HighLiab','HighLiab_TopTercile','THREATS_z'),
    ('D2_Acts_HighLiab','HighLiab_TopTercile','ACTS_z'),
    ('D3_Acts_CashConversionTrap','CashConversionTrap','ACTS_z'),
    ('D4_Threat_DoubleExposure','DoubleExposure_TopTercile','THREATS_z'),
    ('D5_Threat_FragileLowMargin','FragileLowMarginFunding','THREATS_z'),
]
rank=[]
for name,grp,shock in cands:
    rr=reg[(reg.group==grp)&(reg.shock==shock)]
    if rr.empty: continue
    rr=rr.iloc[0]
    e5=pool[(pool.group==grp)&(pool.metric=='car_5')]
    pre=pool[(pool.group==grp)&(pool.metric=='pre_car_5')]
    att=pool[(pool.group==grp)&(pool.metric=='belief_pressure_5')]
    rank.append({
        'direction':name,'group':grp,'shock':shock,
        'beta_daily_spread':rr.beta,'p_hac':rr.p,'n_daily':rr.n,
        'event5_mean_diff':float(e5.mean_diff.iloc[0]) if len(e5) else np.nan,
        'event5_fisher_p':float(e5.fisher_p.iloc[0]) if len(e5) else np.nan,
        'event5_share_negative':float(e5.share_negative.iloc[0]) if len(e5) else np.nan,
        'pre5_mean_diff':float(pre.mean_diff.iloc[0]) if len(pre) else np.nan,
        'pre5_fisher_p':float(pre.fisher_p.iloc[0]) if len(pre) else np.nan,
        'belief_pressure_diff':float(att.mean_diff.iloc[0]) if len(att) else np.nan,
        'belief_pressure_fisher_p':float(att.fisher_p.iloc[0]) if len(att) else np.nan,
    })
rank=pd.DataFrame(rank)
rank.to_csv(OUT/'D1_D5_market_screen.csv',index=False)

summary={
    'symbols_target':len(symbols),'symbols_price_ok':int(px.symbol.nunique()),'price_rows':int(len(px)),
    'gpr_schema':{'ai':ai,'original':orig,'threats':thr,'acts':act},
    'events_n':len(events),'directions':rank.replace({np.nan:None}).to_dict(orient='records')
}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
