from __future__ import annotations
import json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import t as tdist, chi2

START='2020-01-01'; END='2025-12-31'
OUT=Path('data/gpr_market_top5'); OUT.mkdir(parents=True,exist_ok=True)
G=pd.read_csv('data/research_cohorts/vci578_behavioral_groups.csv')
GROUPS=['HighLiab_TopTercile','CashConversionTrap','DoubleExposure_TopTercile','FragileLowMarginFunding','AssetCommitmentMismatch']
G['symbol']=G.symbol.astype(str).str.upper().str.strip()
for c in GROUPS:G[c]=pd.to_numeric(G[c],errors='coerce').fillna(0).astype(int)
SYMBOLS=sorted(G.symbol.unique())
FROM=int(datetime(2020,1,1,tzinfo=timezone.utc).timestamp()); TO=int(datetime(2026,1,1,tzinfo=timezone.utc).timestamp())
URL='https://dchart-api.vndirect.com.vn/dchart/history'

def fetch(sym):
    last=None
    for k in range(4):
        try:
            r=requests.get(URL,params={'resolution':'D','symbol':sym,'from':FROM,'to':TO},timeout=30,headers={'User-Agent':'Mozilla/5.0 academic-research/1.0'})
            r.raise_for_status(); j=r.json()
            if not isinstance(j,dict) or not j.get('t') or not j.get('c'):return sym,None,f'EMPTY:{str(j)[:100]}'
            n=min(len(j['t']),len(j['c'])); v=j.get('v',[np.nan]*n)
            if len(v)<n:v=list(v)+[np.nan]*(n-len(v))
            z=pd.DataFrame({'symbol':sym,'date':pd.to_datetime(j['t'][:n],unit='s',utc=True).tz_convert(None),'close':pd.to_numeric(j['c'][:n],errors='coerce'),'volume':pd.to_numeric(v[:n],errors='coerce')})
            z=z.dropna(subset=['date','close']).sort_values('date'); z=z[(z.date>=START)&(z.date<=END)]
            return (sym,z,None) if len(z) else (sym,None,'EMPTY_FILTER')
        except Exception as e:
            last=e;time.sleep(min(5,1+k))
    return sym,None,f'{type(last).__name__}:{last}'

frames=[];errors=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    fs={ex.submit(fetch,s):s for s in SYMBOLS+['VNINDEX']}
    for i,f in enumerate(as_completed(fs),1):
        s,z,e=f.result()
        if z is not None:frames.append(z)
        else:errors.append({'symbol':s,'error':e})
        if i%50==0:print('dchart',i,'/',len(fs),'ok',len(frames),'err',len(errors),flush=True)
pd.DataFrame(errors).to_csv(OUT/'price_errors_dchart.csv',index=False)
if not frames:raise RuntimeError('No DChart prices')
allpx=pd.concat(frames,ignore_index=True).sort_values(['symbol','date'])
allpx['ret']=allpx.groupby('symbol').close.pct_change()
idx=allpx[allpx.symbol=='VNINDEX'][['date','ret']].rename(columns={'ret':'mkt_ret'}).dropna()
px=allpx[allpx.symbol!='VNINDEX'].merge(idx,on='date',how='left')
# fallback only on dates where VNINDEX unavailable
fallback=allpx[allpx.symbol!='VNINDEX'].groupby('date').ret.median().rename('fallback_mkt')
px=px.merge(fallback,on='date',how='left');px['mkt_ret']=px.mkt_ret.fillna(px.fallback_mkt);px['ar']=px.ret-px.mkt_ret
px['logvol']=np.log1p(px.volume);px['logvol_med60']=px.groupby('symbol').logvol.transform(lambda s:s.shift(1).rolling(60,min_periods=20).median());px['abn_logvol']=px.logvol-px.logvol_med60
px.to_parquet(OUT/'price_daily_2020_2025.parquet',index=False,compression='zstd')

# GPR daily
u='https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv'; rr=requests.get(u,timeout=60,headers={'User-Agent':'Mozilla/5.0'});rr.raise_for_status();(OUT/'ai_gpr_data_daily.csv').write_bytes(rr.content);g=pd.read_csv(OUT/'ai_gpr_data_daily.csv')
def pick(cols,pats):
    norm={re.sub(r'[^a-z0-9]','',str(c).lower()):c for c in cols}
    for p in pats:
        pp=re.sub(r'[^a-z0-9]','',p.lower())
        for n,c in norm.items():
            if n==pp or pp in n:return c
    return None
dc=pick(g.columns,['date','day']); ai=pick(g.columns,['GPR_AI','AI_GPR','aigpr']); orig=pick(g.columns,['GPR_AER','original']); thr=pick(g.columns,['GPR_THREATS','threats','gprt']); act=pick(g.columns,['GPR_ACTS','acts','gpra'])
if not all([dc,ai,thr,act]):raise RuntimeError(f'GPR schema unresolved {list(g.columns)}')
ren={dc:'date',ai:'AI_GPR',thr:'THREATS',act:'ACTS'}
if orig:ren[orig]='GPR_ORIG'
g=g.rename(columns=ren);g['date']=pd.to_datetime(g.date,errors='coerce');g=g[(g.date>=pd.Timestamp(START)-pd.Timedelta(days=10))&(g.date<=END)].sort_values('date')
SHOCKS=['AI_GPR','THREATS','ACTS']+(['GPR_ORIG'] if 'GPR_ORIG' in g else [])
for c in SHOCKS:g[c]=pd.to_numeric(g[c],errors='coerce');g[c+'_innov']=np.log1p(g[c].clip(lower=0)).diff()
trading=pd.DatetimeIndex(sorted(px.date.dropna().unique()));maps=[]
for _,r in g.dropna(subset=['date']).iterrows():
    j=trading.searchsorted(r.date)
    if j>=len(trading):continue
    x={'trade_date':trading[j]}
    for c in SHOCKS:x[c+'_innov']=r[c+'_innov'];x[c+'_level']=r[c]
    maps.append(x)
mg=pd.DataFrame(maps);agg={c+'_innov':'sum' for c in SHOCKS}|{c+'_level':'last' for c in SHOCKS};mg=mg.groupby('trade_date',as_index=False).agg(agg).sort_values('trade_date')
for c in SHOCKS:
    s=mg[c+'_innov'].replace([np.inf,-np.inf],np.nan);sd=s.std(ddof=0);mg[c+'_z']=(s-s.mean())/sd if np.isfinite(sd) and sd>0 else np.nan
mg.to_csv(OUT/'gpr_mapped_trading_daily.csv',index=False)

p=px.merge(G[['symbol']+GROUPS],on='symbol',how='inner'); parts=[]
for grp in GROUPS:
    q=p.dropna(subset=['ar']).groupby(['date',grp]).ar.mean().unstack()
    if 0 in q.columns and 1 in q.columns:
        s=(q[1]-q[0]).rename('spread').reset_index();s['group']=grp;parts.append(s)
spreads=pd.concat(parts,ignore_index=True);spreads.to_csv(OUT/'daily_group_spreads.csv',index=False)
regrows=[]
for grp in GROUPS:
    base=spreads[spreads.group==grp][['date','spread']].merge(mg.rename(columns={'trade_date':'date'}),on='date',how='inner')
    for shock in [c+'_z' for c in SHOCKS]:
        z=base[['spread',shock]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)<60:regrows.append({'group':grp,'shock':shock,'beta':np.nan,'p':np.nan,'n':len(z)});continue
        fit=sm.OLS(z.spread,sm.add_constant(z[[shock]],has_constant='add')).fit(cov_type='HAC',cov_kwds={'maxlags':5});regrows.append({'group':grp,'shock':shock,'beta':fit.params[shock],'p':fit.pvalues[shock],'n':len(z)})
reg=pd.DataFrame(regrows);reg.to_csv(OUT/'group_spread_regressions.csv',index=False)

events=[]
for _,r in mg.dropna(subset=['THREATS_z']).sort_values('THREATS_z',ascending=False).iterrows():
    d=pd.Timestamp(r.trade_date)
    if all(abs((d-e).days)>=10 for e in events):events.append(d)
    if len(events)>=20:break
pd.DataFrame({'event_trade_date':events}).to_csv(OUT/'top20_threat_event_dates.csv',index=False)
by={s:z.set_index('date').sort_index() for s,z in p.groupby('symbol')};gl=G.set_index('symbol')[GROUPS];er=[]
for ed in events:
    ix=trading.get_indexer([ed])[0]
    if ix<0:continue
    pre=trading[max(0,ix-5):ix]
    for sym,z in by.items():
        r={'symbol':sym,'event_trade_date':ed};valid=False
        for h in [1,5,20]:
            win=trading[ix:min(len(trading),ix+h)];car=z.reindex(win).ar.sum(min_count=1);r[f'car_{h}']=car;valid=valid or np.isfinite(car)
        r['pre_car_5']=z.reindex(pre).ar.sum(min_count=1) if len(pre) else np.nan
        win5=trading[ix:min(len(trading),ix+5)];av=z.reindex(win5).abn_logvol;r['abn_volume_5']=av.mean() if av.notna().any() else np.nan
        r['belief_pressure_5']=max(0,-r['car_5'])*max(0,r['abn_volume_5']) if np.isfinite(r['car_5']) and np.isfinite(r['abn_volume_5']) else np.nan
        if sym in gl.index:
            for c in GROUPS:r[c]=int(gl.loc[sym,c])
        if valid:er.append(r)
ef=pd.DataFrame(er);ef.to_csv(OUT/'event_firm_metrics.csv',index=False)
evrows=[]
for grp in GROUPS:
    for metric in ['pre_car_5','car_1','car_5','car_20','abn_volume_5','belief_pressure_5']:
        for ed,z in ef.dropna(subset=[grp,metric]).groupby('event_trade_date'):
            a=z.loc[z[grp]==1,metric];b=z.loc[z[grp]==0,metric]
            if len(a)<8 or len(b)<20:continue
            va,vb=a.var(ddof=1),b.var(ddof=1);diff=a.mean()-b.mean();se=np.sqrt(va/len(a)+vb/len(b));pv=np.nan
            if np.isfinite(se) and se>0:
                t=diff/se;den=(va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1);df=(va/len(a)+vb/len(b))**2/den if den>0 else np.nan;pv=2*tdist.sf(abs(t),df) if np.isfinite(df) else np.nan
            evrows.append({'group':grp,'metric':metric,'event_trade_date':ed,'treated_mean':a.mean(),'control_mean':b.mean(),'diff':diff,'p':pv,'n_t':len(a),'n_c':len(b)})
ev=pd.DataFrame(evrows);ev.to_csv(OUT/'event_group_metrics.csv',index=False)
pool=[]
for (grp,metric),z in ev.groupby(['group','metric']):
    pp=z.p.dropna().clip(1e-12,1);f=float(chi2.sf(-2*np.log(pp).sum(),2*len(pp))) if len(pp) else np.nan;pool.append({'group':grp,'metric':metric,'mean_diff':z['diff'].mean(),'median_diff':z['diff'].median(),'events':len(z),'share_negative':(z['diff']<0).mean(),'fisher_p':f})
pool=pd.DataFrame(pool);pool.to_csv(OUT/'event_group_summary.csv',index=False)
C=[('D1_Threat_HighLiab','HighLiab_TopTercile','THREATS_z'),('D2_Acts_HighLiab','HighLiab_TopTercile','ACTS_z'),('D3_Acts_CashConversionTrap','CashConversionTrap','ACTS_z'),('D4_Threat_DoubleExposure','DoubleExposure_TopTercile','THREATS_z'),('D5_Threat_FragileLowMargin','FragileLowMarginFunding','THREATS_z')]
rank=[]
for name,grp,shock in C:
    rr=reg[(reg.group==grp)&(reg.shock==shock)]
    if rr.empty:continue
    rr=rr.iloc[0]
    def get(m,c):
        z=pool[(pool.group==grp)&(pool.metric==m)];return float(z.iloc[0][c]) if len(z) else np.nan
    rank.append({'direction':name,'group':grp,'shock':shock,'beta_daily_spread':rr.beta,'p_hac':rr.p,'n_daily':rr.n,'event5_mean_diff':get('car_5','mean_diff'),'event5_fisher_p':get('car_5','fisher_p'),'event5_share_negative':get('car_5','share_negative'),'pre5_mean_diff':get('pre_car_5','mean_diff'),'pre5_fisher_p':get('pre_car_5','fisher_p'),'attention_diff':get('abn_volume_5','mean_diff'),'attention_fisher_p':get('abn_volume_5','fisher_p'),'belief_pressure_diff':get('belief_pressure_5','mean_diff'),'belief_pressure_fisher_p':get('belief_pressure_5','fisher_p')})
rank=pd.DataFrame(rank);rank.to_csv(OUT/'D1_D5_market_screen.csv',index=False)
summary={'price_source':'VNDirect DChart','market_benchmark':'VNINDEX with median fallback only if missing','symbols_target':len(SYMBOLS),'symbols_price_ok':int(px.symbol.nunique()),'price_rows':len(px),'vnindex_rows':len(idx),'events_n':len(events),'directions':rank.replace({np.nan:None}).to_dict(orient='records')};(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
