from __future__ import annotations
import time, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from vnstock_data import Market

START="2020-01-01"
END="2025-12-31"
OUT=Path("data/gpr_market_top5")
OUT.mkdir(parents=True,exist_ok=True)

exp=pd.read_csv("data/research_cohorts/vci578_preexposure.csv")
symbols=exp["symbol"].dropna().astype(str).str.upper().drop_duplicates().tolist()

def fetch_one(sym):
    last=None
    for k in range(3):
        try:
            df=Market().equity(sym).ohlcv(start=START,end=END,interval="1D")
            if df is None or df.empty:
                return sym,None,"EMPTY"
            df=df.copy()
            tc=next((c for c in ["time","date","trading_date"] if c in df.columns),None)
            cc=next((c for c in ["close","Close","close_price"] if c in df.columns),None)
            if tc is None or cc is None:
                return sym,None,f"schema={list(df.columns)}"
            z=df[[tc,cc]].rename(columns={tc:"date",cc:"close"})
            z["date"]=pd.to_datetime(z["date"],errors="coerce")
            z["close"]=pd.to_numeric(z["close"],errors="coerce")
            z=z.dropna().sort_values("date")
            z["symbol"]=sym
            return sym,z,None
        except Exception as e:
            last=e; time.sleep(1.5*(k+1))
    return sym,None,f"{type(last).__name__}:{last}"

frames=[]; errors=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    futs={pool.submit(fetch_one,s):s for s in symbols}
    for i,f in enumerate(as_completed(futs),1):
        sym,z,e=f.result()
        if z is not None: frames.append(z)
        else: errors.append({"symbol":sym,"error":e})
        if i%50==0: print("done",i,"/",len(symbols),flush=True)
        time.sleep(0.15)

pd.DataFrame(errors).to_csv(OUT/"price_errors.csv",index=False)
if not frames:
    raise RuntimeError("No price data")
px=pd.concat(frames,ignore_index=True)
px=px.sort_values(["symbol","date"])
px["ret"]=px.groupby("symbol")["close"].pct_change()
mkt=px.groupby("date")["ret"].median().rename("mkt_ret")
px=px.merge(mkt,on="date",how="left")
px["ar"]=px["ret"]-px["mkt_ret"]

url="https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv"
r=requests.get(url,timeout=60); r.raise_for_status()
(OUT/"ai_gpr_data_daily.csv").write_bytes(r.content)
g=pd.read_csv(OUT/"ai_gpr_data_daily.csv")

def pick(cols, pats):
    norm={re.sub(r"[^a-z0-9]","",c.lower()):c for c in cols}
    for p in pats:
        pp=re.sub(r"[^a-z0-9]","",p.lower())
        for n,c in norm.items():
            if n==pp or pp in n:
                return c
    return None

dc=pick(g.columns,["date","day"])
ai=pick(g.columns,["GPR_AI","AI_GPR"])
orig=pick(g.columns,["GPR_AER","GPR Original","original"])
thr=pick(g.columns,["GPR_THREATS","Threats GPR","threats"])
act=pick(g.columns,["GPR_ACTS","Acts GPR","acts"])
if not all([dc,ai,thr,act]):
    raise RuntimeError(f"GPR schema unresolved: {list(g.columns)}")
g=g.rename(columns={dc:"date",ai:"AI_GPR",thr:"THREATS",act:"ACTS", **({orig:"GPR_ORIG"} if orig else {})})
g["date"]=pd.to_datetime(g["date"],errors="coerce")
g=g[(g.date>=START)&(g.date<=END)].copy()
for c in ["AI_GPR","THREATS","ACTS"] + (["GPR_ORIG"] if "GPR_ORIG" in g else []):
    g[c]=pd.to_numeric(g[c],errors="coerce")
    g[c+"_innov"]=np.log1p(g[c].clip(lower=0)).diff()
    s=g[c+"_innov"]
    g[c+"_z"]=(s-s.mean())/s.std(ddof=0)

groups=["double_exposure","shock_amplifier","physical_chain","rollover_assetlight","receivable_heavy","trade_credit_chain"]
p=px.merge(exp[["symbol"]+groups+["liab_exp","wc_exp"]],on="symbol",how="inner")
spread_rows=[]
for grp in groups:
    d=p.dropna(subset=["ar",grp])
    tmp=d.groupby(["date",grp])["ar"].mean().unstack()
    if 0 in tmp.columns and 1 in tmp.columns:
        s=(tmp[1]-tmp[0]).rename("spread").reset_index()
        s["group"]=grp
        spread_rows.append(s)
spreads=pd.concat(spread_rows,ignore_index=True)

rows=[]
for grp in groups:
    s=spreads[spreads.group==grp].merge(g[["date","THREATS_z","ACTS_z","AI_GPR_z"] + (["GPR_ORIG_z"] if "GPR_ORIG_z" in g else [])],on="date",how="inner").dropna()
    for shock in ["THREATS_z","ACTS_z","AI_GPR_z"] + (["GPR_ORIG_z"] if "GPR_ORIG_z" in s else []):
        X=sm.add_constant(s[[shock]])
        fit=sm.OLS(s["spread"],X).fit(cov_type="HAC",cov_kwds={"maxlags":5})
        rows.append({"group":grp,"shock":shock,"beta":fit.params[shock],"p":fit.pvalues[shock],"n":len(s)})
reg=pd.DataFrame(rows)
reg.to_csv(OUT/"group_spread_regressions.csv",index=False)

gg=g.dropna(subset=["THREATS_z"]).sort_values("THREATS_z",ascending=False)
events=[]
for _,r0 in gg.iterrows():
    d=r0["date"]
    if all(abs((d-e).days)>=10 for e in events):
        events.append(d)
    if len(events)>=15: break
pd.DataFrame({"event_date":events}).to_csv(OUT/"top15_threat_event_dates.csv",index=False)

trading=pd.DatetimeIndex(sorted(px.date.dropna().unique()))
event_rows=[]
for ed in events:
    idx=trading.searchsorted(ed)
    if idx>=len(trading): continue
    d0=trading[idx]
    for h in [1,5,20]:
        endidx=min(idx+h-1,len(trading)-1)
        window=trading[idx:endidx+1]
        z=p[p.date.isin(window)].copy()
        car=z.groupby("symbol")["ar"].sum().rename("car").reset_index().merge(exp[["symbol"]+groups],on="symbol")
        for grp in groups:
            a=car[car[grp]==1]["car"]; b=car[car[grp]==0]["car"]
            if len(a)>5 and len(b)>5:
                diff=a.mean()-b.mean()
                se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
                t=diff/se if se>0 else np.nan
                from scipy.stats import t as tdist
                dfw=(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))**2 / ((a.var(ddof=1)/len(a))**2/(len(a)-1)+(b.var(ddof=1)/len(b))**2/(len(b)-1))
                pv=2*tdist.sf(abs(t),dfw) if np.isfinite(t) else np.nan
                event_rows.append({"event_date":ed,"trade_date":d0,"h":h,"group":grp,"treated_car":a.mean(),"control_car":b.mean(),"diff":diff,"p":pv,"n_t":len(a),"n_c":len(b)})
erc=pd.DataFrame(event_rows)
erc.to_csv(OUT/"event_car_by_group.csv",index=False)

pool=[]
for (grp,h),z in erc.groupby(["group","h"]):
    pool.append({"group":grp,"h":h,"mean_diff":z["diff"].mean(),"median_diff":z["diff"].median(),"events":len(z),"share_negative":(z["diff"]<0).mean(),"geo_mean_p":float(np.exp(np.log(z["p"].clip(lower=1e-12)).mean()))})
pool=pd.DataFrame(pool)
pool.to_csv(OUT/"event_car_summary.csv",index=False)

cands=[
("Threat × shock-amplifier -> AR","shock_amplifier","THREATS_z"),
("Threat × double-exposure -> AR","double_exposure","THREATS_z"),
("Threat × physical-chain -> AR","physical_chain","THREATS_z"),
("Acts × shock-amplifier -> AR","shock_amplifier","ACTS_z"),
("AI-GPR × shock-amplifier -> AR","shock_amplifier","AI_GPR_z"),
]
rank=[]
for name,grp,shock in cands:
    rr=reg[(reg.group==grp)&(reg.shock==shock)]
    if rr.empty: continue
    rr=rr.iloc[0]
    es=pool[(pool.group==grp)&(pool.h==5)]
    md=float(es.mean_diff.iloc[0]) if not es.empty else np.nan
    neg=float(es.share_negative.iloc[0]) if not es.empty else np.nan
    score=0
    if rr.beta<0: score+=35
    if rr.p<0.01: score+=30
    elif rr.p<0.05: score+=22
    elif rr.p<0.10: score+=12
    if np.isfinite(md) and md<0: score+=20
    if np.isfinite(neg): score+=15*neg
    rank.append({"model":name,"group":grp,"shock":shock,"beta_daily_spread":rr.beta,"p_hac":rr.p,"event5_mean_diff":md,"event5_share_negative":neg,"empirical_score":score})
rank=pd.DataFrame(rank).sort_values("empirical_score",ascending=False)
rank.to_csv(OUT/"TOP5_market_ranking.csv",index=False)

summary={"symbols_target":len(symbols),"symbols_price_ok":int(px.symbol.nunique()),"price_rows":int(len(px)),"gpr_cols":{"ai":ai,"original":orig,"threats":thr,"acts":act},"top5":rank.to_dict(orient="records")}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
