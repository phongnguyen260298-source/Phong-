"""Integrated reproducibility script v07 for GPR -> OCF.

Synchronization-only release after verified v06 WCB results.
Main sample: NON_FN = NonFin_CompleteCase == 1.
Conventional inference: two-way firm x quarter cluster covariance with component
G/(G-1) corrections; p-values use Student-t with df = Gq - 1.
WCB: restricted wild cluster bootstrap-t by quarter, Rademacher weights,
seed=20260915. Default B=9999.

Default input is the verified v06 workbook. Override explicitly with --file PATH.
"""
from __future__ import annotations
import argparse, json, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy, statsmodels, statsmodels.api as sm
from scipy.linalg import lstsq
from scipy.stats import t as student_t
from statsmodels.stats.sandwich_covariance import cov_cluster

DEFAULT_FILE = Path("GPR_OCF_2026-09-16_v06.xlsx")
MAIN_SAMPLE = "NON_FN"
WCB_SEED = 20260915
WCB_B = 9999

def load_panel(file): return pd.read_excel(file, sheet_name="04_PANEL_PROCESSED")
def select_sample(panel, sample=MAIN_SAMPLE):
    if sample == "NON_FN": return panel.loc[panel["NonFin_CompleteCase"].eq(1)].copy()
    if sample == "FULL": return panel.loc[panel["Full_CompleteCase"].eq(1)].copy()
    raise ValueError("sample must be NON_FN or FULL")
def prep(panel, shock, moderator="HL67", y="cfo_assets", sample=MAIN_SAMPLE, exclude_year=None, exclude_2022h1=False, seasonality=False):
    df=select_sample(panel,sample); cols=["symbol","period_q","year","quarter_num",y,shock,moderator,"log_assets","leverage"]
    x=df[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if exclude_year is not None: x=x.loc[x["year"].ne(exclude_year)].copy()
    if exclude_2022h1: x=x.loc[~(x["year"].eq(2022)&x["quarter_num"].isin([1,2]))].copy()
    x["interaction"]=x[shock]*x[moderator]
    if seasonality:
        for q in (2,3,4): x[f"HL_Q{q}"]=x[moderator]*x["quarter_num"].eq(q).astype(float)
    return x
def _cluster_piece(res,group):
    g=pd.Categorical(group).codes; G=np.unique(g).size
    if G<=1: raise ValueError("Need at least two clusters")
    return cov_cluster(res,g,use_correction=False)*(G/(G-1))
def fit_core(panel,shock,moderator="HL67",y="cfo_assets",sample=MAIN_SAMPLE,controls=True,exclude_year=None,exclude_2022h1=False,seasonality=False):
    x=prep(panel,shock,moderator,y,sample,exclude_year,exclude_2022h1,seasonality); slopes=["interaction"]+(["log_assets","leverage"] if controls else [])
    if seasonality: slopes += ["HL_Q2","HL_Q3","HL_Q4"]
    firm_d=pd.get_dummies(x["symbol"],drop_first=True,dtype=float); qtr_d=pd.get_dummies(x["period_q"],drop_first=True,dtype=float)
    X=pd.concat([x[slopes].reset_index(drop=True),firm_d.reset_index(drop=True),qtr_d.reset_index(drop=True)],axis=1); X=sm.add_constant(X,has_constant="add")
    res=sm.OLS(x[y].to_numpy(),X.to_numpy()).fit(); j=list(X.columns).index("interaction")
    Vf=_cluster_piece(res,x["symbol"]); Vq=_cluster_piece(res,x["period_q"]); inter=pd.Series(x["symbol"].astype(str).to_numpy()+"|"+x["period_q"].astype(str).to_numpy()); Vi=_cluster_piece(res,inter); V=Vf+Vq-Vi
    beta=float(res.params[j]); var=float(V[j,j]); se=np.sqrt(var) if var>0 else np.nan; Gq=int(x["period_q"].nunique()); df_t=Gq-1
    p=float(2*student_t.sf(abs(beta/se),df=df_t)) if np.isfinite(se) else np.nan
    return dict(beta=beta,se=float(se) if np.isfinite(se) else np.nan,p=p,df_t=df_t,n=len(x),firms=int(x.symbol.nunique()),quarters=Gq)
def _wcb_design(panel,shock,moderator,sample):
    x=prep(panel,shock,moderator=moderator,sample=sample); firm=pd.get_dummies(x["symbol"],drop_first=True,dtype=float); qtr=pd.get_dummies(x["period_q"],drop_first=True,dtype=float)
    Z=np.column_stack([np.ones(len(x)),x[["log_assets","leverage"]].to_numpy(float),firm.to_numpy(),qtr.to_numpy()]); y=x["cfo_assets"].to_numpy(float); d=x["interaction"].to_numpy(float)
    coef,_,_,_=lstsq(Z,np.column_stack([y,d]),lapack_driver="gelsy"); R=np.column_stack([y,d])-Z@coef; yr,xr=R[:,0],R[:,1]; denom=float(xr@xr); beta=float((xr@yr)/denom); u=yr-xr*beta
    levels=sorted(x["period_q"].unique()); g=pd.Categorical(x["period_q"],categories=levels).codes; G=len(levels); scores=np.bincount(g,weights=xr*u,minlength=G); se=np.sqrt(np.sum(scores*scores))/denom; t_obs=float(beta/se)
    sliced=np.column_stack([yr*(g==k) for k in range(G)]); cr,_,_,_=lstsq(Z,sliced,lapack_driver="gelsy"); V=sliced-Z@cr
    return x,xr,denom,beta,t_obs,g,V,G
def wild_cluster_p(panel,shock,moderator="HL67",sample=MAIN_SAMPLE,B=WCB_B,seed=WCB_SEED):
    x,xr,denom,beta,t_obs,g,V,G=_wcb_design(panel,shock,moderator,sample); rng=np.random.default_rng(seed); exceed=0
    for _ in range(B):
        w=np.array([rng.choice([-1.0,1.0]) for _k in range(G)]); yr_star=V@w; b=float((xr@yr_star)/denom); u=yr_star-xr*b; scores=np.bincount(g,weights=xr*u,minlength=G); se=np.sqrt(np.sum(scores*scores))/denom; exceed += int(abs(b/se)>=abs(t_obs))
    return dict(sample=sample,shock=shock,moderator=moderator,n=len(x),firms=int(x.symbol.nunique()),quarters=G,beta=beta,t_obs=t_obs,B=B,seed=seed,exceed=exceed,p=(1+exceed)/(B+1))
def run(file,B=WCB_B,out_prefix="GPR_OCF_2026-09-16_v07"):
    panel=load_panel(file)
    for sample in ("NON_FN","FULL"):
        for shock in ("Threat_mean_z","Acts_mean_z"): print("FIT",sample,shock,json.dumps(fit_core(panel,shock,sample=sample)))
    for shock in ("Threat_mean_z_lead1","Threat_mean_z_lag1","Threat_mean_z_lag2","Acts_mean_z_lead1","Acts_mean_z_lag1","Acts_mean_z_lag2"): print("DYNAMIC",shock,json.dumps(fit_core(panel,shock,sample="NON_FN")))
    for shock in ("Threat_mean_z","Acts_mean_z"):
        print("EXCL2022H1",shock,json.dumps(fit_core(panel,shock,sample="NON_FN",exclude_2022h1=True)))
        vals=[]
        for yr in sorted(select_sample(panel,"NON_FN")["year"].dropna().unique()): vals.append([int(yr),fit_core(panel,shock,sample="NON_FN",exclude_year=int(yr))])
        print("LOYO",shock,json.dumps(vals)); print("SEASONALITY",shock,json.dumps(fit_core(panel,shock,sample="NON_FN",seasonality=True)))
    cases=[("NON_FN","Threat_mean_z","HL67"),("NON_FN","Acts_mean_z","HL67"),("NON_FN","Threat_mean_z","baseline_liab_ratio"),("NON_FN","Acts_mean_z","baseline_liab_ratio"),("FULL","Threat_mean_z","HL67"),("FULL","Acts_mean_z","HL67")]; rows=[]
    for sample,shock,mod in cases:
        r=wild_cluster_p(panel,shock,mod,sample,B=B,seed=WCB_SEED); rows.append(r); print("WCB",json.dumps(r),flush=True)
    pd.DataFrame(rows).to_csv(f"{out_prefix}_WCB_output.csv",index=False); runtime={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"statsmodels":statsmodels.__version__,"B":B,"seed":WCB_SEED,"input_file":str(file)}; Path(f"{out_prefix}_runtime.json").write_text(json.dumps(runtime,indent=2),encoding="utf-8")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--file",default=str(DEFAULT_FILE),help="Input workbook; defaults to verified v06 workbook"); ap.add_argument("--B",type=int,default=WCB_B); ap.add_argument("--out-prefix",default="GPR_OCF_2026-09-16_v07"); a=ap.parse_args(); run(Path(a.file),a.B,a.out_prefix)
