from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

RAW_ROOT=Path('data/gpr_bronze_refresh/fundamental_shards')
GPR_LOCK_ROOT=Path('data/gpr_repro_lock')
OUT=Path('data/gpr_gate_b_end_to_end_v2'); OUT.mkdir(parents=True,exist_ok=True)
EXPECTED={
'n':13473,'firms':567,'quarters':24,'hl67_cut':0.4721168508083521,
'Threat_z':{'beta':-0.0065111271294339915,'twoway_se':0.002628103871650693,'twoway_p':0.013243372510630458,'wcb_p':0.067},
'Acts_z':{'beta':-0.005889746491737053,'twoway_se':0.0007872093649343036,'twoway_p':7.815970093361102e-14,'wcb_p':0.003}}

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def nt(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(df,pats):
 for p in pats:
  for c in df.columns:
   if nt(c)==nt(p) or nt(p) in nt(c): return c
 return None

def load_financial():
 fs=sorted(RAW_ROOT.glob('fundamental_shard_*.parquet'))
 if len(fs)!=4: raise RuntimeError(f'expected 4 financial shards, found {len(fs)}')
 raw=pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True)
 raw['symbol']=raw.symbol.astype(str).str.upper().str.strip(); raw['period_q']=raw.period_q.astype(str); raw['value']=pd.to_numeric(raw.value,errors='coerce')
 raw=raw.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
 return raw,fs

def build_financial(raw):
 wanted={'assets':'BS_TOTAL_ASSETS','liabilities':'BS_TOTAL_LIABILITIES','current_liab':'BS_SHORT_TERM_LIABILITIES','cfo':'CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'}
 ids=set(raw.item_id.dropna().astype(str)); missing=[v for v in wanted.values() if v not in ids]
 if missing: raise RuntimeError(f'missing item ids {missing}')
 rev={v:k for k,v in wanted.items()}; z=raw[raw.item_id.isin(rev)].copy(); z['metric']=z.item_id.map(rev)
 z[['symbol','report','period_q','item_id','value']].to_csv(OUT/'financial_raw_relevant.csv',index=False)
 w=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index()
 w[['year','q']]=w.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); w=w.sort_values(['symbol','year','q']); w['pidx']=w.year*4+w.q
 w['avg_assets']=(w.assets+w.groupby('symbol').assets.shift(1))/2; w['cfo_assets']=w.cfo/w.avg_assets; w['leverage']=w.liabilities/w.assets; w['log_assets']=np.log(w.assets.where(w.assets>0)); w['liab_ratio']=w.current_liab/w.assets
 pre=w[(w.year>=2017)&(w.year<=2019)].copy(); st=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),baseline_liab_ratio=('liab_ratio','median')).reset_index(); st=st[st.pre_quarters>=2].copy()
 cut=float(st.baseline_liab_ratio.quantile(2/3)); st['HL67']=(st.baseline_liab_ratio>=cut).astype(int); st.to_csv(OUT/'baseline_highliab.csv',index=False)
 return w,st,cut,wanted

def load_gpr_locked():
 matches=sorted(GPR_LOCK_ROOT.rglob('gpr_raw_daily_2020_2025.csv'))
 if not matches: raise RuntimeError('locked GPR snapshot not found')
 p=matches[0]; g=pd.read_csv(p); dc=pick(g,['date','day']); th=pick(g,['THREATS_GPR_AI','GPR_THREATS']); ac=pick(g,['ACTS_GPR_AI','GPR_ACTS'])
 if not all([dc,th,ac]): raise RuntimeError(f'GPR columns unresolved: {list(g.columns)}')
 g[dc]=pd.to_datetime(g[dc],errors='coerce'); g=g[(g[dc]>='2020-01-01')&(g[dc]<='2025-12-31')].copy(); g.to_csv(OUT/'gpr_raw_daily_2020_2025.csv',index=False)
 g2=g.rename(columns={dc:'date',th:'Threat',ac:'Acts'}); g2['period_q']=g2.date.dt.to_period('Q').astype(str)
 qg=g2.groupby('period_q').agg(Threat=('Threat','mean'),Acts=('Acts','mean')).reset_index().sort_values('period_q')
 for c in ['Threat','Acts']: qg[c+'_z']=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)
 qg.to_csv(OUT/'gpr_quarterly_processed.csv',index=False)
 return qg,p

def prep(panel,shock):
 x=panel[['symbol','pidx','period_q','cfo_assets','log_assets','leverage','HL67',shock]].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['inter']=x[shock]*x.HL67
 return x.set_index(['symbol','pidx']).sort_index()

def fit(x,twoway=False,y=None,restricted=False):
 yy=x.cfo_assets if y is None else y; cols=['log_assets','leverage'] if restricted else ['inter','log_assets','leverage']
 mod=PanelOLS(yy,x[cols],entity_effects=True,time_effects=True,drop_absorbed=True)
 return mod.fit(cov_type='clustered',cluster_time=True,cluster_entity=True) if twoway else mod.fit(cov_type='clustered',cluster_time=True)

def vec(o): return np.asarray(o).reshape(-1)
def wcb(x,B=999,seed=20260914):
 full=fit(x); beta=float(full.params.inter); se=float(full.std_errors.inter); t0=beta/se; r0=fit(x,restricted=True); fitted=vec(r0.fitted_values); resid=vec(r0.resids)
 periods=x.period_q.astype(str).to_numpy(); uq=np.array(sorted(pd.unique(periods))); rng=np.random.default_rng(seed); ts=[]; fail=0
 for _ in range(B):
  signs=dict(zip(uq,rng.choice([-1.0,1.0],size=len(uq)))); ww=np.array([signs[p] for p in periods]); ys=pd.Series(fitted+resid*ww,index=x.index,name='ystar')
  try:
   rb=fit(x,y=ys); sb=float(rb.std_errors.inter)
   if np.isfinite(sb) and sb>0: ts.append(float(rb.params.inter/sb))
   else: fail+=1
  except Exception: fail+=1
 ts=np.asarray(ts); p=(1+int(np.sum(np.abs(ts)>=abs(t0))))/(1+len(ts))
 return {'cluster_time_se':se,'t_obs':t0,'B_requested':B,'B_success':len(ts),'failures':fail,'wcb_p':float(p)}

def close(a,b,atol,rtol=1e-9): return bool(np.isclose(a,b,atol=atol,rtol=rtol,equal_nan=False))

def main():
 raw,fs=load_financial(); w,st,cut,mapping=build_financial(raw); qg,gpr_path=load_gpr_locked()
 panel=w[(w.year>=2020)&(w.year<=2025)].merge(st[['symbol','HL67']],on='symbol',how='inner').merge(qg,on='period_q',how='left'); panel.to_parquet(OUT/'processed_panel_2020_2025.parquet',index=False); panel.to_csv(OUT/'processed_panel_2020_2025.csv',index=False)
 rows=[]; gate=True
 for shock in ['Threat_z','Acts_z']:
  x=prep(panel,shock); tw=fit(x,twoway=True); wc=wcb(x); exp=EXPECTED[shock]
  r={'shock':shock,'n':len(x),'firms':int(x.index.get_level_values(0).nunique()),'quarters':int(x.period_q.nunique()),'hl67_cut':cut,'beta':float(tw.params.inter),'twoway_se':float(tw.std_errors.inter),'twoway_p':float(tw.pvalues.inter),**wc}
  r.update(delta_beta=r['beta']-exp['beta'],delta_twoway_se=r['twoway_se']-exp['twoway_se'],delta_twoway_p=r['twoway_p']-exp['twoway_p'],delta_wcb_p=r['wcb_p']-exp['wcb_p'])
  r.update(pass_n=r['n']==EXPECTED['n'],pass_firms=r['firms']==EXPECTED['firms'],pass_quarters=r['quarters']==EXPECTED['quarters'],pass_cut=close(cut,EXPECTED['hl67_cut'],1e-12),pass_beta=close(r['beta'],exp['beta'],1e-12),pass_twoway_se=close(r['twoway_se'],exp['twoway_se'],1e-12),pass_twoway_p=close(r['twoway_p'],exp['twoway_p'],1e-10),pass_wcb_p=close(r['wcb_p'],exp['wcb_p'],1e-12,0.0),pass_wcb_complete=(r['B_success']==999 and r['failures']==0))
  r['gate_b_row_pass']=all(v for k,v in r.items() if k.startswith('pass_')); gate=gate and r['gate_b_row_pass']; rows.append(r)
 df=pd.DataFrame(rows); df.to_csv(OUT/'old_vs_rerun_delta.csv',index=False)
 audit={'gate':'B','status':'PASS' if gate else 'FAIL','raw_financial_source':'GitHub Actions artifact gpr-bronze-refresh-raw from run 34582712489','raw_financial_sha256':{Path(f).name:sha256(f) for f in fs},'gpr_source':'locked GPR raw snapshot from gpr-repro-pack-v1 artifact, workflow run 34766446325','gpr_snapshot_file':str(gpr_path),'gpr_snapshot_sha256':sha256(gpr_path),'pipeline_scope':'locked raw financial shards + locked raw GPR -> variables -> HighLiab -> panel -> FE -> WCB 999 -> comparison','resolved_item_ids':mapping,'expected':EXPECTED,'results':df.to_dict(orient='records')}
 (OUT/'gate_b_status.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(audit,indent=2,allow_nan=False))
 if not gate: raise SystemExit('GATE_B_FAIL')
if __name__=='__main__': main()
