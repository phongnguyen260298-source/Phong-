from __future__ import annotations
import os,re,time,json
from pathlib import Path
import numpy as np
import pandas as pd
from vnstock_data import Finance
from linearmodels.panel import PanelOLS

OUT=Path('data/gpr_ocf_supervisor_controls'); OUT.mkdir(parents=True,exist_ok=True)
PAUSE=float(os.getenv('REQUEST_PAUSE','0.45'))
EXCL=set('AGR APG APS BIC BMI BVH BVS CTS EVS FTS HCM IVS MBS PGI PSI PTI PVI SHS SSI TVS VCI VDS VIG VNR WSS'.split())
BASE=Path(os.getenv('BASELINE_ROOT','input/vci42'))
syms=sorted({p.stem.upper() for p in BASE.glob('chunk_*/fundamental_quarter/balance_sheet/*.parquet')})

def pq(v):
 m=re.search(r'(20\d{2}|19\d{2})\D*Q\D*([1-4])',str(v),re.I); return f'{m.group(1)}Q{m.group(2)}' if m else None

def fetch(sym,report):
 f=Finance(source='VCI',symbol=sym); fn=getattr(f,report); last=None
 for a in range(4):
  try:
   d=fn(period='quarter',lang='vi',drop_empty=False,format='long')
   if d is None or d.empty: raise RuntimeError('empty')
   if 'item_id' not in d.columns and 'id' in d.columns:d=d.rename(columns={'id':'item_id'})
   d['period_q']=d['period'].map(pq); d=d[d.period_q.notna()].copy(); d['year']=d.period_q.str[:4].astype(int)
   return d[(d.year>=2017)&(d.year<=2025)]
  except Exception as e:
   last=e; time.sleep(min(8,2**(a+1)))
 raise last

def norm(s):return re.sub(r'\s+',' ',str(s).lower().strip())
PAT={
'assets':[r'tổng cộng tài sản',r'tổng tài sản',r'total assets'], 'liabilities':[r'nợ phải trả',r'total liabilities'],
'current_liab':[r'nợ ngắn hạn',r'short.term liabilities'], 'cash':[r'tiền và các khoản tương đương tiền',r'tiền và tương đương tiền',r'cash and cash equivalents'],
'current_assets':[r'tài sản ngắn hạn',r'current assets'], 'ppe':[r'tài sản cố định hữu hình.*giá trị còn lại',r'tài sản cố định hữu hình',r'property.*plant.*equipment',r'tangible fixed assets'],
'inventory':[r'hàng tồn kho',r'inventor'], 'receivables':[r'phải thu ngắn hạn',r'các khoản phải thu',r'receivables'],
'revenue':[r'doanh thu thuần',r'net revenue',r'net sales'], 'net_income':[r'lợi nhuận sau thuế.*thu nhập doanh nghiệp',r'lợi nhuận sau thuế',r'profit after tax',r'net income'],
'ebit':[r'lợi nhuận từ hoạt động kinh doanh',r'operating profit',r'ebit'], 'cfo':[r'lưu chuyển tiền thuần từ hoạt động kinh doanh',r'net cash flows? from operating activities']}

def choose(d,key):
 if d.empty or 'item' not in d or 'value' not in d:return pd.DataFrame()
 z=d.copy(); z['_n']=z['item'].map(norm); m=pd.Series(False,index=z.index)
 for p in PAT[key]:m|=z._n.str.contains(p,regex=True,na=False)
 z=z[m].copy()
 if z.empty:return z
 z['_len']=z._n.str.len(); z=z.sort_values(['period_q','_len']).drop_duplicates('period_q')
 return z[['period_q','value','item_id','item']].rename(columns={'value':key,'item_id':key+'_item_id','item':key+'_item'})

rows=[]; errs=[]
for i,s in enumerate(syms,1):
 try:
  reps={r:fetch(s,r) for r in ['balance_sheet','income_statement','cash_flow']}; pieces=[]
  for k in ['assets','liabilities','current_liab','cash','current_assets','ppe','inventory','receivables']:pieces.append(choose(reps['balance_sheet'],k))
  for k in ['revenue','net_income','ebit']:pieces.append(choose(reps['income_statement'],k))
  pieces.append(choose(reps['cash_flow'],'cfo'))
  sets=[set(x.period_q) for x in pieces if not x.empty]; q=pd.DataFrame({'period_q':sorted(set().union(*sets))}) if sets else pd.DataFrame(columns=['period_q'])
  for x in pieces:
   if not x.empty:q=q.merge(x,on='period_q',how='left')
  q['symbol']=s; rows.append(q)
 except Exception as e:errs.append({'symbol':s,'error':repr(e)})
 print(i,len(syms),s,flush=True); time.sleep(PAUSE)
raw=pd.concat(rows,ignore_index=True); raw.to_parquet(OUT/'controls_raw_audit.parquet',index=False); pd.DataFrame(errs).to_csv(OUT/'errors.csv',index=False)
num=['assets','liabilities','current_liab','cash','current_assets','ppe','inventory','receivables','revenue','net_income','ebit','cfo']
for c in num:raw[c]=pd.to_numeric(raw.get(c),errors='coerce')
raw['qord']=raw.period_q.str[:4].astype(int)*4+raw.period_q.str[-1].astype(int); raw=raw.sort_values(['symbol','qord']); g=raw.groupby('symbol')
raw['avg_assets']=(raw.assets+g.assets.shift(1))/2; raw['size']=np.log(raw.assets.where(raw.assets>0)); raw['leverage']=raw.liabilities/raw.assets
raw['profitability']=raw.net_income/raw.avg_assets; raw['sales_growth']=raw.revenue/g.revenue.shift(1)-1; raw['cash_holdings']=raw.cash/raw.assets
raw['tangibility']=raw.ppe/raw.assets; raw['nwc']=(raw.current_assets-raw.current_liab-raw.cash)/raw.assets; raw['current_ratio']=raw.current_assets/raw.current_liab
raw['inventory_assets']=raw.inventory/raw.assets; raw['receivables_assets']=raw.receivables/raw.assets; raw['asset_growth']=raw.assets/g.assets.shift(1)-1
raw['cfo_assets']=raw.cfo/raw.avg_assets
control_base=['size','leverage','profitability','sales_growth','cash_holdings','tangibility']
control_extra=['nwc','current_ratio','inventory_assets','receivables_assets','asset_growth']
for c in control_base+control_extra:raw[c+'_l1']=g[c].shift(1)
# Exposure is defined separately inside each sample to make the comparison transparent.

def add_exposure(df):
 b=df[df.period_q.str[:4].astype(int)<=2019].copy(); b['lr']=b.current_liab/b.assets; bm=b.groupby('symbol').lr.median(); cut=bm.quantile(2/3)
 z=df.copy(); z['HL67']=z.symbol.map((bm>=cut).astype(int)); return z,float(cut)
# locate official quarterly GPR restored from locked artifacts/repo outputs
gpr_candidates=list(Path('data').rglob('*quarter*gpr*.csv'))+list(Path('data').rglob('*gpr*quarter*.csv')); gpr=None
for f in gpr_candidates:
 try:
  z=pd.read_csv(f)
  if 'period_q' in z.columns and any('threat' in c.lower() for c in z.columns):gpr=z; break
 except:pass
if gpr is None:raise RuntimeError('Quarterly GPR file not found in data/')
def col_like(words):
 for c in gpr.columns:
  if all(w in c.lower() for w in words):return c
th=col_like(['threat']); ac=col_like(['act'])
if not th or not ac:raise RuntimeError(f'Cannot detect Threat/Acts columns: {list(gpr)}')
gpr=gpr[['period_q',th,ac]].drop_duplicates('period_q').rename(columns={th:'Threat',ac:'Acts'})
for c in ['Threat','Acts']:gpr[c+'_z']=(gpr[c]-gpr[c].mean())/gpr[c].std(ddof=0)

# Sequential specifications: M1 benchmark; M2-M4 supervisor core; M5 adds NWC;
# M6 adds liquidity/working-capital composition; M7 adds balance-sheet dynamics.
models=[('M1',[]),('M2',['size_l1','leverage_l1']),('M3',['size_l1','leverage_l1','profitability_l1','sales_growth_l1']),
('M4',[c+'_l1' for c in control_base]),('M5',[c+'_l1' for c in control_base]+['nwc_l1']),
('M6',[c+'_l1' for c in control_base]+['nwc_l1','current_ratio_l1','inventory_assets_l1','receivables_assets_l1']),
('M7',[c+'_l1' for c in control_base+control_extra])]
res=[]; coverage=[]; sample_meta=[]
for sample_name,sdf in [('FULL',raw.copy()),('NON_FINANCIAL',raw[~raw.symbol.isin(EXCL)].copy())]:
 sdf,cut=add_exposure(sdf); p=sdf[sdf.period_q.str[:4].astype(int)>=2020].merge(gpr,on='period_q',how='left'); p['quarter']=p.period_q
 sample_meta.append({'sample':sample_name,'firms_raw':sdf.symbol.nunique(),'cutoff':cut,'excluded_financial_tickers':0 if sample_name=='FULL' else len(EXCL)})
 for c in [x+'_l1' for x in control_base+control_extra]:coverage.append({'sample':sample_name,'variable':c,'nonmissing':int(p[c].notna().sum()),'coverage':float(p[c].notna().mean())})
 for shock in ['Threat_z','Acts_z']:
  p[shock+'_HL']=p[shock]*p.HL67
  for name,cs in models:
   cols=['symbol','quarter','cfo_assets',shock+'_HL']+cs; d=p[cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
   # conservative trimming only for pathological ratio errors; formal winsor robustness remains separate
   for c in cs:
    if c!='size_l1' and len(d):
     lo,hi=d[c].quantile([.001,.999]); d=d[d[c].between(lo,hi)]
   d=d.set_index(['symbol','quarter'])
   try:
    fit=PanelOLS(d.cfo_assets,d[[shock+'_HL']+cs],entity_effects=True,time_effects=True,drop_absorbed=True,check_rank=False).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    term=shock+'_HL'; res.append({'sample':sample_name,'shock':shock,'model':name,'controls':'|'.join(cs) if cs else 'none','n':fit.nobs,'firms':d.index.get_level_values(0).nunique(),'beta':fit.params[term],'se':fit.std_errors[term],'p':fit.pvalues[term],'ci_low':fit.conf_int().loc[term,'lower'],'ci_high':fit.conf_int().loc[term,'upper']})
   except Exception as e:res.append({'sample':sample_name,'shock':shock,'model':name,'controls':'|'.join(cs),'n':len(d),'firms':d.index.get_level_values(0).nunique() if len(d) else 0,'error':repr(e)})
r=pd.DataFrame(res); r.to_csv(OUT/'model_results_full_vs_nonfinancial.csv',index=False); pd.DataFrame(coverage).to_csv(OUT/'coverage_full_vs_nonfinancial.csv',index=False); pd.DataFrame(sample_meta).to_csv(OUT/'sample_comparison.csv',index=False)
# direct comparison table for interpretation
ok=r[r.beta.notna()].copy(); wide=ok.pivot_table(index=['shock','model'],columns='sample',values=['beta','p','n','firms'],aggfunc='first'); wide.to_csv(OUT/'comparison_wide.csv')
raw.to_parquet(OUT/'controls_panel_full.parquet',index=False)
(OUT/'summary.json').write_text(json.dumps({'firms_requested':len(syms),'firms_observed':raw.symbol.nunique(),'financial_exclusion_list':sorted(EXCL),'errors':len(errs),'models':[m[0] for m in models],'controls_base':control_base,'controls_extra':control_extra},indent=2),encoding='utf-8')
print(pd.DataFrame(sample_meta).to_string(index=False)); print(r.to_string(index=False))