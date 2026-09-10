from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import chi2

NEW=Path('data/gpr_bronze_refresh/fundamental_shards'); OLD=Path('input/vci42'); OUT=Path('data/gpr_bronze_refresh/final'); OUT.mkdir(parents=True,exist_ok=True)
new=pd.concat([pd.read_parquet(f) for f in sorted(NEW.glob('fundamental_shard_*.parquet'))],ignore_index=True)
new['symbol']=new.symbol.astype(str).str.upper().str.strip(); new['period_q']=new.period_q.astype(str); new['value']=pd.to_numeric(new.value,errors='coerce')
new=new.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')

# Build a comparable baseline long table from locked artifact.
olds=[]
for report in ['balance_sheet','income_statement','cash_flow']:
  for f in OLD.glob(f'chunk_*/fundamental_quarter/{report}/*.parquet'):
    d=pd.read_parquet(f)
    if not {'period','item_id','value'}.issubset(d.columns): continue
    m=d.period.astype(str).str.extract(r'(20\d{2}|19\d{2})\D*Q\D*([1-4])')
    d['period_q']=m[0].fillna('')+'Q'+m[1].fillna(''); d=d[d.period_q.str.match(r'\d{4}Q[1-4]')]
    d['symbol']=f.stem.upper().strip(); d['report']=report; d['value']=pd.to_numeric(d.value,errors='coerce')
    olds.append(d[['symbol','report','period_q','item_id','value']])
old=pd.concat(olds,ignore_index=True).drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
key=['symbol','report','period_q','item_id']; cmp=old.merge(new[key+['value']],on=key,how='outer',suffixes=('_old','_new'),indicator=True)
both=cmp._merge.eq('both') & cmp.value_old.notna() & cmp.value_new.notna(); den=np.maximum(np.abs(cmp.loc[both,'value_old']),1.0); rel=np.abs(cmp.loc[both,'value_new']-cmp.loc[both,'value_old'])/den
compare={'old_rows':len(old),'new_rows':len(new),'old_symbols':old.symbol.nunique(),'new_symbols':new.symbol.nunique(),'overlap_keys':int((cmp._merge=='both').sum()),'only_old_keys':int((cmp._merge=='left_only').sum()),'only_new_keys':int((cmp._merge=='right_only').sum()),'numeric_pairs':int(both.sum()),'match_rel_1e6':float((rel<=1e-6).mean()) if len(rel) else None,'median_relative_diff':float(rel.median()) if len(rel) else None,'p95_relative_diff':float(rel.quantile(.95)) if len(rel) else None}
pd.DataFrame([compare]).to_csv(OUT/'baseline_vs_refresh_summary.csv',index=False)

# Resolver is auditable: exact preferred IDs first, then conservative token search. No silent substitution.
ids=new[['report','item_id']].dropna().drop_duplicates(); allids=set(ids.item_id.astype(str))
def resolve(preferred,tokens):
  for x in preferred:
    if x in allids: return x
  cand=sorted([x for x in allids if all(t.upper() in x.upper() for t in tokens)])
  return cand[0] if len(cand)==1 else None
mapping={
 'assets':resolve(['BS_TOTAL_ASSETS'],['TOTAL','ASSET']),
 'liabilities':resolve(['BS_TOTAL_LIABILITIES'],['TOTAL','LIABIL']),
 'current_liab':resolve(['BS_SHORT_TERM_LIABILITIES'],['SHORT','LIABIL']),
 'inventory':resolve(['BS_INVENTORIES','BS_INVENTORY'],['INVENTOR']),
 'receivables':resolve(['BS_SHORT_TERM_RECEIVABLES','BS_ACCOUNTS_RECEIVABLE'],['RECEIV']),
 'payables':resolve(['BS_SHORT_TERM_TRADE_PAYABLES','BS_TRADE_PAYABLES'],['PAYABLE']),
 'fixed_assets':resolve(['BS_FIXED_ASSETS'],['FIXED','ASSET']),
 'revenue':resolve(['IS_REVENUE','IS_NET_REVENUE'],['REVENUE']),
 'operating_profit':resolve(['IS_OPERATING_PROFIT','IS_PROFIT_FROM_OPERATING_ACTIVITIES'],['OPERAT','PROFIT']),
 'cfo':resolve(['CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'],['OPERATING','CASH'])
}
(OUT/'selected_item_ids.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
# Save candidate IDs for unresolved concepts.
pat='INVENT|RECEIV|PAYABLE|FIXED|REVENUE|OPERAT.*PROFIT|PROFIT.*OPERAT|CASH.*OPERAT'
ids[ids.item_id.astype(str).str.contains(pat,case=False,regex=True)].sort_values(['report','item_id']).to_csv(OUT/'item_id_candidates.csv',index=False)
req=['assets','liabilities','current_liab','cfo']
if any(mapping[x] is None for x in req): raise RuntimeError(f'Core item ids unresolved: {mapping}')

use={v:k for k,v in mapping.items() if v is not None}
z=new[new.item_id.isin(use)].copy(); z['metric']=z.item_id.map(use); wide=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index(); wide[['year','q']]=wide.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int); wide=wide.sort_values(['symbol','year','q']); wide['pidx']=wide.year*4+wide.q
wide['avg_assets']=(wide.assets+wide.groupby('symbol').assets.shift(1))/2; wide['cfo_assets']=wide.cfo/wide.avg_assets; wide['leverage']=wide.liabilities/wide.assets; wide['log_assets']=np.log(wide.assets.where(wide.assets>0))

# Pre-treatment firm-level ratios, medians across 2017-2019; min two valid quarters.
pre=wide[wide.year<2020].copy()
pre['liab_ratio']=pre.current_liab/pre.assets
if 'inventory' in pre: pre['inventory_ratio']=pre.inventory/pre.assets
if 'receivables' in pre: pre['receivable_ratio']=pre.receivables/pre.assets
if 'payables' in pre: pre['payable_ratio']=pre.payables/pre.assets
if 'fixed_assets' in pre: pre['fixed_ratio']=pre.fixed_assets/pre.assets
if {'inventory_ratio','receivable_ratio','payable_ratio'}.issubset(pre.columns): pre['nwc_tied_ratio']=pre.inventory_ratio+pre.receivable_ratio-pre.payable_ratio
if {'operating_profit','revenue'}.issubset(pre.columns): pre['operating_margin']=pre.operating_profit/pre.revenue.replace(0,np.nan)
metrics=[c for c in ['liab_ratio','nwc_tied_ratio','fixed_ratio','operating_margin'] if c in pre.columns]
stat=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),**{m:(m,'median') for m in metrics}).reset_index(); stat=stat[stat.pre_quarters>=2].copy()
# Transparent pre-declared grouping rules.
q67={m:stat[m].quantile(2/3) for m in metrics}; q33={m:stat[m].quantile(1/3) for m in metrics}
stat['HighLiab_TopTercile']=(stat.liab_ratio>=q67['liab_ratio']).astype(int)
if 'nwc_tied_ratio' in stat:
 stat['CashConversionTrap']=(stat.nwc_tied_ratio>=q67['nwc_tied_ratio']).astype(int); stat['DoubleExposure_TopTercile']=((stat.HighLiab_TopTercile==1)&(stat.CashConversionTrap==1)).astype(int)
if 'operating_margin' in stat: stat['FragileLowMarginFunding']=((stat.HighLiab_TopTercile==1)&(stat.operating_margin<=q33['operating_margin'])).astype(int)
if 'fixed_ratio' in stat: stat['AssetCommitmentMismatch']=((stat.HighLiab_TopTercile==1)&(stat.fixed_ratio>=q67['fixed_ratio'])).astype(int)
stat.to_csv(OUT/'refreshed_pre_treatment_groups.csv',index=False)
criteria={'window':'2017-2019; firm median; require >=2 valid quarters','HighLiab_TopTercile':'liab_ratio >= cross-sectional 66.7 percentile','CashConversionTrap':'nwc_tied_ratio=(inventory+receivables-payables)/assets >= 66.7 percentile','DoubleExposure_TopTercile':'HighLiab AND CashConversionTrap','FragileLowMarginFunding':'HighLiab AND operating_margin <= 33.3 percentile','AssetCommitmentMismatch':'HighLiab AND fixed_assets/assets >= 66.7 percentile','thresholds_q67':q67,'thresholds_q33':q33}
(OUT/'group_criteria.json').write_text(json.dumps(criteria,ensure_ascii=False,indent=2),encoding='utf-8')

# Official AI-GPR vs traditional GPR, quarterly mean standardized on 2020-2025.
g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv')
def nt(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(pats):
 for p in pats:
  for c in g.columns:
   if nt(c)==nt(p) or nt(p) in nt(c): return c
 return None
dc=pick(['date','day']); ai=pick(['GPR_AI','AI_GPR']); th=pick(['THREATS_GPR_AI','GPR_THREATS']); ac=pick(['ACTS_GPR_AI','GPR_ACTS']); orig=pick(['GPR_AER','original'])
if not all([dc,ai,th,ac]): raise RuntimeError(f'AI-GPR columns unresolved: {list(g.columns)}')
g=g.rename(columns={dc:'date',ai:'AI_GPR',th:'Threat',ac:'Acts',**({orig:'GPR_ORIG'} if orig else {})}); g.date=pd.to_datetime(g.date,errors='coerce'); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')]; g['period_q']=g.date.dt.to_period('Q').astype(str); sh=['AI_GPR','Threat','Acts']+(['GPR_ORIG'] if 'GPR_ORIG' in g else []); qg=g.groupby('period_q')[sh].mean().reset_index()
for c in sh: qg[c]=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)

panel=wide[(wide.year>=2020)&(wide.year<=2025)].merge(stat,on='symbol',how='inner').merge(qg,on='period_q',how='left'); panel['Post2022']=(panel.year>=2022).astype(int)

def fit_inter(data,shock,grp):
 x=data[['symbol','pidx','cfo_assets','log_assets','leverage',shock,grp]].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['inter']=x[shock]*x[grp]; x=x.set_index(['symbol','pidx']).sort_index(); r=PanelOLS(x.cfo_assets,x[['inter','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True); return len(x),x.index.get_level_values(0).nunique(),float(r.params['inter']),float(r.std_errors['inter']),float(r.pvalues['inter']),float(r.rsquared_within)

def fit_did(data,grp):
 x=data[['symbol','pidx','cfo_assets','log_assets','leverage','Post2022',grp]].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['did']=x.Post2022*x[grp]; x=x.set_index(['symbol','pidx']).sort_index(); r=PanelOLS(x.cfo_assets,x[['did','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True); return len(x),x.index.get_level_values(0).nunique(),float(r.params['did']),float(r.std_errors['did']),float(r.pvalues['did'])

groups=[c for c in ['HighLiab_TopTercile','CashConversionTrap','DoubleExposure_TopTercile','FragileLowMarginFunding','AssetCommitmentMismatch'] if c in stat]
rows=[]
for grp in groups:
 for shock in sh:
  n,f,b,se,pv,r2=fit_inter(panel,shock,grp); rows.append({'test':'shock_interaction','group':grp,'shock':shock,'n':n,'firms':f,'beta':b,'se':se,'p':pv,'r2_within':r2})
 n,f,b,se,pv=fit_did(panel,grp); rows.append({'test':'DID_Post2022','group':grp,'shock':'Post2022','n':n,'firms':f,'beta':b,'se':se,'p':pv})
res=pd.DataFrame(rows); res['q_fdr_family']=np.nan
for fam,idx in res.groupby('test').groups.items():
 ii=list(idx); pv=res.loc[ii,'p'].to_numpy(); order=np.argsort(pv); m=len(pv); q=np.empty(m); run=1.
 for j in range(m-1,-1,-1): k=order[j]; run=min(run,pv[k]*m/(j+1)); q[k]=min(1.,run)
 res.loc[ii,'q_fdr_family']=q
res.to_csv(OUT/'model_results_with_did.csv',index=False)
qa={'refresh_compare':compare,'selected_item_ids':mapping,'analysis_firms':int(panel.symbol.nunique()),'analysis_rows':int(len(panel)),'duplicate_symbol_quarter':int(panel.duplicated(['symbol','period_q']).sum()),'groups':{g0:{'treated':int(stat.loc[stat[g0]==1,'symbol'].nunique()),'control':int(stat.loc[stat[g0]==0,'symbol'].nunique())} for g0 in groups},'ai_measurement_note':'AI_GPR/Threat/Acts are AI-derived measures of geopolitical risk, not firm AI adoption or an AI economic shock.'}
(OUT/'qa_summary.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(qa,ensure_ascii=False,indent=2))