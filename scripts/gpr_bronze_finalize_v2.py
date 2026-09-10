from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS

ROOT=Path('input/vci42')
NEW=Path('data/gpr_bronze_refresh/fundamental_shards')
OUT=Path('data/gpr_bronze_refresh/final'); OUT.mkdir(parents=True,exist_ok=True)

# ---------- helpers ----------
def bh(p):
    p=np.asarray(p,float); m=len(p); order=np.argsort(p); q=np.empty(m); run=1.0
    for j in range(m-1,-1,-1):
        k=order[j]; run=min(run,p[k]*m/(j+1)); q[k]=min(1.0,run)
    return q

def nt(s): return re.sub(r'[^a-z0-9]','',str(s).lower())

def resolve(allids, preferred, tokens):
    for x in preferred:
        if x in allids: return x
    cand=sorted([x for x in allids if all(t.upper() in x.upper() for t in tokens)])
    return cand[0] if len(cand)==1 else None

# ---------- refreshed fundamentals ----------
new=pd.concat([pd.read_parquet(f) for f in sorted(NEW.glob('fundamental_shard_*.parquet'))],ignore_index=True)
new['symbol']=new.symbol.astype(str).str.upper().str.strip(); new['period_q']=new.period_q.astype(str)
new['value']=pd.to_numeric(new.value,errors='coerce')
new=new.drop_duplicates(['symbol','report','period_q','item_id'],keep='last')
ids=new[['report','item_id']].dropna().drop_duplicates(); allids=set(ids.item_id.astype(str))

mapping={
 'assets':resolve(allids,['BS_TOTAL_ASSETS'],['TOTAL','ASSET']),
 'liabilities':resolve(allids,['BS_TOTAL_LIABILITIES'],['TOTAL','LIABIL']),
 'current_liab':resolve(allids,['BS_SHORT_TERM_LIABILITIES'],['SHORT','LIABIL']),
 'inventory':resolve(allids,['BS_INVENTORIES','BS_INVENTORY'],['INVENTOR']),
 'receivables':resolve(allids,['BS_SHORT_TERM_RECEIVABLES','BS_ACCOUNTS_RECEIVABLE','BS_TRADE_RECEIVABLES'],['SHORT','RECEIV']),
 'payables':resolve(allids,['BS_TRADE_ACCOUNTS_PAYABLE','BS_SHORT_TERM_TRADE_PAYABLES','BS_TRADE_PAYABLES'],['TRADE','PAYABLE']),
 'fixed_assets':resolve(allids,['BS_FIXED_ASSETS'],['FIXED','ASSET']),
 'revenue':resolve(allids,['IS_REVENUE','IS_NET_REVENUE'],['REVENUE']),
 'operating_profit':resolve(allids,['IS_OPERATING_PROFIT','IS_PROFIT_FROM_OPERATING_ACTIVITIES'],['OPERAT','PROFIT']),
 'cfo':resolve(allids,['CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'],['OPERATING','CASH'])
}
(OUT/'selected_item_ids_v2.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
req=['assets','liabilities','current_liab','inventory','receivables','payables','fixed_assets','revenue','operating_profit','cfo']
missing=[x for x in req if mapping[x] is None]
if missing: raise RuntimeError(f'Unresolved required concepts: {missing}; mapping={mapping}')

use={v:k for k,v in mapping.items()}
z=new[new.item_id.isin(use)].copy(); z['metric']=z.item_id.map(use)
wide=z.pivot_table(index=['symbol','period_q'],columns='metric',values='value',aggfunc='last').reset_index()
wide[['year','q']]=wide.period_q.str.extract(r'(\d{4})Q([1-4])').astype(int)
wide=wide.sort_values(['symbol','year','q']); wide['pidx']=wide.year*4+wide.q
wide['avg_assets']=(wide.assets+wide.groupby('symbol').assets.shift(1))/2
wide['cfo_assets']=wide.cfo/wide.avg_assets; wide['leverage']=wide.liabilities/wide.assets; wide['log_assets']=np.log(wide.assets.where(wide.assets>0))

# ---------- transparent pre-2020 grouping ----------
pre=wide[wide.year<2020].copy()
pre['liab_ratio']=pre.current_liab/pre.assets
pre['inventory_ratio']=pre.inventory/pre.assets
pre['receivable_ratio']=pre.receivables/pre.assets
pre['payable_ratio']=pre.payables/pre.assets
pre['fixed_ratio']=pre.fixed_assets/pre.assets
pre['nwc_tied_ratio']=pre.inventory_ratio+pre.receivable_ratio-pre.payable_ratio
pre['operating_margin']=pre.operating_profit/pre.revenue.replace(0,np.nan)
metrics=['liab_ratio','nwc_tied_ratio','fixed_ratio','operating_margin']
stat=pre.groupby('symbol').agg(pre_quarters=('liab_ratio','count'),**{m:(m,'median') for m in metrics}).reset_index()
stat=stat[stat.pre_quarters>=2].copy()
q67={m:float(stat[m].quantile(2/3)) for m in metrics}; q33={m:float(stat[m].quantile(1/3)) for m in metrics}
stat['HighLiab_TopTercile']=(stat.liab_ratio>=q67['liab_ratio']).astype(int)
stat['CashConversionTrap']=(stat.nwc_tied_ratio>=q67['nwc_tied_ratio']).astype(int)
stat['DoubleExposure_TopTercile']=((stat.HighLiab_TopTercile==1)&(stat.CashConversionTrap==1)).astype(int)
stat['FragileLowMarginFunding']=((stat.HighLiab_TopTercile==1)&(stat.operating_margin<=q33['operating_margin'])).astype(int)
stat['AssetCommitmentMismatch']=((stat.HighLiab_TopTercile==1)&(stat.fixed_ratio>=q67['fixed_ratio'])).astype(int)
stat.to_csv(OUT/'refreshed_pre_treatment_groups_v2.csv',index=False)
criteria={'window':'2017-2019; firm median; require >=2 valid quarters','thresholds_q67':q67,'thresholds_q33':q33,
'HighLiab_TopTercile':'current liabilities/assets >= 66.7 percentile','CashConversionTrap':'(inventory+short-term receivables-trade accounts payable)/assets >= 66.7 percentile','DoubleExposure_TopTercile':'HighLiab AND CashConversionTrap','FragileLowMarginFunding':'HighLiab AND operating margin <= 33.3 percentile','AssetCommitmentMismatch':'HighLiab AND fixed assets/assets >= 66.7 percentile'}
(OUT/'group_criteria_v2.json').write_text(json.dumps(criteria,ensure_ascii=False,indent=2),encoding='utf-8')

# ---------- AI-GPR / Threat / Acts / traditional GPR ----------
g=pd.read_csv('https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv')
def pick(pats):
    for p in pats:
        for c in g.columns:
            if nt(c)==nt(p) or nt(p) in nt(c): return c
    return None
dc=pick(['date','day']); ai=pick(['GPR_AI','AI_GPR']); th=pick(['THREATS_GPR_AI','GPR_THREATS']); ac=pick(['ACTS_GPR_AI','GPR_ACTS']); orig=pick(['GPR_AER','original'])
if not all([dc,ai,th,ac,orig]): raise RuntimeError(f'GPR columns unresolved: {list(g.columns)}')
g=g.rename(columns={dc:'date',ai:'AI_GPR',th:'Threat',ac:'Acts',orig:'GPR_ORIG'})
g.date=pd.to_datetime(g.date,errors='coerce'); g=g[(g.date>='2020-01-01')&(g.date<='2025-12-31')]
g['period_q']=g.date.dt.to_period('Q').astype(str); shocks=['AI_GPR','Threat','Acts','GPR_ORIG']
qg=g.groupby('period_q')[shocks].mean().reset_index()
for c in shocks: qg[c]=(qg[c]-qg[c].mean())/qg[c].std(ddof=0)

panel=wide[(wide.year>=2020)&(wide.year<=2025)].merge(stat,on='symbol',how='inner').merge(qg,on='period_q',how='left')
panel['Post2022']=(panel.year>=2022).astype(int); panel['event_q']=panel.pidx-(2022*4+1)
groups=['HighLiab_TopTercile','CashConversionTrap','DoubleExposure_TopTercile','FragileLowMarginFunding','AssetCommitmentMismatch']

def fit_inter(shock,grp):
    x=panel[['symbol','pidx','cfo_assets','log_assets','leverage',shock,grp]].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['inter']=x[shock]*x[grp]
    x=x.set_index(['symbol','pidx']).sort_index(); r=PanelOLS(x.cfo_assets,x[['inter','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    return len(x),x.index.get_level_values(0).nunique(),float(r.params.inter),float(r.std_errors.inter),float(r.pvalues.inter),float(r.rsquared_within)

def fit_did(grp):
    x=panel[['symbol','pidx','cfo_assets','log_assets','leverage','Post2022',grp]].replace([np.inf,-np.inf],np.nan).dropna().copy(); x['did']=x.Post2022*x[grp]
    x=x.set_index(['symbol','pidx']).sort_index(); r=PanelOLS(x.cfo_assets,x[['did','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    return len(x),x.index.get_level_values(0).nunique(),float(r.params.did),float(r.std_errors.did),float(r.pvalues.did)

rows=[]
for grp in groups:
    for shock in shocks:
        n,f,b,se,pv,r2=fit_inter(shock,grp); rows.append({'test':'shock_interaction','group':grp,'shock':shock,'n':n,'firms':f,'beta':b,'se':se,'p':pv,'r2_within':r2})
    n,f,b,se,pv=fit_did(grp); rows.append({'test':'DID_Post2022','group':grp,'shock':'Post2022','n':n,'firms':f,'beta':b,'se':se,'p':pv,'r2_within':np.nan})
res=pd.DataFrame(rows); res['q_fdr_family']=np.nan
for fam,idx in res.groupby('test').groups.items(): res.loc[list(idx),'q_fdr_family']=bh(res.loc[list(idx),'p'].values)
res.to_csv(OUT/'model_results_with_did_v2.csv',index=False)

# ---------- dynamic event study around 2022Q1 ----------
# Omit t=-1. Leads -8..-2 test pre-trends; lags 0..15 show post dynamics.
event_rows=[]; pre_rows=[]
for grp in groups:
    x=panel[['symbol','pidx','event_q','cfo_assets','log_assets','leverage',grp]].replace([np.inf,-np.inf],np.nan).dropna().copy()
    terms=[]; ks=list(range(-8,-1))+list(range(0,16))
    for k in ks:
        nm=('lead_m'+str(abs(k))) if k<0 else ('lag_'+str(k)); x[nm]=((x.event_q==k).astype(int)*x[grp]); terms.append(nm)
    x=x.set_index(['symbol','pidx']).sort_index()
    r=PanelOLS(x.cfo_assets,x[terms+['log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True,check_rank=False).fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
    for k,nm in zip(ks,terms):
        if nm in r.params.index: event_rows.append({'group':grp,'event_q':k,'term':nm,'beta':float(r.params[nm]),'se':float(r.std_errors[nm]),'p':float(r.pvalues[nm])})
    leads=[nm for k,nm in zip(ks,terms) if k<0 and nm in r.params.index]
    b=r.params[leads].values; V=r.cov.loc[leads,leads].values
    try:
        w=float(b@np.linalg.pinv(V)@b); df=len(leads)
        from scipy.stats import chi2
        jp=float(chi2.sf(w,df))
    except Exception: w=np.nan; df=len(leads); jp=np.nan
    pre_rows.append({'group':grp,'lead_terms':df,'wald_chi2':w,'pretrend_joint_p':jp})
ev=pd.DataFrame(event_rows); ev['q_fdr_event']=bh(ev.p.values); ev.to_csv(OUT/'event_study_v2.csv',index=False)
pretest=pd.DataFrame(pre_rows); pretest.to_csv(OUT/'pretrend_summary_v2.csv',index=False)

# ---------- final five-direction comparison ----------
def grab(grp,shock,test='shock_interaction'):
    r=res[(res.group==grp)&(res.shock==shock)&(res.test==test)].iloc[0]; return r
pt=dict(zip(pretest.group,pretest.pretrend_joint_p))
Ntreated={g:int(stat.loc[stat[g]==1,'symbol'].nunique()) for g in groups}

def gate(r,grp):
    return ('PASS' if (r.q_fdr_family<0.05 and pt.get(grp,np.nan)>=0.10 and Ntreated[grp]>=50) else 'HOLD')

dirs=[]
for rank,name,grp,shock in [
 (1,'Threat × HighLiab — anticipation','HighLiab_TopTercile','Threat'),
 (2,'Acts × HighLiab — realization','HighLiab_TopTercile','Acts'),
 (3,'Acts × CashConversionTrap','CashConversionTrap','Acts'),
 (4,'Threat × DoubleExposure + Post2022 DID','DoubleExposure_TopTercile','Threat'),
 (5,'FragileFunding challenger','FragileLowMarginFunding','AI_GPR')]:
    r=grab(grp,shock); did=grab(grp,'Post2022','DID_Post2022')
    dirs.append({'candidate_order':rank,'direction':name,'group':grp,'primary_shock':shock,'beta_primary':r.beta,'p_primary':r.p,'q_primary':r.q_fdr_family,'did_beta':did.beta,'did_p':did.p,'did_q':did.q_fdr_family,'pretrend_joint_p':pt.get(grp,np.nan),'treated_firms':Ntreated[grp],'gate':gate(r,grp)})
# Add AssetCommitmentMismatch evidence to challenger diagnostics without pretending it passed.
amm=grab('AssetCommitmentMismatch','AI_GPR'); dirs[-1]['asset_mismatch_q']=amm.q_fdr_family; dirs[-1]['asset_mismatch_pretrend_p']=pt.get('AssetCommitmentMismatch',np.nan)
final=pd.DataFrame(dirs)
# Selection rule fixed before looking at sign: PASS gate first, then smallest FDR q; ties -> larger treated sample.
final['eligible']=(final.gate=='PASS').astype(int)
final=final.sort_values(['eligible','q_primary','treated_firms'],ascending=[False,True,False]).reset_index(drop=True)
final['final_rank']=np.arange(1,len(final)+1)
final['selected']=(final.final_rank==1)&(final.eligible==1)
final.to_csv(OUT/'FINAL_DIRECTION_COMPARISON_V2.csv',index=False)
selected=final.loc[final.selected,'direction'].tolist()
summary={'baseline_refresh_exact_match':bool(json.loads((OUT/'qa_summary.json').read_text()).get('refresh_compare',{}).get('match_rel_1e6',0)==1.0) if (OUT/'qa_summary.json').exists() else None,
         'analysis_firms':int(panel.symbol.nunique()),'analysis_rows':int(len(panel)),'duplicate_symbol_quarter':int(panel.duplicated(['symbol','period_q']).sum()),'selected_direction':selected[0] if selected else None,'selection_rule':'PASS requires q<0.05, joint pretrend p>=0.10, treated firms>=50; rank PASS candidates by q then treated sample. No p-hacking; negative/null results retained.','measurement_note':'AI_GPR/Threat/Acts are AI-derived geopolitical-risk measures, not firm AI adoption or an AI economic shock.'}
(OUT/'FINAL_QA_AND_SELECTION_V2.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2)); print(final.to_string(index=False))
