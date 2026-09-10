from __future__ import annotations
import re, json, unicodedata, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from linearmodels.panel import PanelOLS

warnings.filterwarnings('ignore')
ROOT=Path('input/vci42')
OUT=Path('output/xy_screen'); OUT.mkdir(parents=True,exist_ok=True)

def normtxt(s):
    s=str(s).replace('Đ','D').replace('đ','d')
    s=unicodedata.normalize('NFD',s); s=''.join(c for c in s if unicodedata.category(c)!='Mn')
    return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()

def qp(v):
    s=str(v); y=re.search(r'(20\d{2}|19\d{2})',s)
    q=re.search(r'[Qq]\s*([1-4])',s) or re.search(r'qu[yý]\s*([1-4])',s.lower())
    if not y: return None
    yy=int(y.group(1)); qq=int(q.group(1)) if q else None
    if qq is None:
        nums=[int(x) for x in re.findall(r'\b[1-4]\b',s)]
        qq=nums[0] if nums else None
    return f'{yy}Q{qq}' if qq else None

def detect(df):
    cols=list(df.columns); low={c:normtxt(c) for c in cols}
    p=next((c for c in cols if low[c] in {'period','ky','report period'} or 'period' in low[c]),None)
    n=next((c for c in cols if low[c] in {'item','item name','item_name','name','chi tieu','ten chi tieu','metric','title'}),None)
    v=next((c for c in cols if low[c] in {'value','gia tri','amount','numeric value'}),None)
    if n is None:
        cand=[c for c in cols if c!=p and df[c].dtype=='object' and low[c] not in {'retrieval source','retrieval utc','ticker','symbol'}]
        n=max(cand,key=lambda c:df[c].nunique(dropna=True),default=None)
    if v is None:
        cand=[c for c in cols if c!=p and pd.api.types.is_numeric_dtype(df[c])]
        v=max(cand,key=lambda c:df[c].notna().sum(),default=None)
    if not p or not n or not v: raise RuntimeError(f'Cannot detect schema: {cols}; detected p={p}, n={n}, v={v}')
    return p,n,v

ALIASES={
'assets':['tong cong tai san','tong tai san','total assets'],
'inventory':['hang ton kho','inventories','inventory'],
'receivables':['cac khoan phai thu ngan han','phai thu ngan han','short term receivables','accounts receivable'],
'current_liab':['no ngan han','current liabilities'],
'liabilities':['no phai tra','tong no phai tra','total liabilities'],
'cash':['tien va cac khoan tuong duong tien','cash and cash equivalents'],
'fixed_assets':['tai san co dinh','fixed assets'],
'revenue':['doanh thu thuan ve ban hang va cung cap dich vu','doanh thu thuan','net revenue','revenue'],
'cogs':['gia von hang ban','cost of goods sold','cost of sales'],
'gross_profit':['loi nhuan gop ve ban hang va cung cap dich vu','loi nhuan gop','gross profit'],
'net_income':['loi nhuan sau thue thu nhap doanh nghiep','loi nhuan sau thue','net profit','net income'],
'cfo':['luu chuyen tien thuan tu hoat dong kinh doanh','luu chuyen tien thuan tu hd kinh doanh','net cash flows from operating activities','net cash flow from operating activities'],
'capex':['tien chi de mua sam xay dung tscd va cac tai san dai han khac','tien chi mua sam xay dung tscd','purchase of fixed assets','payments for acquisition of fixed assets']}

# Prefer stable VCI item_id codes; text matching is only a fallback.
ITEM_IDS={
'assets':['BS_TOTAL_ASSETS'],
'inventory':['BS_INVENTORIES'],
'receivables':['BS_SHORT_TERM_RECEIVABLES'],
'current_liab':['BS_SHORT_TERM_LIABILITIES'],
'liabilities':['BS_TOTAL_LIABILITIES'],
'cash':['BS_CASH_AND_PRECIOUS_METALS'],
'fixed_assets':['BS_FIXED_ASSETS'],
'revenue':['IS_NET_REVENUE'],
'cogs':['IS_COST_OF_GOODS_SOLD'],
'gross_profit':['IS_GROSS_PROFIT'],
'net_income':['IS_NET_PROFIT_AFTER_TAX'],
'cfo':['CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'],
'capex':['CF_PAYMENTS_FOR_FIXED_ASSETS']}
ID_TO_KEY={item_id:k for k,ids in ITEM_IDS.items() for item_id in ids}

def match(name, keys):
    z=normtxt(name)
    for k in keys:
        for a in ALIASES[k]:
            a2=normtxt(a)
            if z==a2 or a2 in z: return k
    return None

def load_report(report, keys):
    fs=sorted(ROOT.glob(f'chunk_*/fundamental_quarter/{report}/*.parquet'))
    rows=[]; schemas=[]
    for f in fs:
        d=pd.read_parquet(f); p,n,v=detect(d)
        if len(schemas)<4: schemas.append({'file':str(f),'columns':list(d.columns),'period':p,'name':n,'value':v})
        cols=[p,n,v] + (['item_id'] if 'item_id' in d.columns and 'item_id' not in {p,n,v} else [])
        x=d[cols].copy(); x['symbol']=f.stem; x['period_q']=x[p].map(qp)
        text_metric=x[n].map(lambda s:match(s,keys))
        if 'item_id' in x.columns:
            id_metric=x['item_id'].astype(str).map(lambda s:ID_TO_KEY.get(s) if ID_TO_KEY.get(s) in keys else None)
            x['metric']=id_metric.where(id_metric.notna(),text_metric)
        else:
            x['metric']=text_metric
        x=x[x.metric.notna() & x.period_q.notna()]
        x['value']=pd.to_numeric(x[v],errors='coerce')
        rows.append(x[['symbol','period_q','metric','value',n]].rename(columns={n:'item_name'}))
    z=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    (OUT/f'{report}_schema.json').write_text(json.dumps(schemas,ensure_ascii=False,indent=2),encoding='utf-8')
    if z.empty:
        raise RuntimeError(f'No matched metrics for report={report}, keys={keys}; VCI files={len(fs)}')
    z['name_len']=z.item_name.astype(str).str.len()
    z=z.sort_values(['symbol','period_q','metric','name_len']).drop_duplicates(['symbol','period_q','metric'])
    return z.pivot(index=['symbol','period_q'],columns='metric',values='value').reset_index()

bs=load_report('balance_sheet',['assets','inventory','receivables','current_liab','liabilities','cash','fixed_assets'])
is_=load_report('income_statement',['revenue','cogs','gross_profit','net_income'])
cf=load_report('cash_flow',['cfo','capex'])
panel=bs.merge(is_,on=['symbol','period_q'],how='outer').merge(cf,on=['symbol','period_q'],how='outer')
panel[['year','q']]=panel.period_q.str.extract(r'(\d{4})Q([1-4])').astype(float); panel=panel.sort_values(['symbol','year','q'])
# core derived metrics
G=panel.groupby('symbol',group_keys=False)
for c in ['assets','inventory','receivables']:
    panel[f'avg_{c}']=(panel[c]+G[c].shift(1))/2
panel['dio']=panel.avg_inventory/panel.cogs.abs()*91.25
panel['dso']=panel.avg_receivables/panel.revenue.abs()*91.25
panel['cfo_assets']=panel.cfo/panel.avg_assets
panel['capex_assets']=panel.capex.abs()/panel.avg_assets
panel['gross_margin']=panel.gross_profit/panel.revenue
panel['roa_q']=panel.net_income/panel.avg_assets
panel['leverage']=panel.liabilities/panel.assets
panel['log_assets']=np.log(panel.assets.where(panel.assets>0))
panel['rev_growth_yoy']=panel.revenue/G.revenue.shift(4)-1
# plausibility cleaning only, no outcome-driven row deletion
for c in ['dio','dso']:
    panel.loc[(panel[c]<0)|(panel[c]>1000),c]=np.nan
for c in ['cfo_assets','capex_assets','gross_margin','roa_q','leverage','rev_growth_yoy']:
    panel.loc[panel[c].abs()>5,c]=np.nan
# pre-2020 fixed exposures
pre=panel[(panel.year>=2017)&(panel.year<=2019)].copy()
pre['inv_exp']=pre.inventory/pre.assets; pre['recv_exp']=pre.receivables/pre.assets; pre['liab_exp']=pre.current_liab/pre.assets
pre['fixed_exp']=pre.fixed_assets/pre.assets; pre['wc_exp']=(pre.inventory+pre.receivables)/pre.assets
expo=pre.groupby('symbol')[['inv_exp','recv_exp','liab_exp','fixed_exp','wc_exp']].median()
for c in expo:
    lo,hi=expo[c].quantile([.01,.99]); expo[c]=expo[c].clip(lo,hi); expo[c]=(expo[c]-expo[c].mean())/expo[c].std(ddof=0)
panel=panel.merge(expo,left_on='symbol',right_index=True,how='left')
# download AI-GPR monthly data (official Iacoviello/Tong page)
url='https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_monthly.csv'
g=pd.read_csv(url)
(OUT/'ai_gpr_columns.json').write_text(json.dumps(list(g.columns),ensure_ascii=False,indent=2))
def pick_col(tokens,exclude=()):
    scored=[]
    for c in g.columns:
        z=normtxt(c)
        if any(e in z for e in exclude): continue
        score=sum(t in z for t in tokens)
        if score: scored.append((score,-len(z),c))
    if not scored: return None
    return sorted(scored,reverse=True)[0][2]
date_col=next((c for c in g.columns if 'date' in normtxt(c) or 'month' in normtxt(c)),g.columns[0])
dt=pd.to_datetime(g[date_col],errors='coerce')
if dt.isna().mean()>.5:
    # supports YYYYMM integer formats
    dt=pd.to_datetime(g[date_col].astype(str).str.replace(r'\.0$','',regex=True),format='%Y%m',errors='coerce')
g['period_q']=dt.dt.to_period('Q').astype(str).str.replace('Q','Q',regex=False)
shock_cols={
'AI_GPR': pick_col(['gpr','ai'],exclude=('oil','threat','act','original','keyword')),
'GPR_ORIGINAL': pick_col(['gpr','original']),
'GPR_THREATS': pick_col(['threat']),
'GPR_ACTS': pick_col(['acts']) or pick_col(['act'])}
if any(v is None for v in shock_cols.values()): raise RuntimeError(f'Cannot identify shock columns: {shock_cols}; cols={list(g.columns)}')
shock=g.groupby('period_q')[[v for v in shock_cols.values()]].mean().rename(columns={v:k for k,v in shock_cols.items()}).reset_index()
for c in shock_cols:
    shock[c]=pd.to_numeric(shock[c],errors='coerce')
    m=(shock.period_q.str[:4].astype(int).between(2020,2025))
    mu=shock.loc[m,c].mean(); sd=shock.loc[m,c].std(ddof=0); shock[c]=(shock[c]-mu)/sd
panel=panel.merge(shock,on='period_q',how='left')
# save consolidated analysis panel
panel.to_parquet(OUT/'vci42_analysis_panel.parquet',index=False)
panel[(panel.year>=2020)&(panel.year<=2025)].to_csv(OUT/'vci42_analysis_panel_2020_2025.csv',index=False,encoding='utf-8-sig')
# exactly 20 pre-specified candidate pairs = 4 risk measures x 5 traditional outcomes
outs=[('DIO','dio','inv_exp'),('DSO','dso','recv_exp'),('CFO_ASSETS','cfo_assets','liab_exp'),('CAPEX_ASSETS','capex_assets','fixed_exp'),('GROSS_MARGIN','gross_margin','wc_exp')]
shocks=list(shock_cols.keys())

def wins(s):
    a=s.quantile(.01); b=s.quantile(.99); return s.clip(a,b)
def fit_pair(sh,y,ex):
    d=panel[(panel.year>=2020)&(panel.year<=2025)][['symbol','period_q',y,sh,ex,'log_assets','leverage']].dropna().copy()
    d['x']=d[sh]*d[ex]; d[y]=wins(d[y]); d['x']=wins(d.x); d['log_assets']=wins(d.log_assets); d['leverage']=wins(d.leverage)
    d=d.set_index(['symbol','period_q']).sort_index()
    if d.index.get_level_values(0).nunique()<80 or len(d)<800: return {'n':len(d),'firms':d.index.get_level_values(0).nunique(),'error':'coverage'}
    mod=PanelOLS(d[y],d[['x','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True)
    r=mod.fit(cov_type='clustered',cluster_entity=True)
    beta=float(r.params['x']); se=float(r.std_errors['x']); p=float(r.pvalues['x']);
    # robustness 1: no controls
    r0=PanelOLS(d[y],d[['x']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True)
    # robustness 2: exclude extreme 2022Q1-Q2 event concentration
    d2=d[~d.index.get_level_values(1).isin(['2022Q1','2022Q2'])]
    r2=PanelOLS(d2[y],d2[['x','log_assets','leverage']],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True)
    return {'n':len(d),'firms':d.index.get_level_values(0).nunique(),'beta':beta,'se':se,'t':beta/se,'p':p,'r2_within':float(r.rsquared_within),'beta_nocontrol':float(r0.params['x']),'p_nocontrol':float(r0.pvalues['x']),'beta_excl_2022h1':float(r2.params['x']),'p_excl_2022h1':float(r2.pvalues['x']),'sign_consistent':bool(np.sign(beta)==np.sign(r0.params['x'])==np.sign(r2.params['x']))}

rows=[]
for sh in shocks:
    for yn,y,ex in outs:
        z={'X':f'{sh} × pre-2020 {ex}','shock':sh,'Y':yn,'yvar':y,'exposure':ex}
        try: z.update(fit_pair(sh,y,ex))
        except Exception as e: z['error']=f'{type(e).__name__}: {e}'
        rows.append(z)
r=pd.DataFrame(rows)
# BH-FDR over valid 20 tests
valid=r.p.notna() if 'p' in r else pd.Series(False,index=r.index)
ps=r.loc[valid,'p'].values; order=np.argsort(ps); q=np.empty(len(ps)); prev=1
for j in range(len(ps)-1,-1,-1):
    rank=j+1; val=ps[order[j]]*len(ps)/rank; prev=min(prev,val); q[order[j]]=min(1,prev)
r.loc[valid,'q_fdr']=q
r['score']=0.0
r.loc[valid,'score']=(-np.log10(r.loc[valid,'q_fdr'].clip(lower=1e-12)) + 0.5*r.loc[valid,'sign_consistent'].astype(float) + 0.25*(r.loc[valid,'firms']/r.loc[valid,'firms'].max()))
r['gate']=np.where((r.q_fdr<.05)&(r.sign_consistent==True)&(r.firms>=100)&(r.n>=1000),'READY',np.where(valid,'HOLD','FAIL'))
r=r.sort_values(['gate','score','p'],ascending=[True,False,True])
# sort READY first explicitly
r['gate_ord']=r.gate.map({'READY':0,'HOLD':1,'FAIL':2}); r=r.sort_values(['gate_ord','score','p'],ascending=[True,False,True]).drop(columns='gate_ord')
r.to_csv(OUT/'xy_20_screen_results.csv',index=False,encoding='utf-8-sig')
top=r.head(5).copy(); top.to_csv(OUT/'xy_top5.csv',index=False,encoding='utf-8-sig')
with pd.ExcelWriter(OUT/'VCI_v42_XY_SCREEN_TOP5.xlsx',engine='openpyxl') as w:
    top.to_excel(w,'TOP5',index=False); r.to_excel(w,'ALL_20',index=False); panel[(panel.year>=2020)&(panel.year<=2025)].head(20000).to_excel(w,'PANEL_SAMPLE',index=False)
summary={'candidate_count':int(len(r)),'ready_count':int((r.gate=='READY').sum()),'top5':top[['X','Y','beta','p','q_fdr','firms','n','gate']].replace({np.nan:None}).to_dict('records'),'shock_columns':shock_cols,'panel_rows_2020_2025':int(((panel.year>=2020)&(panel.year<=2025)).sum()),'panel_symbols':int(panel[(panel.year>=2020)&(panel.year<=2025)].symbol.nunique())}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))