from __future__ import annotations
import json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import t as tdist, chi2

START = '2020-01-01'
END = '2025-12-31'
OUT = Path('data/gpr_market_top5')
OUT.mkdir(parents=True, exist_ok=True)
GROUP_FILE = 'data/research_cohorts/vci578_behavioral_groups.csv'
VND_URL = 'https://finfo-api.vndirect.com.vn/v4/stock_prices/'

G = pd.read_csv(GROUP_FILE)
GROUPS = ['HighLiab_TopTercile','CashConversionTrap','DoubleExposure_TopTercile','FragileLowMarginFunding','AssetCommitmentMismatch']
G['symbol'] = G['symbol'].astype(str).str.upper().str.strip()
for c in GROUPS:
    G[c] = pd.to_numeric(G[c], errors='coerce').fillna(0).astype(int)
SYMBOLS = sorted(G['symbol'].dropna().unique())

session = requests.Session()
session.headers.update({'User-Agent':'Mozilla/5.0 academic-research/1.0','Accept':'application/json'})

def fetch_one(sym):
    params = {
        'sort':'date', 'size':5000, 'page':1,
        'q':f'code:{sym}~date:gte:{START}~date:lte:{END}'
    }
    last = None
    for k in range(5):
        try:
            r = session.get(VND_URL, params=params, timeout=45)
            r.raise_for_status()
            js = r.json()
            data = js.get('data', []) if isinstance(js, dict) else []
            if not data:
                return sym, None, 'EMPTY'
            d = pd.DataFrame(data)
            dc = next((c for c in ['date','tradingDate','trading_date'] if c in d.columns), None)
            cc = next((c for c in ['adClose','close','closePrice'] if c in d.columns), None)
            vc = next((c for c in ['nmVolume','volume','totalVolume'] if c in d.columns), None)
            if dc is None or cc is None:
                return sym, None, f'SCHEMA:{list(d.columns)}'
            z = pd.DataFrame({
                'symbol': sym,
                'date': pd.to_datetime(d[dc], errors='coerce'),
                'close': pd.to_numeric(d[cc], errors='coerce'),
                'volume': pd.to_numeric(d[vc], errors='coerce') if vc else np.nan,
            }).dropna(subset=['date','close']).sort_values('date')
            z = z[(z.date >= START) & (z.date <= END)]
            return (sym, z, None) if len(z) else (sym, None, 'EMPTY_AFTER_FILTER')
        except Exception as e:
            last = e
            time.sleep(min(12, 1.5 * (2 ** k)))
    return sym, None, f'{type(last).__name__}:{last}'

frames, errors = [], []
with ThreadPoolExecutor(max_workers=4) as pool:
    futs = {pool.submit(fetch_one, s): s for s in SYMBOLS}
    for i, fut in enumerate(as_completed(futs), 1):
        sym, z, err = fut.result()
        if z is not None:
            frames.append(z)
        else:
            errors.append({'symbol':sym,'error':err})
        if i % 50 == 0:
            print(f'price {i}/{len(SYMBOLS)} ok={len(frames)} errors={len(errors)}', flush=True)
        time.sleep(0.03)

pd.DataFrame(errors).to_csv(OUT/'price_errors.csv', index=False)
if not frames:
    raise RuntimeError('No price data from VNDirect public API')
px = pd.concat(frames, ignore_index=True).sort_values(['symbol','date'])
px['ret'] = px.groupby('symbol')['close'].pct_change()
# Equal-weight median market proxy is deliberately robust to extreme small-cap moves.
mkt = px.groupby('date')['ret'].median().rename('mkt_ret')
px = px.merge(mkt, on='date', how='left')
px['ar'] = px['ret'] - px['mkt_ret']
px['logvol'] = np.log1p(px['volume'])
px['logvol_med60'] = px.groupby('symbol')['logvol'].transform(lambda s: s.shift(1).rolling(60, min_periods=20).median())
px['abn_logvol'] = px['logvol'] - px['logvol_med60']
px.to_parquet(OUT/'price_daily_2020_2025.parquet', index=False, compression='zstd')

# Official AI-GPR daily dataset.
GPR_URL = 'https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv'
r = requests.get(GPR_URL, timeout=90, headers={'User-Agent':'Mozilla/5.0 academic-research/1.0'})
r.raise_for_status()
(OUT/'ai_gpr_data_daily.csv').write_bytes(r.content)
g = pd.read_csv(OUT/'ai_gpr_data_daily.csv')

def pick(cols, pats):
    norm = {re.sub(r'[^a-z0-9]','',str(c).lower()): c for c in cols}
    for p in pats:
        pp = re.sub(r'[^a-z0-9]','',p.lower())
        for n,c in norm.items():
            if n == pp or pp in n:
                return c
    return None

dc = pick(g.columns, ['date','day'])
ai = pick(g.columns, ['GPR_AI','AI_GPR','aigpr'])
orig = pick(g.columns, ['GPR_AER','original'])
thr = pick(g.columns, ['GPR_THREATS','threats','gprt'])
act = pick(g.columns, ['GPR_ACTS','acts','gpra'])
(OUT/'gpr_schema.json').write_text(json.dumps({'columns':list(map(str,g.columns)),'date':dc,'ai':ai,'original':orig,'threats':thr,'acts':act},ensure_ascii=False,indent=2),encoding='utf-8')
if not all([dc, ai, thr, act]):
    raise RuntimeError(f'GPR schema unresolved: {list(g.columns)}')
rename = {dc:'date', ai:'AI_GPR', thr:'THREATS', act:'ACTS'}
if orig: rename[orig] = 'GPR_ORIG'
g = g.rename(columns=rename)
g['date'] = pd.to_datetime(g['date'], errors='coerce')
g = g[(g.date >= pd.Timestamp(START)-pd.Timedelta(days=10)) & (g.date <= END)].sort_values('date').copy()
SHOCKS = ['AI_GPR','THREATS','ACTS'] + (['GPR_ORIG'] if 'GPR_ORIG' in g else [])
for c in SHOCKS:
    g[c] = pd.to_numeric(g[c], errors='coerce')
    g[c+'_innov'] = np.log1p(g[c].clip(lower=0)).diff()

# Map calendar-day news to the next Vietnam trading day, preserving weekend news.
trading = pd.DatetimeIndex(sorted(px['date'].dropna().unique()))
maprows = []
for _, rr in g.dropna(subset=['date']).iterrows():
    j = trading.searchsorted(rr['date'])
    if j >= len(trading):
        continue
    row = {'trade_date':trading[j]}
    for c in SHOCKS:
        row[c+'_innov'] = rr[c+'_innov']
        row[c+'_level'] = rr[c]
    maprows.append(row)
mg = pd.DataFrame(maprows)
agg = {c+'_innov':'sum' for c in SHOCKS} | {c+'_level':'last' for c in SHOCKS}
mg = mg.groupby('trade_date', as_index=False).agg(agg).sort_values('trade_date')
for c in SHOCKS:
    s = mg[c+'_innov'].replace([np.inf,-np.inf],np.nan)
    sd = s.std(ddof=0)
    mg[c+'_z'] = (s-s.mean())/sd if np.isfinite(sd) and sd > 0 else np.nan
mg.to_csv(OUT/'gpr_mapped_trading_daily.csv', index=False)

# Daily treated-control abnormal-return spreads.
p = px.merge(G[['symbol']+GROUPS], on='symbol', how='inner')
spread_parts = []
for grp in GROUPS:
    tmp = p.dropna(subset=['ar']).groupby(['date',grp])['ar'].mean().unstack()
    if 0 in tmp.columns and 1 in tmp.columns:
        s = (tmp[1]-tmp[0]).rename('spread').reset_index()
        s['group'] = grp
        spread_parts.append(s)
spreads = pd.concat(spread_parts, ignore_index=True) if spread_parts else pd.DataFrame()
spreads.to_csv(OUT/'daily_group_spreads.csv', index=False)

regrows = []
for grp in GROUPS:
    base = spreads[spreads.group == grp][['date','spread']].merge(mg.rename(columns={'trade_date':'date'}), on='date', how='inner')
    for shock in [c+'_z' for c in SHOCKS]:
        z = base[['spread',shock]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z) < 60 or z[shock].std(ddof=0) == 0:
            regrows.append({'group':grp,'shock':shock,'beta':np.nan,'p':np.nan,'n':len(z),'status':'INSUFFICIENT'})
            continue
        fit = sm.OLS(z['spread'], sm.add_constant(z[[shock]], has_constant='add')).fit(cov_type='HAC', cov_kwds={'maxlags':5})
        regrows.append({'group':grp,'shock':shock,'beta':float(fit.params[shock]),'p':float(fit.pvalues[shock]),'n':len(z),'status':'OK'})
reg = pd.DataFrame(regrows)
reg.to_csv(OUT/'group_spread_regressions.csv', index=False)

# Top 20 non-overlapping Threat innovation trading days.
events = []
for _, rr in mg.dropna(subset=['THREATS_z']).sort_values('THREATS_z', ascending=False).iterrows():
    d = pd.Timestamp(rr['trade_date'])
    if all(abs((d-e).days) >= 10 for e in events):
        events.append(d)
    if len(events) >= 20:
        break
pd.DataFrame({'event_trade_date':events}).to_csv(OUT/'top20_threat_event_dates.csv', index=False)

# Firm-event metrics. belief_pressure combines negative price pressure and abnormal attention.
rows = []
by_symbol = {s:z.set_index('date').sort_index() for s,z in p.groupby('symbol')}
group_lookup = G.set_index('symbol')[GROUPS]
for ed in events:
    idx = trading.get_indexer([ed])[0]
    if idx < 0: continue
    prewin = trading[max(0,idx-5):idx]
    for sym,zs in by_symbol.items():
        row = {'symbol':sym,'event_trade_date':ed}
        valid = False
        for h in [1,5,20]:
            win = trading[idx:min(len(trading),idx+h)]
            car = zs.reindex(win)['ar'].sum(min_count=1)
            row[f'car_{h}'] = car
            valid = valid or np.isfinite(car)
        row['pre_car_5'] = zs.reindex(prewin)['ar'].sum(min_count=1) if len(prewin) else np.nan
        win5 = trading[idx:min(len(trading),idx+5)]
        av = zs.reindex(win5)['abn_logvol']
        row['abn_volume_5'] = av.mean() if av.notna().any() else np.nan
        if np.isfinite(row['car_5']) and np.isfinite(row['abn_volume_5']):
            row['belief_pressure_5'] = max(0.0,-row['car_5']) * max(0.0,row['abn_volume_5'])
        else:
            row['belief_pressure_5'] = np.nan
        if sym in group_lookup.index:
            for c in GROUPS: row[c] = int(group_lookup.loc[sym,c])
        if valid: rows.append(row)
ef = pd.DataFrame(rows)
ef.to_csv(OUT/'event_firm_metrics.csv', index=False)

# Event-by-event Welch comparisons; later combine evidence across events.
eventrows = []
for grp in GROUPS:
    for metric in ['pre_car_5','car_1','car_5','car_20','abn_volume_5','belief_pressure_5']:
        for ed, ze in ef.dropna(subset=[grp,metric]).groupby('event_trade_date'):
            a = ze.loc[ze[grp]==1,metric]; b = ze.loc[ze[grp]==0,metric]
            if len(a) < 8 or len(b) < 20: continue
            va, vb = a.var(ddof=1), b.var(ddof=1)
            diff = a.mean()-b.mean(); se = np.sqrt(va/len(a)+vb/len(b))
            pv = np.nan
            if np.isfinite(se) and se > 0:
                tt = diff/se
                den = (va/len(a))**2/(len(a)-1) + (vb/len(b))**2/(len(b)-1)
                dfw = (va/len(a)+vb/len(b))**2/den if den > 0 else np.nan
                pv = 2*tdist.sf(abs(tt),dfw) if np.isfinite(dfw) else np.nan
            eventrows.append({'group':grp,'metric':metric,'event_trade_date':ed,'treated_mean':a.mean(),'control_mean':b.mean(),'diff':diff,'p':pv,'n_t':len(a),'n_c':len(b)})
ev = pd.DataFrame(eventrows)
ev.to_csv(OUT/'event_group_metrics.csv', index=False)

pool = []
for (grp,metric), z in ev.groupby(['group','metric']):
    pp = z['p'].dropna().clip(1e-12,1)
    fisher = float(chi2.sf(-2*np.log(pp).sum(),2*len(pp))) if len(pp) else np.nan
    pool.append({'group':grp,'metric':metric,'mean_diff':z['diff'].mean(),'median_diff':z['diff'].median(),'events':len(z),'share_negative':(z['diff']<0).mean(),'fisher_p':fisher})
pool = pd.DataFrame(pool)
pool.to_csv(OUT/'event_group_summary.csv', index=False)

CANDS = [
    ('D1_Threat_HighLiab','HighLiab_TopTercile','THREATS_z'),
    ('D2_Acts_HighLiab','HighLiab_TopTercile','ACTS_z'),
    ('D3_Acts_CashConversionTrap','CashConversionTrap','ACTS_z'),
    ('D4_Threat_DoubleExposure','DoubleExposure_TopTercile','THREATS_z'),
    ('D5_Threat_FragileLowMargin','FragileLowMarginFunding','THREATS_z'),
]
rank = []
for name,grp,shock in CANDS:
    rr = reg[(reg.group==grp)&(reg.shock==shock)]
    if rr.empty: continue
    rr = rr.iloc[0]
    def get(metric,col):
        z = pool[(pool.group==grp)&(pool.metric==metric)]
        return float(z.iloc[0][col]) if len(z) else np.nan
    rank.append({
        'direction':name,'group':grp,'shock':shock,
        'beta_daily_spread':rr.beta,'p_hac':rr.p,'n_daily':rr.n,
        'event5_mean_diff':get('car_5','mean_diff'),'event5_fisher_p':get('car_5','fisher_p'),'event5_share_negative':get('car_5','share_negative'),
        'pre5_mean_diff':get('pre_car_5','mean_diff'),'pre5_fisher_p':get('pre_car_5','fisher_p'),
        'attention_diff':get('abn_volume_5','mean_diff'),'attention_fisher_p':get('abn_volume_5','fisher_p'),
        'belief_pressure_diff':get('belief_pressure_5','mean_diff'),'belief_pressure_fisher_p':get('belief_pressure_5','fisher_p'),
    })
rank = pd.DataFrame(rank)
rank.to_csv(OUT/'D1_D5_market_screen.csv', index=False)

summary = {
    'price_source':'VNDirect public finfo API',
    'symbols_target':len(SYMBOLS),'symbols_price_ok':int(px.symbol.nunique()),'price_rows':int(len(px)),
    'gpr_schema':{'ai':ai,'original':orig,'threats':thr,'acts':act},'events_n':len(events),
    'directions':rank.replace({np.nan:None}).to_dict(orient='records')
}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
