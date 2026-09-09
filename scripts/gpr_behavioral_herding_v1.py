from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT=Path('data/gpr_market_top5')
px=pd.read_parquet(OUT/'price_daily_2020_2025.parquet')
mg=pd.read_csv(OUT/'gpr_mapped_trading_daily.csv',parse_dates=['trade_date'])
groups=pd.read_csv('data/research_cohorts/vci578_behavioral_groups.csv')
GROUPS=['HighLiab_TopTercile','CashConversionTrap','DoubleExposure_TopTercile','FragileLowMarginFunding','AssetCommitmentMismatch']

px['date']=pd.to_datetime(px['date'])
# CCK-style daily market dispersion using equal-weight mean return.
daily=px.dropna(subset=['ret']).groupby('date').agg(rm=('ret','mean'), n=('ret','size')).reset_index()
tmp=px.dropna(subset=['ret']).merge(daily[['date','rm']],on='date',how='left')
csad=tmp.assign(dev=lambda x:(x['ret']-x['rm']).abs()).groupby('date')['dev'].mean().rename('csad').reset_index()
daily=daily.merge(csad,on='date').merge(mg.rename(columns={'trade_date':'date'}),on='date',how='left')
daily['abs_rm']=daily['rm'].abs(); daily['rm_sq']=daily['rm']**2; daily['down']=(daily['rm']<0).astype(float)

def hac_fit(df, shock, down_only=False):
    d=df.copy()
    if down_only: d=d[d['rm']<0].copy()
    d['shock_rm_sq']=d[shock]*d['rm_sq']
    d=d[['csad','abs_rm','rm_sq',shock,'shock_rm_sq']].replace([np.inf,-np.inf],np.nan).dropna()
    if len(d)<100: return None
    X=sm.add_constant(d[['abs_rm','rm_sq',shock,'shock_rm_sq']],has_constant='add')
    fit=sm.OLS(d['csad'],X).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    return {
        'n':len(d),'beta_rm_sq':float(fit.params['rm_sq']),'p_rm_sq':float(fit.pvalues['rm_sq']),
        'beta_shock':float(fit.params[shock]),'p_shock':float(fit.pvalues[shock]),
        'beta_shock_x_rm_sq':float(fit.params['shock_rm_sq']),'p_shock_x_rm_sq':float(fit.pvalues['shock_rm_sq'])
    }

rows=[]
for shock in ['THREATS_z','ACTS_z','AI_GPR_z'] + (['GPR_ORIG_z'] if 'GPR_ORIG_z' in daily.columns else []):
    for state in ['all','down']:
        r=hac_fit(daily,shock,state=='down')
        if r: r.update(scope='market',group='ALL',shock=shock,state=state); rows.append(r)

# Group-specific CSAD: does GPR make vulnerable firms move more as a crowd than controls?
p=px.merge(groups[['symbol']+GROUPS],on='symbol',how='inner')
for grp in GROUPS:
    for val,label in [(1,'treated'),(0,'control')]:
        z=p[p[grp]==val].dropna(subset=['ret']).copy()
        dm=z.groupby('date').agg(rm=('ret','mean'),n=('ret','size')).reset_index()
        zz=z.merge(dm[['date','rm']],on='date',how='left')
        dc=zz.assign(dev=lambda x:(x['ret']-x['rm']).abs()).groupby('date')['dev'].mean().rename('csad').reset_index()
        dm=dm.merge(dc,on='date').merge(mg.rename(columns={'trade_date':'date'}),on='date',how='left')
        dm['abs_rm']=dm['rm'].abs(); dm['rm_sq']=dm['rm']**2
        for shock in ['THREATS_z','ACTS_z']:
            for state in ['all','down']:
                r=hac_fit(dm,shock,state=='down')
                if r: r.update(scope='group',group=grp,side=label,shock=shock,state=state); rows.append(r)
herd=pd.DataFrame(rows)
herd.to_csv(OUT/'behavioral_herding_csad.csv',index=False)

# Difference-in-coefficient summary: more negative shock×Rm^2 => stronger herding under GPR.
comp=[]
for grp in GROUPS:
    for shock in ['THREATS_z','ACTS_z']:
        for state in ['all','down']:
            a=herd[(herd.scope=='group')&(herd.group==grp)&(herd.side=='treated')&(herd.shock==shock)&(herd.state==state)]
            b=herd[(herd.scope=='group')&(herd.group==grp)&(herd.side=='control')&(herd.shock==shock)&(herd.state==state)]
            if len(a) and len(b):
                comp.append({'group':grp,'shock':shock,'state':state,
                    'treated_beta':float(a.iloc[0].beta_shock_x_rm_sq),'treated_p':float(a.iloc[0].p_shock_x_rm_sq),
                    'control_beta':float(b.iloc[0].beta_shock_x_rm_sq),'control_p':float(b.iloc[0].p_shock_x_rm_sq),
                    'beta_gap_treated_minus_control':float(a.iloc[0].beta_shock_x_rm_sq-b.iloc[0].beta_shock_x_rm_sq)})
pd.DataFrame(comp).to_csv(OUT/'behavioral_herding_group_compare.csv',index=False)

# Short-run reversal test. Negative initial CAR followed by positive days 6-20 is consistent with overreaction/correction.
ef=pd.read_csv(OUT/'event_firm_metrics.csv',parse_dates=['event_trade_date'])
ef['car_6_20']=ef['car_20']-ef['car_5']
rev=[]
for grp in GROUPS:
    for side,label in [(1,'treated'),(0,'control')]:
        z=ef[ef[grp]==side]
        rev.append({'group':grp,'side':label,'n_firm_events':len(z),
            'mean_car1':z['car_1'].mean(),'mean_car5':z['car_5'].mean(),'mean_car20':z['car_20'].mean(),
            'mean_car6_20':z['car_6_20'].mean(),'share_initial_negative':(z['car_5']<0).mean(),
            'share_reversal_after_negative':((z['car_5']<0)&(z['car_6_20']>0)).mean(),
            'mean_abn_volume5':z['abn_volume_5'].mean(),'mean_belief_pressure5':z['belief_pressure_5'].mean()})
rev=pd.DataFrame(rev)
rev.to_csv(OUT/'behavioral_event_reversal.csv',index=False)

summary={
 'interpretation_rule': 'Negative shock_x_rm_sq indicates GPR-associated herding in CCK-CSAD framework; negative CAR with abnormal volume and subsequent reversal supports behavioral overreaction more than pure fundamentals; persistent CAR plus later CFO deterioration supports rational anticipation or market-feedback channel.',
 'market_herding': herd[herd.scope=='market'].replace({np.nan:None}).to_dict(orient='records'),
 'group_compare': comp,
 'reversal': rev.replace({np.nan:None}).to_dict(orient='records')
}
(OUT/'behavioral_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
