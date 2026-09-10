from __future__ import annotations
import json, runpy
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import chi2

# Run the locked v2 pipeline first. It rebuilds the refreshed panel, primary models,
# two-way-clustered event-study coefficients, and all audit files.
ns = runpy.run_path('scripts/gpr_bronze_finalize_v2.py')
panel = ns['panel']
groups = ns['groups']
stat = ns['stat']
res = ns['res']
OUT = Path('data/gpr_bronze_refresh/final')

# v2 used the Moore-Penrose inverse of a two-way clustered covariance matrix for
# the joint lead test. Two-way clustered covariance can be indefinite, which can
# produce an impossible negative Wald chi-square. For the pretrend GATE only,
# use entity-clustered covariance: the firm-cluster sandwich is PSD and gives a
# valid non-negative Wald statistic. Primary coefficients remain two-way clustered.
pre_rows = []
for grp in groups:
    x = panel[['symbol','pidx','event_q','cfo_assets','log_assets','leverage',grp]].replace([np.inf,-np.inf],np.nan).dropna().copy()
    ks = list(range(-8,-1)) + list(range(0,16))
    terms = []
    for k in ks:
        nm = ('lead_m'+str(abs(k))) if k < 0 else ('lag_'+str(k))
        x[nm] = (x.event_q.eq(k).astype(int) * x[grp])
        terms.append(nm)
    x = x.set_index(['symbol','pidx']).sort_index()
    r = PanelOLS(
        x.cfo_assets,
        x[terms + ['log_assets','leverage']],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False,
    ).fit(cov_type='clustered', cluster_entity=True)
    leads = [nm for k,nm in zip(ks,terms) if k < 0 and nm in r.params.index]
    b = r.params[leads].to_numpy(float)
    V = r.cov.loc[leads,leads].to_numpy(float)
    # Symmetrize and project tiny numerical negative eigenvalues to zero.
    V = (V + V.T) / 2
    eigval, eigvec = np.linalg.eigh(V)
    tol = max(float(np.max(np.abs(eigval))) * 1e-10, 1e-14)
    eigval_psd = np.where(eigval > tol, eigval, 0.0)
    Vpsd = (eigvec * eigval_psd) @ eigvec.T
    rank = int(np.sum(eigval_psd > 0))
    if rank == 0:
        w = np.nan; jp = np.nan
    else:
        w = float(b @ np.linalg.pinv(Vpsd, rcond=1e-12) @ b)
        w = max(w, 0.0)
        jp = float(chi2.sf(w, rank))
    pre_rows.append({'group':grp,'lead_terms':len(leads),'cov_rank':rank,'wald_chi2':w,'pretrend_joint_p':jp,'covariance':'entity-clustered PSD'})

pre = pd.DataFrame(pre_rows)
pre.to_csv(OUT/'pretrend_summary_v3.csv', index=False)
pt = dict(zip(pre.group, pre.pretrend_joint_p))
Ntreated = {g:int(stat.loc[stat[g].eq(1),'symbol'].nunique()) for g in groups}

def grab(grp, shock, test='shock_interaction'):
    return res[(res.group.eq(grp)) & (res.shock.eq(shock)) & (res.test.eq(test))].iloc[0]

def gate(r, grp):
    ppre = pt.get(grp, np.nan)
    return 'PASS' if (r.q_fdr_family < 0.05 and np.isfinite(ppre) and ppre >= 0.10 and Ntreated[grp] >= 50) else 'HOLD'

specs = [
    (1,'Threat × HighLiab — anticipation','HighLiab_TopTercile','Threat'),
    (2,'Acts × HighLiab — realization','HighLiab_TopTercile','Acts'),
    (3,'Acts × CashConversionTrap','CashConversionTrap','Acts'),
    (4,'Threat × DoubleExposure + Post2022 DID','DoubleExposure_TopTercile','Threat'),
    (5,'FragileFunding challenger','FragileLowMarginFunding','AI_GPR'),
]
dirs = []
for candidate_order, name, grp, shock in specs:
    r = grab(grp, shock)
    did = grab(grp, 'Post2022', 'DID_Post2022')
    dirs.append({
        'candidate_order':candidate_order,'direction':name,'group':grp,'primary_shock':shock,
        'beta_primary':r.beta,'p_primary':r.p,'q_primary':r.q_fdr_family,
        'did_beta':did.beta,'did_p':did.p,'did_q':did.q_fdr_family,
        'pretrend_joint_p':pt.get(grp,np.nan),'treated_firms':Ntreated[grp],'gate':gate(r,grp)
    })
amm = grab('AssetCommitmentMismatch','AI_GPR')
dirs[-1]['asset_mismatch_q'] = amm.q_fdr_family
dirs[-1]['asset_mismatch_pretrend_p'] = pt.get('AssetCommitmentMismatch',np.nan)
final = pd.DataFrame(dirs)
final['eligible'] = final.gate.eq('PASS').astype(int)
final = final.sort_values(['eligible','q_primary','treated_firms'], ascending=[False,True,False]).reset_index(drop=True)
final['final_rank'] = np.arange(1, len(final)+1)
final['selected'] = final.final_rank.eq(1) & final.eligible.eq(1)
final.to_csv(OUT/'FINAL_DIRECTION_COMPARISON_V3.csv', index=False)
selected = final.loc[final.selected,'direction'].tolist()

prev = json.loads((OUT/'FINAL_QA_AND_SELECTION_V2.json').read_text(encoding='utf-8'))
summary = {
    'baseline_refresh_exact_match': prev.get('baseline_refresh_exact_match'),
    'analysis_firms': int(panel.symbol.nunique()),
    'analysis_rows': int(len(panel)),
    'duplicate_symbol_quarter': int(panel.duplicated(['symbol','period_q']).sum()),
    'selected_direction': selected[0] if selected else None,
    'selection_rule': 'PASS requires BH-FDR q<0.05, valid entity-clustered joint pretrend p>=0.10, and treated firms>=50; rank PASS by q then treated sample. Primary models retain two-way firm+time clustered SE.',
    'pretrend_fix': 'v2 negative Wald statistics were invalid because two-way clustered covariance was indefinite; v3 gate uses entity-clustered PSD covariance and non-negative Wald chi-square.',
    'measurement_note': 'AI_GPR/Threat/Acts are AI-derived geopolitical-risk measures, not firm AI adoption or an AI economic shock.'
}
(OUT/'FINAL_QA_AND_SELECTION_V3.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(pre.to_string(index=False))
print(final.to_string(index=False))
