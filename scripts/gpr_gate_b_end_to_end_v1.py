from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

RAW_ROOT = Path('data/gpr_bronze_refresh/fundamental_shards')
OUT = Path('data/gpr_gate_b_end_to_end_v1')
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    'n': 13473,
    'firms': 567,
    'quarters': 24,
    'hl67_cut': 0.4721168508083521,
    'Threat_z': {
        'beta': -0.0065111271294339915,
        'twoway_se': 0.002628103871650693,
        'twoway_p': 0.013243372510630458,
        'wcb_p': 0.067,
    },
    'Acts_z': {
        'beta': -0.005889746491737053,
        'twoway_se': 0.0007872093649343036,
        'twoway_p': 7.815970093361102e-14,
        'wcb_p': 0.003,
    },
}


def nt(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def pick_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    for p in patterns:
        for c in df.columns:
            if nt(c) == nt(p) or nt(p) in nt(c):
                return c
    return None


def load_financial_raw() -> pd.DataFrame:
    files = sorted(RAW_ROOT.glob('fundamental_shard_*.parquet'))
    if len(files) != 4:
        raise RuntimeError(f'Expected 4 raw fundamental shards, found {len(files)}')
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw['symbol'] = raw['symbol'].astype(str).str.upper().str.strip()
    raw['period_q'] = raw['period_q'].astype(str)
    raw['value'] = pd.to_numeric(raw['value'], errors='coerce')
    raw = raw.drop_duplicates(['symbol', 'report', 'period_q', 'item_id'], keep='last')
    return raw


def resolve_ids(raw: pd.DataFrame) -> dict[str, str]:
    ids = set(raw['item_id'].dropna().astype(str))

    def resolve(preferred: list[str], tokens: list[str]) -> str | None:
        for x in preferred:
            if x in ids:
                return x
        cand = sorted([x for x in ids if all(t.upper() in x.upper() for t in tokens)])
        return cand[0] if len(cand) == 1 else None

    mapping = {
        'assets': resolve(['BS_TOTAL_ASSETS'], ['TOTAL', 'ASSET']),
        'liabilities': resolve(['BS_TOTAL_LIABILITIES'], ['TOTAL', 'LIABIL']),
        'current_liab': resolve(['BS_SHORT_TERM_LIABILITIES'], ['SHORT', 'LIABIL']),
        'cfo': resolve(['CF_NET_CASH_FLOWS_FROM_OPERATING_ACTIVITIES'], ['OPERATING', 'CASH']),
    }
    if any(v is None for v in mapping.values()):
        raise RuntimeError(f'Could not resolve required item IDs: {mapping}')
    return mapping  # type: ignore[return-value]


def build_financial_panel(raw: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rev = {v: k for k, v in mapping.items()}
    z = raw[raw['item_id'].isin(rev)].copy()
    z['metric'] = z['item_id'].map(rev)
    finraw = z[['symbol', 'report', 'period_q', 'item_id', 'value']].copy()
    finraw.to_csv(OUT / 'financial_raw_relevant.csv', index=False)

    w = z.pivot_table(index=['symbol', 'period_q'], columns='metric', values='value', aggfunc='last').reset_index()
    w[['year', 'q']] = w['period_q'].str.extract(r'(\d{4})Q([1-4])').astype(int)
    w = w.sort_values(['symbol', 'year', 'q'])
    w['pidx'] = w['year'] * 4 + w['q']
    w['avg_assets'] = (w['assets'] + w.groupby('symbol')['assets'].shift(1)) / 2
    w['cfo_assets'] = w['cfo'] / w['avg_assets']
    w['leverage'] = w['liabilities'] / w['assets']
    w['log_assets'] = np.log(w['assets'].where(w['assets'] > 0))
    w['liab_ratio'] = w['current_liab'] / w['assets']
    return w, finraw


def build_highliab(w: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    pre = w[(w['year'] >= 2017) & (w['year'] <= 2019)].copy()
    st = pre.groupby('symbol').agg(
        pre_quarters=('liab_ratio', 'count'),
        baseline_liab_ratio=('liab_ratio', 'median'),
    ).reset_index()
    st = st[st['pre_quarters'] >= 2].copy()
    cut = float(st['baseline_liab_ratio'].quantile(2 / 3))
    st['HL67'] = (st['baseline_liab_ratio'] >= cut).astype(int)
    st.to_csv(OUT / 'baseline_highliab.csv', index=False)
    return st, cut


def load_and_process_gpr() -> tuple[pd.DataFrame, pd.DataFrame]:
    url = 'https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv'
    g = pd.read_csv(url)
    dc = pick_col(g, ['date', 'day'])
    th = pick_col(g, ['THREATS_GPR_AI', 'GPR_THREATS'])
    ac = pick_col(g, ['ACTS_GPR_AI', 'GPR_ACTS'])
    if not all([dc, th, ac]):
        raise RuntimeError(f'Could not resolve GPR columns: {list(g.columns)}')
    g[dc] = pd.to_datetime(g[dc], errors='coerce')
    g = g[(g[dc] >= '2020-01-01') & (g[dc] <= '2025-12-31')].copy()
    g.to_csv(OUT / 'gpr_raw_daily_2020_2025.csv', index=False)

    g2 = g.rename(columns={dc: 'date', th: 'Threat', ac: 'Acts'}).copy()
    g2['period_q'] = g2['date'].dt.to_period('Q').astype(str)
    qg = g2.groupby('period_q').agg(Threat=('Threat', 'mean'), Acts=('Acts', 'mean')).reset_index().sort_values('period_q')
    for c in ['Threat', 'Acts']:
        qg[c + '_z'] = (qg[c] - qg[c].mean()) / qg[c].std(ddof=0)
    qg.to_csv(OUT / 'gpr_quarterly_processed.csv', index=False)
    return g, qg


def build_analysis_panel(w: pd.DataFrame, st: pd.DataFrame, qg: pd.DataFrame) -> pd.DataFrame:
    panel = (
        w[(w['year'] >= 2020) & (w['year'] <= 2025)]
        .merge(st[['symbol', 'HL67']], on='symbol', how='inner')
        .merge(qg, on='period_q', how='left')
    )
    panel.to_parquet(OUT / 'processed_panel_2020_2025.parquet', index=False)
    panel.to_csv(OUT / 'processed_panel_2020_2025.csv', index=False)
    return panel


def prepare(panel: pd.DataFrame, shock: str) -> pd.DataFrame:
    cols = ['symbol', 'pidx', 'period_q', 'cfo_assets', 'log_assets', 'leverage', 'HL67', shock]
    x = panel[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    x['inter'] = x[shock] * x['HL67']
    return x.set_index(['symbol', 'pidx']).sort_index()


def fit_twoway(x: pd.DataFrame):
    mod = PanelOLS(
        x['cfo_assets'],
        x[['inter', 'log_assets', 'leverage']],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    return mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)


def fit_time_cluster(x: pd.DataFrame, y: pd.Series | None = None):
    yy = x['cfo_assets'] if y is None else y
    mod = PanelOLS(
        yy,
        x[['inter', 'log_assets', 'leverage']],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    return mod.fit(cov_type='clustered', cluster_time=True)


def fit_restricted_time(x: pd.DataFrame):
    mod = PanelOLS(
        x['cfo_assets'],
        x[['log_assets', 'leverage']],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    return mod.fit(cov_type='clustered', cluster_time=True)


def vec(obj) -> np.ndarray:
    return np.asarray(obj).reshape(-1)


def wild_cluster_p(x: pd.DataFrame, B: int = 999, seed: int = 20260914) -> dict:
    full = fit_time_cluster(x)
    beta = float(full.params['inter'])
    se = float(full.std_errors['inter'])
    t_obs = beta / se

    r0 = fit_restricted_time(x)
    fitted = vec(r0.fitted_values)
    resid = vec(r0.resids)
    periods = x['period_q'].astype(str).to_numpy()
    uq = np.array(sorted(pd.unique(periods)))
    rng = np.random.default_rng(seed)
    ts: list[float] = []
    failures = 0

    for _ in range(B):
        signs = dict(zip(uq, rng.choice([-1.0, 1.0], size=len(uq))))
        ww = np.array([signs[p] for p in periods])
        ys = pd.Series(fitted + resid * ww, index=x.index, name='ystar')
        try:
            rb = fit_time_cluster(x, ys)
            sb = float(rb.std_errors['inter'])
            if np.isfinite(sb) and sb > 0:
                ts.append(float(rb.params['inter'] / sb))
            else:
                failures += 1
        except Exception:
            failures += 1

    ts_arr = np.asarray(ts)
    p = (1 + int(np.sum(np.abs(ts_arr) >= abs(t_obs)))) / (1 + len(ts_arr))
    return {
        'cluster_time_se': se,
        't_obs': t_obs,
        'B_requested': B,
        'B_success': len(ts),
        'failures': failures,
        'wcb_p': float(p),
    }


def almost_equal(a: float, b: float, atol: float, rtol: float = 1e-9) -> bool:
    return bool(np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=False))


def main() -> None:
    raw = load_financial_raw()
    mapping = resolve_ids(raw)
    w, _ = build_financial_panel(raw, mapping)
    st, cut = build_highliab(w)
    _, qg = load_and_process_gpr()
    panel = build_analysis_panel(w, st, qg)

    rows = []
    gate_pass = True
    for shock in ['Threat_z', 'Acts_z']:
        x = prepare(panel, shock)
        tw = fit_twoway(x)
        wc = wild_cluster_p(x)
        row = {
            'shock': shock,
            'n': len(x),
            'firms': int(x.index.get_level_values(0).nunique()),
            'quarters': int(x['period_q'].nunique()),
            'hl67_cut': cut,
            'beta': float(tw.params['inter']),
            'twoway_se': float(tw.std_errors['inter']),
            'twoway_p': float(tw.pvalues['inter']),
            **wc,
        }
        exp = EXPECTED[shock]
        row['delta_beta'] = row['beta'] - exp['beta']
        row['delta_twoway_se'] = row['twoway_se'] - exp['twoway_se']
        row['delta_twoway_p'] = row['twoway_p'] - exp['twoway_p']
        row['delta_wcb_p'] = row['wcb_p'] - exp['wcb_p']
        row['pass_n'] = row['n'] == EXPECTED['n']
        row['pass_firms'] = row['firms'] == EXPECTED['firms']
        row['pass_quarters'] = row['quarters'] == EXPECTED['quarters']
        row['pass_cut'] = almost_equal(row['hl67_cut'], EXPECTED['hl67_cut'], atol=1e-12)
        row['pass_beta'] = almost_equal(row['beta'], exp['beta'], atol=1e-12)
        row['pass_twoway_se'] = almost_equal(row['twoway_se'], exp['twoway_se'], atol=1e-12)
        row['pass_twoway_p'] = almost_equal(row['twoway_p'], exp['twoway_p'], atol=1e-10)
        row['pass_wcb_p'] = almost_equal(row['wcb_p'], exp['wcb_p'], atol=1e-12, rtol=0.0)
        row['pass_wcb_complete'] = row['B_success'] == 999 and row['failures'] == 0
        row_pass = all(row[k] for k in row if k.startswith('pass_'))
        row['gate_b_row_pass'] = bool(row_pass)
        gate_pass = gate_pass and bool(row_pass)
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'old_vs_rerun_delta.csv', index=False)

    audit = {
        'gate': 'B',
        'status': 'PASS' if gate_pass else 'FAIL',
        'raw_financial_source': 'GitHub Actions artifact gpr-bronze-refresh-raw from run 34582712489',
        'gpr_source': 'https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv',
        'pipeline_scope': 'raw financial shards + raw GPR -> variables -> HighLiab -> panel -> FE -> WCB 999 -> comparison',
        'resolved_item_ids': mapping,
        'expected': EXPECTED,
        'results': results.to_dict(orient='records'),
    }
    (OUT / 'gate_b_status.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(audit, indent=2, allow_nan=False))

    if not gate_pass:
        raise SystemExit('GATE_B_FAIL')


if __name__ == '__main__':
    main()
