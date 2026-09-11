# GPR / AI-GPR final evidence scorecard v2

Latest evidence incorporated through commit `4b7ffef` (Threat × DoubleExposure validation) and `fe0fd40` (Threat × HighLiab rescue v3).

## Final decision

**Selected direction: 2 — Acts × HighLiab realization.**

This remains the only direction that combines strong corporate-CFO evidence, supportive DID evidence, clean pretrend diagnostics, stable interpretation, and no new validation failure that overturns the main result. Selection is based on the full evidence pattern, not a single p-value.

## Updated evidence matrix

| Direction | Corporate CFO | Market / linkage | New validation | Final status |
|---|---|---|---|---|
| 1 Threat × HighLiab anticipation | Older specification had signal, but prespecified rescue v3 does not replicate it | Market HAC null; event CAR5 linkage previously significant | Locked pre-treatment HighLiab primary: beta = 0.0001680058, p = 0.9440209, q = 0.9440209; upgrade gate FAIL | **REJECT as core thesis** |
| 2 Acts × HighLiab realization | Primary beta = -0.0058897465, q = 1.563e-12; DID beta = -0.0097226260, q = 0.0020045823; pretrend p = 0.7060769384; 190 treated firms | Full-history market HAC q = 0.5478189316; market→future CFO null | No new robustness failure identified | **SELECTED — main thesis** |
| 3 Acts × CashConversionTrap | Primary q = 0.0260664389; DID q = 0.0422380057; pretrend p = 0.3535971586 | Winsor linkage signal but reverse-timing placebo p = 0.0268269319 | Placebo concern remains | **HOLD / secondary only** |
| 4 Threat × DoubleExposure | Corporate primary q = 0.0564054269; DID q = 0.0396379465; pretrend p = 0.7635935340 | CAR5→future CFO beta = -0.1321924268, post-window BH-FDR q = 2.6045832e-11; reverse-timing p = 0.5108455372 | 15/15 leave-one-event-out CAR5 coefficients remain negative, but pre5 placebo beta = 0.1256443539, p = 0.0131708954; CAR1 q = 0.2285822136 and CAR20 q = 0.1735772440; upgrade gate FAIL | **Strong event-specific mechanism, not main causal thesis** |
| 5 AssetCommitmentMismatch / FragileFunding | FragileFunding corporate specification previously PASS; AssetCommitmentMismatch unstable | Market HAC null; AssetCommitmentMismatch event CAR5 q = 0.0451193497 | Measurement/mechanism remains less stable | **Challenger only** |

## Updated analytical scorecard

Scores are analytical judgments on a 0–10 scale, not statistical tests.

| Direction | Empirical robustness | Novelty | Theory / mechanism | Identification | Predictive value | Application / policy / investment | Measurement credibility | Publishability | Total / 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Threat × HighLiab | 4 | 8 | 8 | 5 | 6 | 8 | 8 | 5 | 52 |
| **2 Acts × HighLiab** | **10** | 8 | **10** | **9** | 6 | **9** | **9** | **9** | **70** |
| 3 Acts × CashConversionTrap | 6 | 8 | 8 | 5 | 5 | 8 | 7 | 6 | 53 |
| 4 Threat × DoubleExposure | 7 | **10** | 9 | 5 | **10** | 9 | 7 | 7 | 64 |
| 5 Asset / FragileFunding | 6 | 8 | 8 | 7 | 5 | 8 | 6 | 6 | 54 |

## QA and interpretation lock

- Price universe: 578 symbols, 02/01/2020–31/12/2025.
- CFO panel: 578 symbols, 2020Q1–2025Q4, zero duplicate symbol-quarter keys.
- Threat × DoubleExposure validation uses 15 Threat events, 101 treated symbols and 477 controls.
- Core CFO missingness is low: CFO/assets about 0.95%; next-quarter CFO about 5.11%; log assets/leverage about 0.007%.
- `vci578_preexposure.csv` is not a 578-firm estimation sample; it contains 378 pre-treatment-exposure firms, and the latest Threat × HighLiab analysis retains 325 firms after merge/availability filters. The 578-symbol behavioral/linkage cohort is a separate downstream cohort.
- Multiple testing is controlled with BH-FDR for the prespecified post-event windows where reported.
- Threat × DoubleExposure is not driven by one single Threat event: all 15 leave-one-event-out CAR5 coefficients remain negative. However, the significant pre5 placebo means anticipation/confounding cannot be ruled out, so the event linkage cannot be upgraded to a clean causal channel.
- Full-history market HAC tests remain non-significant after FDR across the five directions; market ranking alone does not identify a market-reaction mechanism.
- Survivorship/selection risk remains because analysis membership depends on listing/data availability and exposure/group construction. This must be disclosed in the thesis.
- Market reaction → future CFO is predictive/associational only. Do not use causal terms such as self-fulfilling expectations, investor pressure, or behavioral feedback without an additional exogenous identification strategy.

## Thesis claim permitted by the current evidence

The strongest defensible main claim is that **realized geopolitical-risk acts are associated with a larger subsequent deterioration in operating cash-flow outcomes among firms with high pre-treatment liability exposure**, supported by strong corporate-panel evidence, a supportive Post-2022 DID specification, and no detected pretrend failure.

`Threat × DoubleExposure` should be retained as the main extension: its 5-day event reaction strongly predicts weaker future CFO and is robust to leave-one-event-out checks, but the significant pre-event placebo prevents a clean causal anticipation interpretation.
