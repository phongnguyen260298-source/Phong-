# GPR / AI-GPR final evidence scorecard v1

Checkpoint source commit before this lock: `3c6cb60566bfda3d1edbf27a75b23e624e23de5f`.

## Decision

**Selected direction: 2 — Acts × HighLiab realization.**

This is the strongest thesis direction after combining the locked corporate-CFO evidence, full-history market tests, and market-reaction-to-future-CFO linkage. The selection is based on the full evidence pattern, not on a single p-value.

## Evidence matrix

| Direction | Corporate CFO | Full-history market | Market → future CFO | Identification / placebo | Final assessment |
|---|---|---|---|---|---|
| 1 Threat × HighLiab anticipation | PASS; primary beta -0.0065111271294339915, q 0.02037441924712378; DID q 0.0020045822583181216; pretrend joint p 0.7060769383667109 | HAC q 0.9720372844915885 | event CAR5 interaction beta -0.11769573278968325, q 0.0017685600907277887; primary linkage q 0.09183126394998808 | reverse-timing linkage placebo p 0.12672008715892713 | Strong secondary direction; predictive event evidence but weaker primary corporate effect than direction 2 |
| 2 Acts × HighLiab realization | **PASS; primary beta -0.005889746491737053, q 1.5631940186722204e-12; DID beta -0.009722626045573425, q 0.0020045822583181216; pretrend joint p 0.7060769383667109; 190 treated firms** | HAC beta -0.00013474200097234734, q 0.5478189315774706 | primary q 0.4251957988944297; event CAR5 q 0.6528354774759029 | reverse-timing linkage placebo p 0.9395728119531956 | **Selected. Most robust corporate-side identification and clearest realization mechanism; market/linkage nulls limit claims about an investor-feedback channel but do not invalidate the corporate result** |
| 3 Acts × CashConversionTrap | PASS; primary q 0.026066438877469622; DID q 0.04223800569099628; pretrend joint p 0.353597158577343 | HAC q 0.7616089681623229 | winsor interaction p 0.016663155466278612, q 0.09997893279767167 | reverse-timing placebo interaction p 0.026826931850910674 | Downgraded because a reverse-timing placebo is also significant |
| 4 Threat × DoubleExposure / Post2022 | HOLD; primary q 0.05640542691673023; DID q 0.039637946534659685; pretrend joint p 0.7635935339806172; 99 treated firms | highest market empirical score, but HAC q 0.5478189315774706 | **event CAR5 interaction beta -0.13219242681879378, q 5.2091664315412345e-11; treated total p 0.00010012826317029178**; primary linkage q 0.3861680704917807 | reverse-timing event-linkage placebo p 0.6115811850738795 | Important mechanism/challenger. Evidence is event-specific predictive, not sufficient for a general causal behavioral-feedback claim |
| 5 AssetCommitmentMismatch / FragileFunding challenger | FragileFunding corporate specification PASS and ranks second in corporate v3; AssetCommitmentMismatch not supported as a stable corporate effect | market HAC not FDR-significant | AssetCommitmentMismatch event CAR5 q 0.04511934968426079; other primary/winsor evidence weak | placebos generally null | Challenger only; measurement/mechanism less stable across specifications |

## Scorecard

Scores are analytical judgments on a 0–10 scale based on the evidence above. They are not additional statistical tests.

| Direction | Empirical robustness | Novelty | Theory / mechanism | Identification | Predictive value | Application / policy / investment | Measurement credibility | Publishability | Total / 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Threat × HighLiab | 8 | 8 | 9 | 8 | 8 | 8 | 9 | 8 | 66 |
| **2 Acts × HighLiab** | **10** | 8 | **10** | **9** | 6 | **9** | **9** | **9** | **70** |
| 3 Acts × CashConversionTrap | 6 | 8 | 8 | 5 | 5 | 8 | 7 | 6 | 53 |
| 4 Threat × DoubleExposure | 7 | **10** | 9 | 6 | **10** | 9 | 7 | 8 | 66 |
| 5 Asset / FragileFunding | 6 | 8 | 8 | 7 | 5 | 8 | 6 | 6 | 54 |

## Mandatory QA interpretation

- Full-history market prices cover 2020-01-02 to 2025-12-31. The linkage QA reports 578 market symbols, 578 group-file symbols, 578 CFO symbols, 13,792 symbol-quarter rows, and zero duplicate symbol-quarter keys.
- Missingness in the linkage panel is low for core accounting controls (CFO lag about 5.02%, CFO assets about 0.96%, CFO next about 5.09%); event-specific market variables are intentionally sparse because they only exist around selected events.
- Full-history market HAC tests for all five directions fail BH-FDR significance. Market ranking therefore cannot by itself identify a market-reaction channel.
- `vci578_preexposure.csv` must not be described as a 578-firm estimation sample. It contains the older pre-treatment-exposure subset (378 firms). Cohort metadata separately reports modern anchor = 586 firms and union = 614 firms. The 578-firm linkage/group cohort is a different downstream cohort and must be described separately.
- The cohort metadata states that anchor cohorts use current/known HOSE-HNX exchange metadata and listing intervals, transfer-history cases are flagged for audit, and there is no current-survivor requirement. Residual selection/look-ahead risk remains because group formation and availability filters can still condition sample membership; this must be disclosed.
- Multiple testing is controlled using BH-FDR where reported. Direction selection must rely on consistency across primary, DID/event, pretrend/placebo and robustness specifications, not nominal p-values alone.
- Direction 3 is penalized because the reverse-timing placebo is significant.
- Direction 4 has a clean reverse-timing placebo for its strongest event-CAR5 linkage, but primary and winsor linkage specifications are null; therefore describe this as event-specific predictive evidence only.
- Do not call market reaction → future CFO, self-fulfilling expectations, behavioral feedback, or investor pressure causal without an additional identification strategy that isolates exogenous market belief/reaction from common shocks and firm fundamentals.

## Thesis claim permitted by current evidence

The defensible main claim is that **realized geopolitical-risk acts are associated with a larger subsequent deterioration in operating cash-flow outcomes among firms with high pre-treatment liability exposure**, with strong corporate-panel robustness, a supportive Post-2022 DID specification, and no detected pretrend failure under the corrected clustered-covariance test.

The current evidence does **not** support a claim that equity-market reactions cause later CFO deterioration. `Threat × DoubleExposure` may be retained as a mechanism/predictive extension and tested further under stronger identification.
