# Experiment 04: Feature Discovery, Normalization, and Forensics

## 1. Key Research Reports
- `FEATURE_DISCOVERY_REPORT.md`: 11 Hypotheses tested against held-out match data.
  * **Key Discovery**: Bench Depth (`bench_value`) improves log-loss by -0.00319 nats ($t = +3.63, \text{AUC} = 0.589$).
  * **Key Discovery**: RAPM is redundant/harmful when dynamic team ratings ($\alpha_i, \beta_j$) are explicit (+0.0108 nats).
  * **Key Discovery**: Slower exponential form kernels ($t_{1/2} \approx 16$ matches) beat short 8-match discrete windows.
- `PLAYER_NORMALISATION_REPORT.md`: 9 Player rating normalization strategies benchmarked across 8,192 matches.
  * Individual player exposure shrinkage ($n / (n + 20)$) + Starting-XI sum preserves true team strength.
- `PXG_RAPM_EDA_REPORT.md`: Cross-league validation against official Opta/SofaScore xG (8,622 matches, $r = 0.835$).
- `AGE_WEALTH_FINDINGS.md`: Age and market valuation demographic structure.

## 2. Forensic & Benchmarking Scripts
- `r50_rating_structure.jl`, `r51_rapm_normalisation_bench.jl`: Rating structure and paired bootstrap tests.
- `r60_form_kernel_forensics.jl`, `r61_squad_dynamics_forensics.jl`, `r62_feature_gauntlet.jl`: Feature discovery gauntlet.
- `r92_pxg_vs_sofascore_xg_all_leagues.jl`: Cross-league xG correlation.
- `r93_feature_synergy_and_correlations.jl`: Feature synergy matrix.
- `r94_pxg_rapm_forensics.jl`, `r95_pxg_model_forensics.jl`: Model forensics and decay sweeps.
