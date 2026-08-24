# Findings — issue 05 league indexing

## Execution status

Validated on `mcmc-beast` at commit `2dfbd63`, using saved artifact
`recomb_pxg_wealth_integrated_hl365_hs2_20260823_075833`. No MCMC was run.

## Hypothesis under test

For a one-column l05 saved artifact, current DataFrame prediction maps tournament 56 to `delta_league[1]` but maps 57 to requested column 2 and then silently uses zero. The compatible interpretation of that artifact pools both ScottishLower tournaments into column 1.

## Observed evidence

Selected fold 38 contained both leagues: 10 tournament-56 and 5 tournament-57 fixtures.
The chain had exactly one fitted league column. Issue-01 name mapping and issue-02 tau scaling
were active in both comparison arms; one genuinely unseen side (`east-kilbride`) was reported.

```text
delta_league[1]: q05=-0.1612, median=-0.0079, mean=-0.0083, q95=0.1519
exp(delta):       q05= 0.8511, median= 0.9922, mean= 0.9963, q95=1.1640
base_mu+delta:    q05= 0.0268, median= 0.0738, mean= 0.0732, q95=0.1189
```

All assertions passed:

- tournament 56 was exactly unchanged draw-wise;
- tournament 57 candidate/current open-rate ratio equalled `exp(delta_league[1])` draw-wise;
- all score grids normalized;
- unknown tournaments were explicitly rejected.

For the five tournament-57 fixtures, posterior-mean market changes (pooled minus legacy) were:

```text
home win: -0.00120 to -0.00032
draw:     +0.00074 to +0.00092
BTTS:     -0.00448 to -0.00398
over 2.5: -0.00620 to -0.00529
```

The mean draw-wise multiplier was `0.99627`; posterior-mean fixture rates moved somewhat more
because `delta_league[1]` is correlated with the remaining posterior rate terms.

## Cumulative permanent-patch validation

At commit `324b227`, the saved-chain validator composed all three corrections:

```text
l03 NegBin: 22 known OOS sides activated; mapping/league/tau passed;
              full route blocked by independent referee artifact drift
l04 wealth:  22 known OOS sides activated; mapping/league/tau and full route passed
l05 pxG:     22 known OOS sides activated; mapping/league/tau and full route passed
l05 parity:  tournament 56 stable; tournament 57 exactly matched the pooled candidate
```

The l03 warning was quantified as one fitted referee column versus 57 refs in the reconstructed
FeatureSet. It occurs after the mapping, tau, and league assertions and is tracked separately.
No OOS latents, books, backtests, or Layer-2 outputs have been rebuilt.

## Caveats

- This empirical magnitude is for the l05 pxG champion and selected fold; l03/l04 artifact magnitudes still require validation.
- This isolates league semantics only after composing the issue-01 bridge and l03 tau helper. It does not repair unknown-team fallback, training features, or any l03-l05 production behavior.
- Total goal-rate ratios do not equal `exp(delta)` because penalty conversion and own-goal noise are additive.
