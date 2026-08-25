# Code review — `scottish_lower/proxy_xg`

Reviewed 2026-08-25. Scope: all 7 files in this directory, plus the `src` code they depend on
(`shot_parser.jl`, `plus_minus_extractors.jl`, `bbc_extractors.jl`, `goals_funnel_plus_minus_league.jl`,
`score_computation/poisson.jl`, `evaluation/batch_runner.jl`).

---

## Summary

**The maths is correct.** The likelihoods in `l02` are written as compressed "sufficient statistic"
formulas rather than per-match `logpdf` calls — fast, but easy to get subtly wrong. I checked all
three blocks against direct `Distributions.logpdf` on 200 randomised observations:

| block | max abs diff |
|---|---|
| Arm A quadratic — `Gamma(ν, μ/ν)` + `Poisson(κμ)` | 1.7e-13 |
| Arm A linear — `Gamma(μ/θ, θ)` | 2.8e-14 |
| Arm B — `Poisson(λ_s)` + `Gamma(ν_q·S, q/ν_q)` + `Poisson(κ·λ_s·q)` | 3.4e-13 |

Gamma is shape–**scale** as the header claims, so `ν` really is a precision. The conditional-on-S
construction in Arm B is right (`Var = S·q²/ν_q`, `CV = 1/√(ν_q·S)`) and is the correct way to stop
the xG pillar re-counting the shot-volume information the funnel already owns. The masking contract
(dummy value + 0.0 mask, never `missing`/`NaN`/0.0) is respected everywhere, and the `n_safe` guard
against `loggamma(0)·0 → NaN` is real and correct. All seven files parse cleanly.

**The problems are not in the engine — they are in the inference chain built on top of it.** The
go/no-go gate cannot return "no", and the written conclusions contradict the numbers in the same
document.

---

## Blockers

### B1. Every include path is stale — nothing here can run

All files reference `current_development/scottish_proxy_xg/…`. The directory is now
`current_development/scottish_lower/proxy_xg/`. Affected:

- `r00_data_qa.jl:18`
- `r01_eda_informativeness.jl:31`
- `r02_smoke.jl:31`
- `r03_grid.jl:41`, and `r03_grid.jl:151` (the convergence-report write path)
- `r04_eval.jl:41`

Each will raise `SystemError` on the first `include`. Header comments (`l01:1`, `l02:1`, the
copy-paste run lines in every runner) carry the same stale path.

### B2. The go/no-go gate is built so it cannot say "no" — and that is what happened

`r01_eda_informativeness.jl:210`:

```julia
E2_ARM_A = (t_CB_goals > 1.5) || (t_CB_res < -1.5) || (abs(t_CB_goals) < 1.0 && abs(t_CB_res) < 1.0)
```

The third clause treats **"too small to measure"** as a pass. Feeding in the recorded results
(`RESULTS` §2):

| quantity | value | clause |
|---|---|---|
| `t_CB_goals` | −0.21 | not > 1.5 |
| `t_CB_res` | +0.57 | not < −1.5 |
| both `abs(t) < 1.0` | yes | **fires** |

`E2_ARM_B` is false (t = +0.35 / −0.74). `E3_PASS` (`r01:253`) is false — shot counts beat proxy xG
on *both* halves of the reliability test (self 0.896 vs 0.826; predicts-goals 0.798 vs 0.779).

So `GO = E2_ARM_A || E2_ARM_B || E3_PASS` (`r01:358`) was true **only** because of the
null-counts-as-pass clause. Every measured effect in the ladder was indistinguishable from noise,
and the 25-hour grid launched anyway.

`E3_PASS` compounds it: it is a bare `>` between two correlations over 23 teams with no significance
test — roughly a coin flip.

**Fix:** drop the third clause, or make it an explicit `E2_ARM_A_TIE` flag that is reported but does
not feed `GO`. Give `E3_PASS` a confidence interval.

### B3. `RESULTS` §6 does not follow from `RESULTS` §5, and it breaks the pre-registered rule

The rule fixed in advance (`RESULTS` §0, echoed at `r04_eval.jl:213-216`):

> cell 2 or cell 4 must beat cell 1 on **hurdle_G** for the **totals and BTTS** families on the
> **Betfair** book, with per-line LogLoss no worse, at ≥95% fold convergence.

What §5 reports is **ROI on the Bet365 close**. There is no Betfair table and no `hurdle_G` anywhere
in the document, although `r04`'s `money_tables` prints `roi_pct`, `hurdle_G` *and* `bets_placed`.
The Betfair block (`r04:189-208`) is wrapped in `try/catch` that only `@error`s — so "it crashed" and
"we didn't record it" look identical from the write-up.

`r04:11-14` warns explicitly that this book switch once reversed the ranking on this exact family
(+6.8% Betfair vs −9.5% Bet365, `[[apm-l1-graduation]]`). The deciding test was not applied.

§6 also calls cell 4 the "CLEAR WINNER across all families". §5's own table says otherwise:

| family | c1 `funnel_apm_ctl` | c2 `pxg_apm` | c3 `pxg_noapm` | c4 `funnel_pxg_apm` | best |
|---|---|---|---|---|---|
| x12 | +0.0090 | +0.0049 | **+0.0046** | +0.0054 | c3 |
| btts | +0.0026 | +0.0025 | **−0.0005** | −0.0001 | c3 |
| totals | −0.0017 | −0.0019 | −0.0041 | **−0.0042** | c4 (by 1e-4) |
| totals_tails | −0.0022 | −0.0024 | **−0.0047** | −0.0044 | c3 |

Cell 3 wins three of four. **Cell 3 is the control with the RAPM pillar switched off** — the
stripped-down model. That is a direct answer to the "cell 2 vs cell 3 — pillar or RAPM?" question
posed at `r04:219`, and it points the opposite way to §6.

§6 claim 3 states the RAPM weights are "statistically positive (p < 0.01)". No p-value is computed
anywhere in `r03` or `r04` — the diagnostics block (`r03:134-143`) prints mean and a 90% band only.
That figure is unsupported.

Finally, the growth table compares cells with very different bet counts (127 vs 226 on Over 2.5).
Raw ROI across different bet volumes is not comparable; a model that bets less often can look better
without being better. `hurdle_G` — the metric the rule named — is what corrects for this.

### B4. Stale status headers

`RESULTS_scottish_proxy_xg.md:8` and `NOTES.md:169` both still say **"nothing run yet"** above six
fully-populated result sections. On a document whose stated purpose is pre-registration, this matters.

---

## Genuine bugs

### C1. Arm B is missing Arm A's NaN backstop — `l02:502`

Arm A closes with (`l02:299-302`):

```julia
total = ll_xg_h + ll_xg_a + ll_g_h + ll_g_a
Turing.@addlogprob! ifelse(isnan(total), -Inf, total)
```

with a comment explaining exactly why: a NaN *shape scalar* reaches `loggamma` rather than `log_μ`,
so the upstream `is_bad` rejection cannot see it, and `-Inf + NaN == NaN` silently defeats the
rejection.

Arm B ends with a bare add:

```julia
l02:502   Turing.@addlogprob! ll_s_h + ll_s_a + ll_q_h + ll_q_a + ll_g_h + ll_g_a
```

Arm B has *more* exposure to this, not less: `lνq = log(ν_q)` (`l02:478`) and
`loggamma.(ν_q .* n_ev)` (`l02:493`, `:499`) both execute after the `isnan(ν_q)` check has already
emitted its `-Inf`. Apply the same `ifelse(isnan(total), -Inf, total)` wrapper.

### C2. `r01` E5 does not filter `period == "ALL"` — `r01:320`

`ds.statistics` holds one row per `(match_id, period)` — `"ALL"`, `"1ST"`, `"2ND"`. Every `src`
extractor filters it (`src/features/extractors/stats_extractors.jl:6`, `:23`, `:35`). `r01:320`
iterates `eachrow(st)` unfiltered, so each match contributes ~3 rows pairing **full-match** proxy xG
against **half-match** SofaScore xG.

Consistent with the symptoms in `RESULTS` §2 E5: `n = 2,332` (≈3× the covered-match count),
`cor = 0.614` against a published team-level 0.817, and `slope = 0.689` — biased low exactly as
mixing halves into the denominator would predict.

**Fix:** `for r in eachrow(st) ... r.period == "ALL" || continue`.

### C3. Arm B clamps in a different place when training vs predicting

- model (`l02:445`, `:452`): `clamp(shot_scale + lin, −10, 10)` — offset is *inside* the clamp
- extractor (`l02:601`, `:649`): `exp(shot_scale + clamp(lin, −10, 10))` — offset is *outside*

At realistic values (log λ_s ≈ 2.3) the clamp never binds, so this changes nothing today. It is a
silent train/predict divergence waiting on a pathological fold. Arm A is consistent (`l02:246`,
`:601` — no offset either side).

### C4. `_pxg_outfield` drops the missing-side mask — `l02:104`

`src`'s `_pm_outfield` (`goals_plus_minus_league.jl:172-176`):

```julia
ifelse.(tot .> 0.0, tot .- (10.0 * base), zero(eltype(tot)))
```

`l02:104`:

```julia
_pxg_outfield(D, M, F, base) = (D .+ M .+ F) .- 10.0 * base
```

The `tot > 0` guard is gone. `_pxg_extract_core:594-597` has the same asymmetry.

For RAPM `base == 0.0`, so the two are numerically identical and nothing is wrong today — the type
bound `P<:Features.AbstractPlusMinusFeature` makes any other case unreachable. But the docstring's
claim that "a non-RAPM rating family drops in unchanged" is **false**: with SofaScore ratings
(`base = 6.5`) a side with no lineup data would be fed −65 into the pillar, which is precisely the
failure `_pm_outfield`'s docstring was written to prevent. Fix the code or fix the comment.

---

## Gates that cannot fail

### D1. `r00:135-140` — the free-kick leak gate is tautological

```julia
n_fk_zone = count(==(:free_kick_zone), shots.zone)
_mark("5b. :free_kick_zone fully remapped away", n_fk_zone == 0)
```

`parse_shot` (`shot_parser.jl:114-117`) rewrites `:free_kick_zone → :outside_box` **before it
returns**, so `shots.zone` can never contain that symbol. The count is 0 by construction and the
gate tests nothing.

**Fix:** count `(zone == :outside_box) & (context == :direct_free_kick)` and assert it matches the
expected ~515, or assert the conversion rate of that subset is *not* ~1.0 (which is the leak the
remap exists to kill).

### D2. `r02:144` — the ν prior gate passes on zero learning

```julia
_mark("3c. nu moved off its prior (posterior sd < prior sd 1.5)", std(ν) < 1.5)
```

`truncated(Normal(4.0, 1.5), lower = 0.5)` has sd ≈ 1.48, so this passes even if the posterior is
the prior. **Fix:** compare against the actual truncated prior sd, or gate on a ratio
(`std(ν) / prior_sd < 0.5`).

---

## Fragilities

### E1. One bad metric deletes a whole model row

`src/evaluation/batch_runner.jl:44-52` pushes the result row **only if every metric succeeded**;
otherwise the model is silently absent from the returned DataFrame. `r04:96-98` bundles
`RQR()` + 17 × `LogLoss` + 17 × `GLMEdge` into a single call. A degenerate GLM on a thin
selection — `under_05` is the obvious candidate — removes that cell's entire LogLoss row.

`present = sort(unique(metrics_eval.model))` (`r04:101`) is then computed from whatever survived, so
downstream tables just have fewer rows with no error. **Fix:** run `GLMEdge` in its own
`evaluate_experiments` call.

### E2. The leak check was designed but never run

`NOTES.md` §4 limitation 1 states `ProxyXGFeature(fit_on = :training)` "exists so the difference can
be MEASURED rather than asserted". No grid cell in `r03` uses it — `r00:190` only checks it returns
finite values. The leak is genuinely small (a fold's ~25 matches against a ~19.5k-attempt league-wide
table), but it remains asserted. Same for `team_quality_on = false`, described in `l02:582` as a
"cheap sanity cell" and never scheduled.

### E3. E4's variance-law verdict has no uncertainty

`RUN_LINVAR` is switched on by `b = 1.123 < 1.5`, where `b` is the slope of a log–log fit through
10 decile points, with no standard error, derived from a Gamma GLM that itself assumes the quadratic
law being tested. Suggestive, not a test. Report a CI on `b` before treating it as a decision.

### E4. `r04` season split depends on the OOS cache

`r04:128-145` rebuilds a season-filtered DataStore and re-runs `evaluate_experiments`. That works
only because §2 already ran and cached `oos_latents.jls` — on a cold cache,
`extract_oos_predictions` would hit the "DataStore drift detected" guard
(`post_processing.jl:147-149`) because the filtered `ds.matches` yields fewer boundaries than there
are splits. It is inside a `try/catch` labelled diagnostic-only, so this is correct as written, but
the ordering dependency is implicit and worth a comment.

---

## Modelling caveat not currently documented

Both arms treat proxy xG and goals as **conditionally independent** given the rate — `μ` in Arm A,
`(λ_s, q)` in Arm B. They are not. A goal *is* one of the shots proxy xG is summed over, so the two
observations are positively correlated beyond the shared rate, and the posterior is over-sharpened.

Arm B's conditional-on-S construction removes the *volume* double-count — correctly, and it is the
best idea in the stream — but not this one. The magnitude is modest (~1.2 goal-shots out of ~9.1
shots per side) and it is the same class of error the funnel's thinning structure was built to
avoid. Given how carefully the rest of the double-counting reasoning is documented in `NOTES` §3,
this deserves a line there.

---

## Suggested order of work

1. **B1** — fix the include paths (mechanical, unblocks everything else).
2. **C1, C2, C3** — the three real code bugs (all small, all local).
3. **D1, D2** — replace the two dead gates with tests that can fail.
4. **B2** — decide what the gate should do on a null, then re-read `RESULTS` §2 against it.
5. **B3, B4** — run the Betfair / `hurdle_G` leg, then rewrite §6 against whatever it says. Cell 3
   currently outscores cell 4 on three of four families and is the simpler model.
6. **C4, E1–E4** — hardening and documentation.
