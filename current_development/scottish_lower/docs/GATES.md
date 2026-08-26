# Scottish Lower gates 0–7

**Purpose.** This is the consolidated gate contract for `00_team_poisson` (Poisson) and `01_team_poisson` (negative binomial). It records what the loaders actually assert, rather than treating a runner comment as an implementation. The shared scope is pooled tournaments 56/57, development `24/25`, two history seasons, and sealed `25/26` and `26/27`. The contract book is 1X2, BTTS, and O/U 0.5, 1.5, 2.5, 3.5; `max_goals = 12` means grid indices represent scores 0–11.

Every gate prints named results and is followed by `@assert sl_gate_table(...)`. Results must be appended to the model's `FINDINGS.md` with the configuration hash.

## Status and documented/implemented differences

- `PROTOCOL.md` calls the gradient target “≈0.6 ms”; model loaders say `<1 ms`; neither makes latency a failing condition. It is reported only. Model 01 measured 1.15–1.45 ms and passed; model 00 measured 0.030 ms.
- `PROTOCOL.md` says Gate 4 synthetic λ parity is “~1e-10”. Model 01 implements `1e-12` for λ and `r`; model 00 implements `1e-10` for λ. The stricter model-01 value is the current implemented rule for that model.
- `PROTOCOL.md` says Gate 5 matrices sum to `1 ± tol`. The implemented NegBin and Poisson gates correctly test against the **truncated grid mass**, not one: neither pricer normalises its grid. Gate 6 normalises market probabilities before score calculation.
- `GATES_6_7_PLAN.md` specifies a calibration-slope point-estimate band `[0.7, 1.3]`. Model 01 initially implemented that concept, then changed the actual pass rule to no finite calibration slope more than two standard errors from 1. Model 00 does not implement calibration in its `tp00_gate_not_broken` at all. Thus there is no common implemented `[0.7,1.3]` gate.
- Gate 3's smoke is one fold; its chain is persisted and reloaded by Gate 4. Gates 6–7 use the all-fold grid. The grid convergence call is labelled `6.0` in runners, but is Gate-3 sampling/convergence evidence, not a new protocol gate.
- Gate 7 is described as “CLV” versus Betfair close, but bets are also **executed and settled at Betfair close**. It is a closing-line benchmark / closing-price backtest, not entry-to-close CLV capture. No explicit numerical CLV pass threshold is implemented.
- Model 00 is a custom prototype engine and reimplements extraction; model 01 dispatches to `src`. This is contrary to the protocol preference to extend package APIs, but is the present code. Model 00's simpler Gate 4 does not test draw ordering, `r`, OOS match-ID equality, posterior depth, or the global-home-advantage fallback defect that model 01 tests.
- The initialisation documentation is internally stale: the findings describe `init_range=0.1`, while current `_protocol/config.jl` sets `init_range=2` in value space. The current code is authoritative for a new run.

## Gate 0 — Contract

**Objective:** make the fitted and genuinely held-out populations visible and prevent use of a sealed season.

**Inputs:** pinned/cached `DataStore`; `SLContract`; boundaries from `Data.create_id_boundaries`; OOS fixtures from `Data.get_next_matches`.

**Construction and outputs:** A fold contains `fitted_ids = history_match_ids ∪ target_match_ids` (all observations through step `t`) and OOS fixtures at `t+1`. `target_match_ids` are fitted, not test data. It prints fold, season, model step, fitted count, dropped count, `t+1` count, last fitted date, and first OOS date. Output is `Vector{TPFold}`/`Vector{TP00Fold}` plus five PASS/FAIL records.

**Implemented strict checks:**

1. all contract tournament IDs are present;
2. target fold seasons have empty intersection with `sealed_seasons`;
3. at least one fold exists;
4. every fold has at least one OOS fixture;
5. concatenated OOS match IDs have no duplicates.

A datastore fingerprint is required by `PROTOCOL.md`, but neither model's `l03_gates.jl` computes or asserts one. Boundary dates are printed at date rather than full kickoff resolution. The currently recorded normal inventory is 20 folds and 360 OOS fixtures.

## Gate 1 — Config

**Objective:** make every modelling and execution-relevant choice reproducible.

**Inputs:** model object and `SLContract`.

**Outputs:** printed component menu/fields and priors, required feature list, eight-character `sl_hash(repr(model))`, and gate records.

**Implemented strict checks:**

1. `Features.required_features(model)` is non-empty;
2. two hashes are equal and have length 8;
3. time-decay half-life is positive;
4. `l02_equations.jl` accepts the selected default component set (`*_assert_default`).

The walkthrough reporter prints component fields; the gate function itself does not prove each declared feature has a live extractor. That stronger requirement is documented in `PROTOCOL.md` but not directly asserted. Any non-default component deliberately fails the l02-coverage check until its independent referee is extended.

## Gate 2 — Features / anti-leakage

**Objective:** prove features are temporally admissible and extraction identity cannot silently zero team effects.

**Inputs:** datastore, Gate-0 folds, model, splitter, feature collections built with `Features.create_features(splits, ds, model, splitter)`.

**Outputs:** per-fold `FeatureSet`s, map/coverage diagnostics, and seven (model 01) or equivalent checks. The splitter-aware overload is required so calendar bins are aligned and compressed to model states.

**Formula:** for a fold cutoff
\[
 c=\min_{j\in\mathrm{OOS}}\{\operatorname{kickoff}_j\},\qquad
 \mathrm{fitted}=\{i\in H\cup T:\operatorname{kickoff}_i<c\}.
\]
Kickoff is `DateTime(match_date) + Hour(match_hour)`, not merely `match_date`.

**Implemented strict checks:**

1. every retained fitted kickoff is strictly before `c`;
2. truncating every match-ID-carrying datastore table to `fitted_ids ∪ oos_ids` leaves one selected fold's FeatureSet `isequal` key-for-key (future perturbation);
3. every numeric feature vector has no `missing` and no floating `NaN`;
4. every `team_map` is string-keyed, matching the extraction lookup;
5. each present `time_indices` vector is exactly `1:K` without gaps.

**Reported, not failing:** number and identity of filtered pre-cutoff violations (`dropped_ids`), plus unmapped OOS home/away sides and their population fallback. Promoted teams are legitimate, so coverage is not required to be zero. The perturbation test is only one configured fold (`perturb_fold=1`), not every fold.

The local trim is now normally a no-op after T001's shared calendar clock. It remains a defensive test.

## Gate 3 — Sampling

### 3a. Equation parity

**Objective:** establish that the Turing density is the independently documented model.

**Inputs:** a FeatureSet, three seeded prior `VarInfo` draws (`20260825, 7, 991`), DynamicPPL log-density, and a standalone `l02_equations.jl` referee.

**Formula:** for each draw \(\theta\), require
\[
 |\log p_{\rm DynamicPPL}(\theta)-\ell_{\rm l02}(\theta)|\le10^{-8}.
\]
The referee includes priors and weighted likelihood, not merely \(\lambda\). Default intensity is
\[
 \eta_h=\mu+\delta_m+\gamma_h+\alpha_h+\beta_a,\quad
 \eta_a=\mu+\delta_m+\alpha_a+\beta_h,\quad \lambda=\exp(\eta),
\]
with \(w_i=0.5^{\Delta_i/H}\); model 00 uses weighted Poisson log-pmf and model 01 weighted mean-parameterised NegBin log-pmf.

**Strict checks:** maximum parity error `≤1e-8`; exact equality of observed and documented sampled-site names; parameter count `4+2N_teams` (00) or `5+2N_teams` (01).

### 3b. Gradient parity and timing

**Objective:** verify the compiled ReverseDiff tape that NUTS uses is correct, finite, static-tape safe, and costed.

**Inputs:** same prior point, `f(θ)=logdensity`, compiled and fresh ReverseDiff, ForwardDiff, centred finite differences with \(\epsilon=10^{-6}\), three perturbations \(\theta+\delta\sin(1{:}d)\), \(\delta\in\{.001,-.002,.003\}\).

**Formula:** relative error is
\[
 \mathrm{relerr}(a,b)=\frac{\lVert a-b\rVert}{\max(\lVert a\rVert,\lVert b\rVert,1)}.
\]
Finite difference is \([f(\theta+\epsilon e_k)-f(\theta-\epsilon e_k)]/(2\epsilon)\).

**Strict checks:** finite density and all gradients; fresh vs compiled ReverseDiff relerr `≤1e-8`; compiled ReverseDiff vs ForwardDiff relerr `≤1e-6`; maximum absolute finite-difference error `≤1e-4`; compiled tape vs fresh ReverseDiff at probes relerr `≤1e-8`.

**Timing:** median of 50 compiled gradient evaluations and tape compilation time are reported with a `<1 ms` target; timing always returns `pass=true`. Model 01 additionally profiles tape instructions/allocations, also diagnostic only. NUTS is currently hardcoded to `AutoReverseDiff(compile=true)` in `src`.

### 3c. Initialisation, smoke, and convergence

**Objective:** prove a real persisted experiment samples safely. Smoke is one season-opening fold, four chains, 500 warmup and 500 retained draws. It is saved through `Experiments.save_experiment`; full grid is all folds, four chains, 800/800, with 16 queued tasks.

**Inputs/outputs:** `ExperimentConfig`, chains in `training_results`, saved path, sampler internals `numerical_error`, `tree_depth`, and `hamiltonian_energy`. Outputs include smoke/grid wall cost and convergence records.

**BFMI formula per chain:**
\[
 \mathrm{BFMI}=\frac{\sum_t(E_t-E_{t-1})^2}{N\operatorname{Var}(E)}.
\]

**Implemented strict checks (over every returned chain/fold):** returned folds must be nonzero and match `expected_folds` when supplied; maximum R-hat `≤1.01`; minimum bulk **and** tail ESS `≥400`; divergence rate `≤0.001` (0.1% of retained draws); no `tree_depth ≥ max_depth` (current cap 10); minimum BFMI `≥0.3`.

For any divergent fold, the additional strict funnel test computes, for `dyn.σ_a` and `dyn.σ_d`,
\[
 \frac{\operatorname{mean}(\sigma\mid\mathrm{divergent})}{\operatorname{mean}(\sigma\mid\mathrm{nondivergent})}\ge0.5.
\]
This implements the 2026-08-26 amendment: tolerated divergences require both low rate and no small-scale clustering. **Ambiguity:** if sampler internals omit `:numerical_error`, the rate check sets `pass` from a computed `-1` surrogate (normally true) while its detail says it was not recorded; this does not strictly fail missing divergence telemetry. Similarly BFMI/depth checks are omitted, rather than failed, if their internals are absent. Model 00 findings show a 0.49 ratio but mark it informational; its current code would fail `<0.5` if the check is actually included, so the historic report and implementation conflict.

## Gate 4 — Extraction / inference

**Objective:** prove the posterior quantities priced downstream are the quantities fitted above.

**Inputs:** Gate-3 chain; a synthetic multi-chain `Chains` fixture; test fixtures; FeatureSet/team map; `extract_parameters`; Gate-3 artifact reloaded from disk.

### 4a. Synthetic parity

Known, distinct parameter draws are placed column-major in two chains. For every test fixture and draw, independently reconstruct \(\lambda_h,\lambda_a\) from l02 and compare to extraction; model 01 also reconstructs \(r_h,r_a\).

**Strict checks:** fixture count/keys and posterior draw count (01); λ and r parity `≤1e-12` (01) / λ `≤1e-10` (00); no collapsed draws (all synthetic home λ values distinct). This catches variable-name, omitted-scale, mapping, and flatten-order errors. Model 00's check is narrower: it does not assert fixture keys/count or draw lengths explicitly.

### 4b. Real-chain plumbing

Reload saved smoke result, run `Experiments.extract_oos_predictions(...; force=true)`, and return latents.

**Strict checks, model 01:** priced OOS row count equals actual `t+1` count and is nonzero; match-ID sets equal; each fixture retains chain depth \(n_{iter}n_{chains}\); all λ and r finite and positive; all λ in `[0.2,6]`. Model 00 instead checks only nonempty rows, finite positive λ, and median home λ in `[0.5,3]`.

### 4c. Fallbacks and manifest

Model 01 strictly tests that an unmapped home side retains global home advantage (it currently exposes T003), and that month changes are inert under `GlobalInterception`; model 00 only requires finite fallback λ. Model 01's real-chain test reports the range of `r` to establish that the training clamp \(r=\exp(\operatorname{clamp}(\log r,-10,10))\) cannot differ from unclamped extraction. `PROTOCOL.md` additionally requires every sampled site be consumed or explicitly declared unused. The current site manifest check is Gate 3, not a full extraction consumption audit.

## Gate 5 — score matrix and markets

**Objective:** prove posterior latents are translated into the intended score distribution and book prices.

**Inputs:** extracted latent rows, `Predictions.extract_params`, `compute_score_matrix`, contract `max_goals`, and independent stock-distribution grids. Up to five rows × 25 draws are checked.

**Reference grids:**
\[
 S_{h,a}=p_h(h)p_a(a),\quad h,a=0,\ldots,G-1,
\]
where \(p=\operatorname{Poisson}(\lambda)\) for 00, and for 01
\[
 p=\operatorname{NegBin}(r,r/(r+\lambda)).
\]

**5a strict checks:** model has the intended abstract subtype; resolved score-pricer file is `poisson.jl` (00) or `negativebinomial.jl` (01); grid first dimensions are `(G,G)`. Model 01 also requires `extract_params` from `negativebinomial.jl` and explicit `r_h/r_a` route.

**5b strict checks:** maximum cellwise reference-grid error `≤1e-12`; every cell `≥0`; on highest mean-total fixture, change in posterior mean P(over 3.5) from `G=12` to `G=20` is `≤1e-4`; orientation test requires both marginal first moments within `0.05|λ_h-λ_a|` of the matching intensity; exact truncated first-moment error `≤1e-12`.

The exact orientation/moment reference includes both retained marginals, e.g.
\[
 E_G[Y_h]=\left(\sum_{h<G}h p_h(h)\right)\left(\sum_{a<G}p_a(a)\right).
\]
Raw truncation mass \(1-\sum S\) is reported, not directly thresholded.

**5c strict checks:** integer O/U lines are rejected; for every tested draw, 1X2, BTTS, and each O/U pair each partition the grid mass within `1e-12`; family sums agree within `1e-12`; O/U under equals direct sum of cells with `h+a < line` within `1e-12`; under probabilities are non-decreasing across configured lines. The correct identity is `P(home)+P(draw)+P(away)=sum(S)`, not 1.

## Gate 6 — OOS evaluation

**Objective:** reject broken probabilities, evaluate predictive shape, and compare each model on aligned `24/25` outcomes. It is not a requirement to beat the market.

**Inputs:** all-fold latents; realised scores; Bet365 proportional de-vig close `p=prob_implied_close/overround_close`; Betfair TWA close over `(-20,0)` minutes; market book; folds. Incomplete Betfair markets are dropped before evaluation.

**6a book integrity strict checks (model 01):** nonempty book; exactly one winner per fixture-market; de-vigged probabilities sum to 1 within `1e-9`; every probability strictly in `(0,1)`; expected 3 legs for 1X2 and 2 otherwise; present market-line set exactly equals contract book. Model 00 currently checks only nonempty book: its comments overstate its implementation.

**6b alignment strict checks (model 01):** nonempty model book; each baseline covers at least 80% of model fixtures (fixture, not row, coverage); Bet365/Betfair graders agree wherever overlap; all model probabilities finite and strictly `(0,1)`. Per-line coverage is reported. Model 00 checks merely nonempty joined rows, so it does not implement the protocol's identical-set assertion or 80% rule.

**Model probabilities:** score-grid prices are normalised drawwise by `mass_k=sum(S[:,:,k])`, then posterior-averaged. Normalisation magnitude is retained in fixture output. This is deliberate correction for truncation before comparison with de-vigged probabilities.

**Proper scores per selection/line:**
\[
 LL(p,y)=-\operatorname{mean}[y\log\tilde p+(1-y)\log(1-\tilde p)],\quad
 Brier=\operatorname{mean}(p-y)^2,
\]
with \(\tilde p=\operatorname{clamp}(p,10^{-9},1-10^{-9})\). Paired delta is \(\Delta LL=LL_{model}-LL_{market}\), with SE of per-row loss difference divided by \(\sqrt n\). Negative is model-better. Scores are fixture-pooled (not fold-averaged) and reported per selection/line; multiclass per-market summaries are display-only and respect one winner.

**6c predictive shape:** scoreline log predictive density is
\[
 LPD_i=\log\left[\frac1S\sum_s\frac{S_s(y_h+1,y_a+1)}{mass_s}\right].
\]
It is not gated, and is comparable only between models on identical fixtures. RQR uses posterior-predictive analytic marginal CDF bounds,
\[
 u\sim U(F(y-1),F(y)),\quad r=\Phi^{-1}(u),
\]
with Poisson (00) or NegBin (01) marginals. Model 01 strictly requires `|mean(RQR)|≤0.15`, RQR sd in `[0.85,1.15]`, home/away mean gap `≤0.30`, no off-grid scoreline, and mean grid-mass correction within `1e-3` of one. Model 00 implements mean `≤0.15` and sd `[0.80,1.25]` only; LPD is printed but no off-grid/mass gate exists.

**6d draw check:** model 01 uses \(z=(\bar y_{draw}-\bar p_{draw})/\sqrt{\bar y_{draw}(1-\bar y_{draw})/n}\) and requires `|z|≤2`. Model 00 instead requires absolute draw-rate difference `≤0.06`. The documented protocol does not fix a numeric draw threshold.

**6e “not broken”:** model 01 requires worst per-selection `ΔLL≤+0.02` and no finite calibration slope farther than 2 SE from 1. It reports probability-spread ratio, market-beating lines, and encompassing-regression model coefficient (the latter is not a gate). Model 00 only requires worst `ΔLL≤+0.02`; despite calculating a calibration slope, it does not enforce it. The planned `[0.7,1.3]` slope band is therefore not implemented.

## Gate 7 — growth / closing benchmark

**Objective:** establish that the executable book, solver, path, and reporting are honest. It does not establish an edge merely because ROI is positive.

**Inputs:** Gate-6/all-fold latents, Betfair close quotes, realised score settlement, `src/Portfolio`, `BookSpec`, 2% `PerBetCommission`, `DeArb`, `KellyLogUtility`, `BakerMcHale`, and `DailySlate` policies: full book, totals only, totals+BTTS, and 1X2 only. Policies are declared before results.

**Constraints:** `require_complete_markets=true`; per-selection stake `≤0.50`; `FixedCap(0.20)` limits simultaneous slate exposure; `SlateDrawdown(λ=23)` and `FlatTrust(0.30)` are current contract settings. The cap is approximately described as 0.2 in prose but is exactly 0.20 in code.

**7a book-construction strict checks:** built-book fixture coverage `≥50%`; every Kelly solve converges; among books with `maximum(a_kelly)>0`, KKT residual `≤1e-4`; every `p_grid` sums to 1 within `1e-9`; every book settles; books are chronological; at least one selection exists. KKT is deliberately not judged for no-bet boundary solutions.

**7b simulation strict checks:** all bankroll values positive; slates chronological; maximum exposure `≤0.20+1e-9`; no slate P&L `≤-1`; at least one bet recorded.

**7c verdict strict checks:** every declared policy has bets; final bankroll positive for each; full-book row is present. Bootstrap ROI uses 4,000 resamples by match. ROI interval exclusion, P&L concentration (top 10 profit share), and positive growth are findings/reports, not pass conditions. This is why an apparent 21% full-book ROI with top-10 P&L >100% passes integrity but is explicitly **not** a demonstrated edge.

No function computes a separate entry-versus-close CLV, nor is there an implemented CLV threshold; “CLV primary” remains a decision rule for comparison rather than a numerical gate.
