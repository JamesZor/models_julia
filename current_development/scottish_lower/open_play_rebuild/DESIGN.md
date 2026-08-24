# Open-play rebuild: auditable v1 design contract

**Status:** Stages 1–6 are remotely validated; no MCMC has run. V1 is deliberately small: **only the NP-NOG component has team, league, month, and home effects.**

## 1. Objective and notation

For match `i`, at kickoff `t_i`, with home/away teams `h_i,a_i`, source league `r_i∈{56,57}`, model league `ℓ_i∈{1,2}`, and calendar month `m_i∈{1,…,12}`, reconstruct each final score as a sum of independent components:

`G_is = Y_is + C_is + O_is`, for scoring side `s∈{H,A}`.

| Symbol | Meaning |
|---|---|
| `G_is` | official final goals credited to side `s` |
| `Y_is` | non-penalty, non-own-goal (NP-NOG) goals credited to `s` |
| `A_is` | penalties awarded to `s`; any non-negative integer is permitted |
| `C_is` | converted penalties, with `0≤C_is≤A_is` |
| `O_is` | own goals credited to `s` (committed by its opponent) |
| `w_i` | fixed history-only training weight |
| `η_is` | NP-NOG log intensity |

A goal is in `Y` iff it is credited to that side and is neither a converted penalty nor an own goal. It therefore includes ordinary non-penalty goals and non-penalty set pieces; “open play” is historical shorthand, not an event-taxonomy assertion.

Raw own-goal events can be recorded under either the committing or benefiting team. Stage-2 reconciliation established that this snapshot's provider `is_home` value denotes the **beneficiary/scoring side**: all 39 informative own-goal matches uniquely reconciled under that convention and none under the committing-side interpretation. Therefore canonical `O_iH` counts `ownGoal` events with `is_home=true`, and `O_iA` counts those with `is_home=false`. This remains a snapshot-tested contract, not an undocumented assumption: every future split/snapshot must retain reconciliation and must quarantine ambiguity rather than guessing it into `Y`.

## 2. History-only filtration, identities, and reconciliation

For every temporal split, `cutoff` is the held-out fixture kickoff boundary. Training rows have `kickoff_i < cutoff`; validation/OOS rows have `kickoff_i ≥ cutoff`. OOS observations, outcome-derived normalisers, training weights, and mappings learned from results never enter training. Event corrections must be present in the data snapshot chosen before the run; later corrections are not silently backfilled.

Build a canonical global identity crosswalk once from canonical team/source identifiers, not display names or split-local categoricals. It is stable over the declared snapshot, spans both leagues and aliases/promotions/relegations, and provides a stable canonical identity for every known team.

This global crosswalk is **not** the posterior coordinate vector. For each split, posterior team columns contain **history-seen teams only**, in canonical-crosswalk order. Store the name/ID-to-posterior-column map with the split manifest; that stored map is authoritative for extraction and inference. A canonical team that is target-only/unseen in that split has no posterior column and receives `α=β=0`, not an unlearned prior draw. The same fallback applies to an identity absent from the global snapshot. Never rebuild or infer a posterior column from target data.

The only legal source-league map is `league_index(56)=1`, `league_index(57)=2`. It is an explicit checked dictionary, never raw-ID indexing or sorting-dependent encoding. Other IDs are data-contract errors. Tests must prove both mappings, both league presence, and length-two league vectors.

### Mandatory component reconciliation QA

Before a row enters a builder, QA must:

1. make every `G,Y,A,C,O` a non-negative integer;
2. enforce `C≤A` (with no artificial `A≤1` restriction);
3. resolve own-goal side to the beneficiary convention;
4. prove `G_is=Y_is+C_is+O_is` for both sides;
5. prove the two-side total equals the official match total;
6. report event IDs and raw/derived fields for every failure; and
7. exclude failures only through a versioned, reviewed quarantine table, counted by season, league, and reason.

Failures are not repaired by clipping, residual allocation, or treating them as normal zeros. The training manifest records snapshot hash, cutoff, included/quarantined match IDs, and component counts.

## 3. V1 feature contract, effects, and priors

Use the fixed preconfigured time-decay weight `w_i=2^{-d_i/H}`, where `d_i≥0` is days from match to cutoff and `H>0` is the half-life. It multiplies every observed likelihood term for both sides and all components; priors are unweighted.

### NP-NOG: the sole hierarchical component

Let `J` be the number of history-seen teams for the split, ordered by the canonical crosswalk, and let `j(h_i),j(a_i)` be their stored posterior columns. Draw independent non-centred raw effects `z^A_j,z^D_j∼Normal(0,1)`, center over these `J` history-seen columns,

`z̃^A_j=z^A_j-mean_k(z^A_k)`, `z̃^D_j=z^D_j-mean_k(z^D_k)`,

and set

`α_j=τ_A z̃^A_j`, `β_j=τ_D z̃^D_j`,

where `τ_A=exp(κ_A)`, `τ_D=exp(κ_D)`, and `κ_A,κ_D∼Normal(log(0.35),0.50)`. Here `α` is attacking strength and `β` is defensive vulnerability, so positive `β` increases the opponent's rate. Centering identifies the NP-NOG intercept. Extraction returns raw `κ` and transformed `τ`.

Use the centered two-league contrast `Δ∼Normal(0,0.50)`,

`L_1=+Δ/2`, `L_2=-Δ/2`.

For months, use a non-centred hierarchy: `z^M_m∼Normal(0,1)`, `σ_M=exp(ξ_M)`, `ξ_M∼Normal(log(0.20),0.50)`, and `M_m=σ_M(z^M_m-mean_k(z^M_k))`. Thus league and month effects each sum exactly to zero. Let `μ_Y∼Normal(log(1.20),0.50)` and NP-NOG home advantage `b_Y∼Normal(0,0.35)`. Then

`η_iH=μ_Y+L_ℓi+M_mi+b_Y+α_j(h_i)+β_j(a_i)`,

`η_iA=μ_Y+L_ℓi+M_mi+α_j(a_i)+β_j(h_i)`,

Define the branch-free smooth saturation `s(x)=20 tanh(x/20)`. Then `λ^Y_is=exp(s(η_is))+10^-6`, and `Y_is|θ∼Poisson(λ^Y_is)`. Unlike a hard clamp, `s` remains valid when a compiled ReverseDiff tape is evaluated across saturation regimes.

There are no NP-NOG loading parameters: attack and defence enter with unit coefficients by definition.

### Penalty awards and conversion: global only

Penalty awards have no team, league, month, or NP-NOG loading. Let

`pen_base∼Normal(log(0.12),1.0)`, `pen_home∼Normal(0,0.35)`,

`λ_pen,H=exp(s(pen_base+pen_home))+10^-6`,

`λ_pen,A=exp(s(pen_base))+10^-6`.

For every match and side,

`A_iH∼Poisson(λ_pen,H)`, `A_iA∼Poisson(λ_pen,A)`,

`q_pen∼Beta(8,2)`, `C_is|A_is,q_pen∼Binomial(A_is,q_pen)`.

This explicitly permits multiple penalties and conversions. Shoot-outs and extra time remain excluded/quarantined unless final-score/event semantics are separately approved.

### Own goals: global only

Use one global scoring-side own-goal rate,

`λ_og∼Gamma(shape=2, scale=0.015)`,

where the Julia scale convention gives prior mean `0.03`. For every match and scoring side,

`O_iH,O_iA | λ_og ∼ Poisson(λ_og)`.

There is no latent `r`, no Gamma--Poisson mixture, no negative-binomial-like construction, and no own-goal team, league, month, or home effect.

These are the complete V1 priors; no unlisted hierarchy is implied.

## 4. Complete weighted joint likelihood

Let `θ={zA,zD,κ_A,κ_D,μ_Y,Δ,zM,ξ_M,b_Y,pen_base,pen_home,q_pen,λ_og}` and let `p(θ)` be exactly the prior product above (with `α,β,τ,L,M,σ_M` deterministic transforms). For reconciled training rows `D`, the complete weighted log density is

`log p(θ,D) = log p(θ) + Σ_i∈train w_i { Σ_s∈{H,A} [ log Pois(Y_is;λ^Y_is) + log Binomial(C_is;A_is,q_pen) + log Pois(O_is;λ_og) ] + log Pois(A_iH;λ_pen,H) + log Pois(A_iA;λ_pen,A) }`.

The factor `w_i` intentionally power-weights all observed component terms for match `i`; it does not weight priors. The Binomial likelihood is evaluated only on reconciled integer observations satisfying `0≤C≤A`; no model-side branch repairs invalid data.

Conditional on parameters, the two sides and the three components are independent except that each side’s observed conversion is conditional on its observed award count. This is a component likelihood, not a direct final-score likelihood.

## 5. Extraction, fallback, and predictive score reconstruction

For each posterior draw, extraction re-evaluates the exact Section 3 transforms: stored source-league map, history-seen posterior-column map/order, centered effects, month convention, smooth saturation/floor, and home term. A parity test compares builder and extracted deterministic rates on selected training and OOS rows to `≤1e-10`.

For inference, use `α=β=0` for a side whose canonical ID has no entry in the split’s authoritative history-seen posterior map, including target-only canonical teams. Retain global NP-NOG posterior uncertainty and all applicable league/month/home terms. Report `team_status∈{history_seen,population_fallback}` per side; never drop, alias, or generate a new team coordinate.

For a draw, Poisson thinning gives the predictive converted-penalty distribution directly:

`C_iH∼Poisson(q_pen λ_pen,H)`, `C_iA∼Poisson(q_pen λ_pen,A)`.

Own goals are `O_is∼Poisson(λ_og)` and NP-NOG goals are `Y_is∼Poisson(λ^Y_is)`. Retain an explicit component convolution for audit, using finite PMFs on `0:K` and enlarging each support until its tail is below configured `ε` (default `1e-10`):

`p_Y(k)=Pois(k;λ^Y_is)`, `p_C(k)=Pois(k;q_pen λ_pen,s)`, `p_O(k)=Pois(k;λ_og)`,

`p_G(g)=Σ_(y+c+o=g) p_Y(y)p_C(c)p_O(o)`.

The convolution must be retained even though the sum is analytically Poisson, because it audits the three score components. Convolve separately for home and away, form `P^(d)_xy=p_G,H^(d)(x)p_G,A^(d)(y)`, expand joint support to residual mass below `ε`, normalize, and assert finite non-negative entries and unit sum. Average posterior-draw tensors, normalize/check again, then derive scorelines, 1X2, BTTS, totals, and correct scores.

Required outputs include draws/summaries for the NP-NOG hierarchy and transforms; `pen_base`, `pen_home`, `λ_pen,H`, `λ_pen,A`, `q_pen`, and thinned conversion rates; `λ_og`; NP-NOG/component rates and PMFs; score tensors/markets; team statuses; and reconciliation/extraction provenance.

## 6. Explicit exclusions

V1 has no penalty team/month/league effects or loadings; no own-goal hierarchy, latent intensity, mixture, or NB-like construction; and no referee, finance, pxG, Dixon--Coles, direct negative-binomial score likelihood, market likelihood, player hierarchy, side dependence/copula, extra-time/shoot-out model, or legacy leaderboard reuse. Legacy code/artifacts remain untouched.

## 7. AD correctness and performance contract

The future Turing implementation follows `docs/turing_ad_performance_guide.md`:

1. The feature builder performs filtration, reconciliation, map validation, history-seen column lookup/fallback flags, concrete `Vector{Int}`/`Vector{Float64}` conversion, weights, and all conditional logic.
2. `@model` has no scalar `for`/comprehension likelihoods, `if`/`else`, `isnan`, or parameter-dependent control flow. Use broadcasts, `logpdf.`, masks, weighted sums, and `@addlogprob!`.
3. Do not subset tracked intermediates; select sampled team vectors with `view`, avoid allocating `A[indices]`, and do not mutate tracked arrays.
4. Smoothly saturate log rates with `s(x)=20tanh(x/20)` and add floors. Stage-5 testing demonstrated that hard `clamp` records stale parameter-dependent control flow in a compiled ReverseDiff tape; it is therefore prohibited here. Numerical guards must be branch-free and avoid early returns.
5. Compile a production-shaped `ReverseDiff.GradientTape`; gate on finite gradients, tape compilation, near-zero allocations, target `<1 ms` gradient evaluation (investigate `>5 ms`), finite-difference agreement, and repeatability.
6. Change parameters across regimes and require compiled/uncompiled log density and gradients to agree, catching sampled-parameter branches.

### Stage 7 remote NUTS protocol

MCMC is remote only. Follow the project remote-execution protocol: commit/push the implementation, sync it on beast, and execute in the persistent remote session with `julia --project -t16`. In Julia call `BLAS.set_num_threads(1)` and `ThreadPinning.pinthreads(:cores)` before sampling. Use `QueuedNUTS` as four independent concurrent chain tasks, with default `800` warmup plus `800` retained draws per chain (both configurable). Monitor the remote tmux pane; never run heavy MCMC locally.

NUTS chains are single-threaded. Four tasks leave 16-core headroom for runtime, data work, monitoring, and stable pinning. Preferred convergence is `Rhat≤1.01`; `Rhat≤1.05` is the hard smoke gate. Record bulk/tail ESS and investigate low values; require zero divergences, tree-depth/saturation reporting within maximum, and BFMI reporting/checking where available. A failed hard gate is diagnostic-only and cannot produce an evaluation artifact.

## 8. Numbered implementation and validation stages

1. **Contract freeze.** Approve source columns, provider semantics, snapshot policy, and quarantine ownership. *(Approved for the audited snapshot.)*
2. **Component audit.** Build history-only audit/reconciliation ledger; hand-check own-goal beneficiary conversion and publish league/season counts/errors. *(Completed: 718/720 reconciled; beneficiary convention supported 39–0; two matches quarantined.)*
3. **Identity/features.** *(Implemented in `l02_rebuild_features.jl`.)* Canonical registry access is parameterized, transaction-read-only, timeout-bounded, and uses only `BF_DB_URL`; the pure builder accepts its already-fetched DataFrame. It validates one row per requested match, provider IDs/slugs/names and explicit alias conflicts, fingerprints deterministic match-ID-sorted serialization, audits history only, excludes all quarantines, and stores history-only maps/manifest. `resolve_oos_identity` returns a stored column or explicit population fallback. `r02_validate_maps_and_filtration.jl` tests mapping and leakage. Do not extend `Features.create_features` until l03 defines a model type; add a thin adapter then, never a monkey patch.
4. **Pure equations.** *(Remotely validated at commit `02a4414`.)* `l03_rebuild_equations.jl` defines the exact primitive parameter contract, generic non-mutating transforms, vectorized component rates, data-only weighted likelihood, scalar validation reference, predictive component rates, and flatten/unflatten helpers. `r03_validate_equation_parity.jl` passed centering, support, both-league, rate/likelihood parity, smooth-saturation safety, thinning, no-mutation, ForwardDiff, and central-difference checks on the 718-row Stage-3 FeatureSet. Priors remain deliberately outside `weighted_data_loglikelihood`.
5. **AD/model smoke.** *(Remotely validated at commit `c929bf0`.)* The model-owned feature adapter and 66-parameter, one-`@addlogprob!` Turing wrapper call the exact pure equation layer. All primitive-site, finite-density, fresh/compiled ReverseDiff, ForwardDiff, finite-difference, nearby-point, and smooth-saturation gates passed. Compiled ReverseDiff measured 0.633 ms median / 0.695 ms p95 with 18 allocations (1,920 bytes), meeting the `<1 ms` target. No sampling was performed.
6. **Extraction/recombination.** *(Remotely validated at commit `6b59173`.)* `l05_rebuild_extraction_recombination.jl` requires an exact primitive parameter-section manifest, stacks iterations×chains without touching raw storage, recomputes every transform, resolves OOS identities only through the registry and stored history map, and performs audited three-component Poisson convolution. The deterministic 12×2-chain metadata-only OOS runner passed both leagues, target-only fallback, transform/equation parity, team swaps, adaptive-tail normalization, and ordinary `model_inference`; its 30×30×24 tensor matched normalized direct Poisson to `8.67e-17`.
7. **Remote sampling smoke.** Run the specified `-t16`, pinning, four-chain `800+800` protocol; archive commands/configuration/diagnostics and no-local-MCMC claim.
8. **OOS inference/evaluation.** Generate only new OOS artifacts, inspect reconciliation/fallback rates, then evaluate without legacy leaderboard reuse.

A single split passes only when all manifest parameters extract; builder/extraction parity is `≤1e-10`; both league mappings occur; known and target-only fallback fixtures infer without remap/drop; every draw and averaged score tensor is finite, non-negative, and normalized to `1e-10`; `model_inference` completes; and Stage 7 hard diagnostics pass.

## 9. Parameter and extraction manifest

One versioned split manifest records snapshot/version hash, cutoff, source/model league map, canonical-crosswalk hash, the ordered history-seen team IDs and authoritative name/ID-to-posterior-column map, fallback IDs, month convention, half-life, support/tail tolerance, quarantine summary, and sampler/AD configuration.

| Group | Extracted parameters / derived values |
|---|---|
| NP-NOG team hierarchy | `zA[J]`, `zD[J]`, `κ_A`, `κ_D`, `τ_A`, `τ_D`, `α[J]`, `β[J]`; `J` is history-seen only |
| NP-NOG global | `μ_Y`, `Δ`, `L[2]`, `zM[12]`, `ξ_M`, `σ_M`, `M[12]`, `b_Y` |
| penalty | `pen_base`, `pen_home`, `λ_pen,H`, `λ_pen,A`, `q_pen`, `q_pen*λ_pen,H`, `q_pen*λ_pen,A` |
| own goal | `λ_og` |
| predictive | `η_Y`, `λ_Y`, component PMFs, `P_G`, score tensor, market summaries, side team status, provenance |

Derived entries are recomputed from primitive posterior draws and recorded alongside them, so transform parity is auditable.
