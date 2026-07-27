# Kickoff brief: graduate the Poisson smile model into `src/` + verification runner

> Paste this file into a fresh Claude session in the `BayesianFootball` repo
> (`@current_development/split_market_pillar/PROMPT_graduate_smile_to_src.md`).
> It is self-contained; verify any claim against the code before acting on it.

## Mission (two deliverables, in order)

1. **Graduate** the validated Poisson smile engine
   (`LocalIntensitySmileDoublePoissonModel`, currently loader-local in
   `current_development/split_market_pillar/l03_local_intensity_poisson.jl`) into `src/`,
   following the codebase's engine/feature/prediction conventions.
2. **Create `current_development/split_market_pillar/r20_smoke_src_smile.jl`** — a
   single-split runner that builds the model **from src only** (no `include` of l03 —
   that's the proof of graduation) and verifies it end-to-end.

## Why (context you'd otherwise have to re-derive)

- The smile pillar is the **validated keeper** from the split_market_pillar stream: see
  `RESULTS_smile_grid.md` (§5 verdict). Cell `li_smile50` (smile_weight=0.5) had the best
  BTTS GLMEdge in the grid (7.24, p=0.01) AND top pooled-totals coefficient, at the lowest
  bias of the edge-carrying cells. Routing: **smile → totals/BTTS; 1X2 has no edge for any
  model** — don't "improve" that away.
- The smile is a **pricing object, not a likelihood object**: a global per-strike shape
  `log_φ ∈ ℝ^{Kmax+1}` (φ≡1 ⇒ plain Poisson) that anchors `log(λ_tot)+log_φ(K)` to the
  market-inverted per-strike intensity `Λ^mkt(K)`. φ does NOT enter the goals likelihood.
  Per-line O/U is priced `P(N≤K)=cdf(Poisson(λ_tot·φ(K)),K)`; 1X2/BTTS/correct-score come
  from the unchanged (λ_h, λ_a) grid. Keep this split intact.
- **The market inversion must stay Poisson-referenced.** Do not "upgrade" it to NegBin:
  pregame totals are ~Poisson in reality while the market prices over-dispersion — that gap
  IS the edge being harvested (memories: `no-pregame-intensity-smile`,
  `totals-compression-is-denoising`). An NB inversion would absorb the edge into a nuisance
  parameter.
- Hierarchical σ variants were tested and **nulled** (l08/r17, l09/r18/r19): keep the
  global scalar sampled σ exactly as l03 has it. Do not add hierarchy.

## Read before writing any code

1. `CLAUDE.md` (repo conventions: AD-safety rules, engine/component/feature/extractor
   recipes, prototyping workflow).
2. `current_development/split_market_pillar/l03_local_intensity_poisson.jl` — the ~450-line
   loader you are graduating. Everything you need is in it.
3. Sibling src engine for structure/naming:
   `src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl`
   (market version) and `..._no_market.jl`.
4. `src/predictions/score_computation/poisson.jl` + `src/predictions/inference.jl` (how
   `extract_params` → `compute_score_matrix` → `compute_market_probs` dispatch works).
5. `src/features/extractors/market_extractors.jl` (where the smile extractor belongs).
6. `current_development/split_market_pillar/NOTES.md` (canonical naming + findings log) and
   `RESULTS_smile_grid.md` (the numbers your smoke test should be consistent with).

## Graduation map (l03 piece → src home)

| l03 piece (line refs approximate) | src destination |
|---|---|
| `MarketSmileFeature` config + `Features.add_feature!` extractor (~l.69–140) — off-AD-path Poisson-CDF inversion of de-vigged O/U fair probs into `Λ^mkt(K)`, `Kmax=4` default | config registered like other feature configs; `add_feature!` into `src/features/extractors/market_extractors.jl` (follow the `Val`/config dispatch pattern used there) |
| Model struct + Turing engine + `build_turing_model` + `required_features` (~l.141–400) | new file `src/models/pregame/engines/player_level/time_decay/outfield_xg_smile_double_poisson.jl` (name to match siblings); export from the PreGame module like the other engines |
| `SmileScoreMatrix <: AbstractScoreMatrix`, `extract_params`, `compute_score_matrix`, per-line `compute_market_probs(::SmileScoreMatrix, ::MarketOverUnder)` + grid fallback (~l.399–452) | `src/predictions/score_computation/` (new file, e.g. `smile_poisson.jl`), included where the other score files are |
| `extract_parameters` (chains → per-match λ/φ) | with the engine file, like the siblings |

**Struct naming:** follow the src convention (siblings are
`Dynamic<X>OutfieldPlayerTimeDecay[NoMarket]Model`). Suggested:
`DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel`. A NEW name in src is safe
**as long as you do not touch l03** — old saved grids (`./data/double_poisson_smile_grid/`)
deserialize against the loader struct. **Do not rename, move, or edit l03 / l08 / saved
`.jls` dirs.**

**Defaults for the src struct = the keeper cell:** `smile_weight = 0.5`,
`supremacy_weight = 1.0`, `Kmax = 4`, sampled σs with l03's priors, `market_on = true`
toggle preserved.

## Gotchas (each has bitten before)

- **Prediction dispatch** (`dixoncoles-prediction-dispatch-union` memory): a new engine
  must either join the relevant score-computation Union or ship explicit
  `extract_params`/`compute_score_matrix` overrides — for the smile you MUST keep the
  dedicated `SmileScoreMatrix` path (a plain Poisson Union route would silently price O/U
  without φ and the model would look mysteriously "de-smiled"). Verify O/U probabilities
  differ from the grid-only computation in the smoke test.
- **AD-safety** (`docs/turing_ad_performance_guide.md`): no branching/loops on parameters
  in `@model`; masks + broadcast; the market inversion stays in the feature builder
  (off the AD path); sanitize `xg_rate` separately from λ (`xg-pillar-nan-and-sampler-gotchas`:
  present xG=0 → floor 1e-3; NaN would reach Gamma before the -Inf guard).
- **Convergence checks**: `Experiments.Diagnostics.check_convergence`'s curated df **drops
  parameters it doesn't recognize** (log_φ, σ_smile will be missing). The smoke runner must
  read raw chains with `ess_rhat` for the smile params.
- Betfair vs SofaScore: train pillar from plain `ds.odds` (SofaScore-derived), eval vs
  Betfair-swapped `ds1` — keep that separation if the runner evaluates anything.
- Run Julia on the **kaimon server REPL**, not locally (memory: `kaimon-repl-on-server`).
  Local edits reach the server only via `git push` → `git pull` in `/root/BayesianFootball`
  → **restart the REPL** (Revise won't pick up struct changes). Never retry
  `start_session` on timeout (duplicate sessions).
- Run `julia --project -e 'using Pkg; Pkg.test()'` (or at minimum parse/load checks) after
  the src edits — graduation touches module include lists.

## r20 smoke runner spec (single split, Ireland)

Conventions from `r03_split_controls.jl` / `r08_run_split_negbin.jl`:
`Data.Ireland()`, `target_seasons=["2026"]`, `history_seasons=2`, `warmup_period=21`,
`dynamics_col=:match_week`, `samples=1000, warmup=500, chains=4, use_queue=true,
max_depth=10`, `pinthreads(:cores)`.

Checks, in order (print a ✅/❌ verdict for each):
1. **Builds from src alone** — no `include` of any l0X loader anywhere in the file.
2. **Feature plumbing** — `required_features` returns the smile feature; the built model's
   data contains the `Λ^mkt(K)` vectors (spot-check a few matches' inverted intensities are
   finite and ordered sensibly).
3. **Convergence** — raw-chain `ess_rhat` on σ_smile / σ_sup / `log_φ[k]` (and components):
   R-hat ≤ ~1.01. (Reference: the l03 grid cells converged cleanly at these settings.)
4. **φ shape sanity** — posterior φ(K) should be a gentle monotone ≈ 0.93 → 1.05 over
   strikes 0.5→4.5 with every CI crossing 1.0 (known result; a wildly different shape means
   a porting bug, most likely in the strike indexing or the inversion).
5. **PPD end-to-end** — `Predictions.model_inference(ds, res)` runs; O/U selections priced
   via the smile path (assert they differ from grid-implied O/U for the same draws — that's
   the φ actually being used); 1X2/BTTS/CS via the grid; no `:r` ArgumentError.
6. (Optional, cheap) LogLoss vs Betfair close for the single split — just confirm it's in
   the plausible range of the r10 cells (diff_ll ≈ −0.02, not positive, not −0.2).

## Record-keeping (required)

- Append a dated entry to `current_development/split_market_pillar/NOTES.md` findings log
  (what was graduated, src file paths, r20 verdicts).
- Add a pointer line in `RESULTS_smile_grid.md` ("graduated to src as <name> on <date>").
- Update `CLAUDE.md`'s key-models list if it enumerates exported engines.
- If anything in this brief turns out to be wrong against the code, trust the code and note
  the discrepancy in NOTES.md.
