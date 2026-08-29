# T006 — Scottish Lower arm loaders re-include a shared loader and redefine the models under test

| | |
|---|---|
| **Status** | open |
| **Severity** | low — invisible when each arm runs alone; corrupts any session that loads two arms |
| **Area** | `current_development/scottish_lower/02_poisson_wealth/`, `03_poisson_distance/`, `04_poisson_wealth_distance/` |
| **Raised** | 2026-08-28, by `05_composable_count_builder/r01_demo.jl` |
| **Verified on** | Julia 1.12.6, ScottishLower 56+57 |

## Summary

Two independent loader hygiene defects in arms 02/03/04. Neither shows up in the arms' own
walkthroughs, because each walkthrough loads exactly one arm into a fresh session. Both
break the moment anything loads more than one — which is what any cross-arm comparison,
A/B harness, or shared grid runner has to do.

## Defect 1 — the include guard tests a name that is never defined

`03_poisson_distance/l01_model.jl:2` and `04_poisson_wealth_distance/l01_model.jl:2`:

```julia
if !isdefined(@__MODULE__, :SLFeaturePoissonModel)
    include(joinpath(@__DIR__, "..", "02_poisson_wealth", "l00_feature_poisson.jl"))
end
```

`SLFeaturePoissonModel` does not exist. `02_poisson_wealth/l00_feature_poisson.jl:4` defines
the abstract type as **`AbstractSLFPModel`**. The guard is therefore always false and the
include always fires.

`02_poisson_wealth/l01_model.jl:2` has no guard at all.

Consequence: loading arms 02, 03 and 04 into one session includes
`l00_feature_poisson.jl` three times, redefining `SLFPLogSumWealthFeature`,
`AbstractSLFPModel`, all three `DynamicPoisson*GoalsTimeDecayModel` structs, `SLFPParams`,
and the `Features.add_feature!` / `build_turing_model` / `extract_parameters` methods
attached to them. Julia 1.12 permits top-level struct redefinition, so this does not error
— it silently produces new types with the same names while objects built from the earlier
ones are still in scope.

### Reproduction

```julia
include("current_development/scottish_lower/02_poisson_wealth/l01_model.jl")
T1 = SLFPLogSumWealthFeature
include("current_development/scottish_lower/03_poisson_distance/l01_model.jl")
T1 === SLFPLogSumWealthFeature      # false — same name, different type
```

### Fix

Guard on a name the file actually defines, in all three call sites, and add one to arm 02:

```julia
if !isdefined(@__MODULE__, :AbstractSLFPModel)
    include(joinpath(@__DIR__, "..", "02_poisson_wealth", "l00_feature_poisson.jl"))
end
```

Prefer a guard on the abstract type rather than on one concrete struct: it is the one name
in that file whose disappearance would mean the file genuinely had not loaded.

## Defect 2 — `subset` is called unqualified

`03_poisson_distance/l00_distance_feature.jl:277`:

```julia
selected_matches = subset(ds.matches, :match_id => ByRow(id -> Int(id) in selected_ids))
```

`DataFrames` and `DynamicPPL` both export `subset`. In the arm's own walkthrough only
`DataFrames` is in scope, so this resolves. In any session that also does
`using DynamicPPL` — which anything writing or benchmarking a Turing engine does — the
name is ambiguous and feature construction dies at the first distance fold:

```
UndefVarError: `subset` not defined in `Main`
Hint: It looks like two or more modules export different bindings with this name
```

### Fix

`DataFrames.subset(...)`. Then grep `current_development/scottish_lower/` for other
unqualified calls to names that DynamicPPL and DataFrames both export (`subset`, `select`,
`transform`, `combine`) and qualify those too.

## Blast radius

Confined to `current_development/scottish_lower/`. Nothing in `src/` is affected, and no
published gate result is wrong: every arm's gate run loaded exactly one arm.

The consumer that hit both defects,
`05_composable_count_builder/r01_demo.jl`, works around them — it declines to include
arms 02/03/04's `l01_model.jl` and reconstructs their configurations directly (§1, §3), and
its own loaders `import DynamicPPL` rather than `using` it. **Remove those two workarounds
when this ticket lands**, and let it use `tp02_model` / `tp03_model` / `tp04_model` as it
should have from the start.

## Acceptance criteria

1. `include`ing all of arms 00, 02, 03 and 04's `l01_model.jl` into one fresh session
   leaves `SLFPLogSumWealthFeature` and `AbstractSLFPModel` bound to exactly one type each.
2. A session with `using DynamicPPL` in scope can build a fold-1 FeatureSet for arm 03.
3. Each arm's `v01_walkthrough.jl` still passes gates 0-5 unchanged.
4. `05_composable_count_builder/r01_demo.jl` still reports 64/64 with the workarounds
   removed and the arms' own constructors restored.

## Scope guard

Do not touch the engines, the priors, the features, or anything under `src/`. This is
include hygiene and one qualified name. In particular do NOT "tidy" the three
`_engw`/`_engd`/`_engj` engines while you are in that file — replacing them is
`05_composable_count_builder`'s business and is already prototyped.
