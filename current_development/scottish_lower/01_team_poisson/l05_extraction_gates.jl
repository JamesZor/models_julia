# ==============================================================================
# Model 01 — GATE 4 : EXTRACTION
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Gate 3a proved the FITTED model is the documented model. Gate 4 asks the
# separate question: is the PRICED model the same one that was fitted?
#
# These are genuinely different code paths. Training runs inside `@model` on
# TrackedArrays; extraction runs afterwards on an MCMCChains object, indexing by
# variable NAME and reassembling the same arithmetic by hand. Nothing in the
# package forces them to agree, and the 2026-08-24 audit found a prototype where
# they did not: it re-implemented `extract_parameters`, keyed `team_map` by index
# instead of name, and silently priced every match off team `-1`.
#
# The gate is therefore built the same way as 3a — against l02_equations.jl, an
# independent implementation written from MODEL.md rather than from the engine.
#
#   4a  synthetic chain   fabricate known parameters, price them, compare to l02
#   4b  real chain        load the gate 3c artifact off disk and run the ordinary
#                         Experiments.extract_oos_predictions path end to end
#
# 4a is exact and cheap. 4b is plumbing: it cannot check arithmetic (the true
# values are unknown) but it is the only thing that exercises the real loader,
# the real OOS lookup, and the real DataFrame assembly.
#
# ==============================================================================

using BayesianFootball
using MCMCChains
using DataFrames
using Dates
using Statistics
using Random
using Printf


# ==============================================================================
# 1. Synthetic chain construction
# ==============================================================================

"""
    tp_synthetic_chain(draws; n_chains) -> Chains

Build an `MCMCChains.Chains` holding exactly the parameter draws given, under the
site names `tp_sampled_sites` documents.

**Draw ordering is the point of this function.** Every extractor in src flattens
with `vec(Array(chain[name]))`, which is column-major over an `(n_iter, n_chains)`
slice — so flat sample `k` is iteration `((k-1) % n_iter) + 1` of chain
`((k-1) ÷ n_iter) + 1`. Different components are extracted by separate calls, and
nothing checks that they agree on that convention. If any one of them flattened
differently, λ would silently mix the μ of one draw with the α of another.

So the draws here are deliberately all DIFFERENT, and placed at the position that
convention implies. `n_chains > 1` is required for the test to have teeth: with a
single chain, row-major and column-major coincide and the bug would pass.
"""
function tp_synthetic_chain(draws::Vector{TPParams}; n_chains::Int = 2)
    n_draws = length(draws)
    n_draws % n_chains == 0 ||
        error("n_draws ($n_draws) must divide evenly into n_chains ($n_chains)")
    n_iter  = n_draws ÷ n_chains
    n_teams = length(first(draws).raw_a)

    names = tp_sampled_sites(n_teams)
    arr   = zeros(Float64, n_iter, length(names), n_chains)

    for (k, p) in enumerate(draws)
        it = ((k - 1) % n_iter) + 1
        ch = ((k - 1) ÷ n_iter) + 1
        row = vcat(p.μ, p.log_r, p.γ, p.σ_a, p.σ_d, p.raw_a, p.raw_d)
        arr[it, :, ch] = row
    end

    return Chains(arr, Symbol.(names))
end

"""
    tp_synthetic_draws(n_teams, n_draws; seed) -> Vector{TPParams}

Independent parameter draws, spread widely enough that a collapsed or duplicated
draw cannot pass by coincidence. Not prior draws — the point is coverage of the
arithmetic, not plausibility.
"""
function tp_synthetic_draws(n_teams::Int, n_draws::Int; seed::Int = 20260826)
    rng = Random.MersenneTwister(seed)
    return [TPParams(
                μ     = 0.1  + 0.3 * randn(rng),
                log_r = 3.1  + 0.4 * randn(rng),
                γ     = 0.2  + 0.2 * randn(rng),
                σ_a   = 0.15 + 0.10 * rand(rng),
                σ_d   = 0.15 + 0.10 * rand(rng),
                raw_a = randn(rng, n_teams),
                raw_d = randn(rng, n_teams),
            ) for _ in 1:n_draws]
end

"""
    tp_synthetic_fixtures(team_map; n, unmapped) -> DataFrame

Fixtures shaped like what `get_next_matches` returns: the columns
`extract_parameters` actually reads (`match_id`, `home_team`, `away_team`,
`match_date`) and nothing else.

`season_idx` is deliberately OMITTED, to exercise the documented fallback
(`goals.jl:161` falls back to `n_seasons` when the column is absent). Under
`GlobalInterception` that fallback is harmless because every season shares one μ —
gate 4c asserts exactly that, so the day a seasonal component is swapped in, the
fallback stops being harmless and the gate says so.

`unmapped` appends a fixture with a team that is not in `team_map`, to measure the
population fallback rather than assume it.
"""
function tp_synthetic_fixtures(team_map; n::Int = 6, unmapped::Bool = false)
    teams = sort(String.(collect(keys(team_map))))
    length(teams) >= 4 || error("need at least 4 mapped teams")

    rows = NamedTuple[]
    for i in 1:n
        push!(rows, (
            match_id   = 900_000 + i,
            home_team  = teams[((2i - 2) % length(teams)) + 1],
            away_team  = teams[((2i - 1) % length(teams)) + 1],
            match_date = Date(2025, ((i - 1) % 12) + 1, 15),
        ))
    end
    if unmapped
        push!(rows, (
            match_id   = 999_999,
            home_team  = "___not_a_real_team___",
            away_team  = teams[1],
            match_date = Date(2025, 3, 15),
        ))
    end
    return DataFrame(rows)
end


# ==============================================================================
# 2. GATE 4a — Synthetic-chain parity
# ==============================================================================

"""
    tp_gate_extraction_synthetic(model, fs; n_draws, n_chains, tol) -> Vector

Price a fabricated chain through the package's own `extract_parameters` and
compare every λ and r against `l02_equations.jl`.

Tolerance is 1e-12 rather than exact 0.0: extraction reassembles the arithmetic in
a different order from the Turing model (matrix ops over draws rather than
broadcasts over matches), so the last bit or two of floating point can differ
legitimately. Anything above 1e-12 is a real disagreement, not rounding.
"""
function tp_gate_extraction_synthetic(model, fs;
                                      n_draws::Int  = 8,
                                      n_chains::Int = 2,
                                      tol::Float64  = 1e-12)
    tp_assert_default(model)

    n_teams  = Int(fs.data[:n_teams])
    team_map = fs.data[:team_map]
    draws    = tp_synthetic_draws(n_teams, n_draws)
    chain    = tp_synthetic_chain(draws; n_chains = n_chains)
    df       = tp_synthetic_fixtures(team_map; n = 6)

    priced = PGm.extract_parameters(model, df, fs, chain)

    results = Any[]

    push!(results, (
        name   = "every fixture priced",
        pass   = length(priced) == nrow(df) && all(haskey(priced, Int(r.match_id)) for r in eachrow(df)),
        detail = "$(length(priced)) of $(nrow(df)) fixtures returned",
    ))

    push!(results, (
        name   = "draws preserved",
        pass   = all(length(priced[Int(r.match_id)].λ_h) == n_draws for r in eachrow(df)),
        detail = "$(n_draws) draws expected across $(n_chains) chains",
    ))

    # Compare draw by draw. l02 is scalar-per-draw; extraction is vector-over-draws.
    worst_λ = 0.0
    worst_r = 0.0
    for row in eachrow(df)
        h = team_map[row.home_team]
        a = team_map[row.away_team]
        got = priced[Int(row.match_id)]
        for (k, p) in enumerate(draws)
            λ_h, λ_a = tp_intensities(p, [h], [a])
            r_h, r_a = tp_dispersion(p)
            worst_λ  = max(worst_λ, abs(got.λ_h[k] - λ_h[1]), abs(got.λ_a[k] - λ_a[1]))
            worst_r  = max(worst_r, abs(got.r_h[k] - r_h),    abs(got.r_a[k] - r_a))
        end
    end

    push!(results, (
        name   = "λ parity vs l02",
        pass   = worst_λ <= tol,
        detail = @sprintf("max |Δλ| = %.3e over %d draws x %d fixtures x 2 sides",
                          worst_λ, n_draws, nrow(df)),
    ))

    push!(results, (
        name   = "r parity vs l02",
        pass   = worst_r <= tol,
        detail = @sprintf("max |Δr| = %.3e", worst_r),
    ))

    # A draw-ordering bug survives the above only if it happens to be the identity.
    # Check directly that distinct draws produced distinct prices.
    first_row = first(eachrow(df))
    λs = priced[Int(first_row.match_id)].λ_h
    push!(results, (
        name   = "draws not collapsed",
        pass   = length(unique(round.(λs; digits = 12))) == n_draws,
        detail = "$(length(unique(round.(λs; digits=12)))) distinct λ_h across $(n_draws) draws",
    ))

    return results
end

"""
    tp_gate_extraction_fallbacks(model, fs; n_draws) -> Vector

What extraction does with a team it has never seen, and whether the season/month
fallbacks are inert for THIS component set.

Both are measured, not assumed. `goals.jl:154-161` substitutes zeros for unmapped
teams and `n_seasons` for a missing `season_idx`; whether those are the right
population values depends entirely on which components are configured.
"""
function tp_gate_extraction_fallbacks(model, fs; n_draws::Int = 8)
    n_teams  = Int(fs.data[:n_teams])
    team_map = fs.data[:team_map]
    draws    = tp_synthetic_draws(n_teams, n_draws)
    chain    = tp_synthetic_chain(draws; n_chains = 2)
    df       = tp_synthetic_fixtures(team_map; n = 2, unmapped = true)

    priced  = PGm.extract_parameters(model, df, fs, chain)
    unm     = priced[999_999]
    away_i  = team_map[String(df[end, :away_team])]

    # Expected under a CORRECT population fallback: an unknown team carries the
    # population mean attack and defence (zero, by the zero-sum constraint) but the
    # home side still gets the global home advantage, which is not team-specific.
    want_λ_h = [exp(p.μ + p.γ + 0.0 + (tp_team_effects(p)[2])[away_i]) for p in draws]
    got_λ_h  = unm.λ_h
    drop     = maximum(abs.(got_λ_h .- want_λ_h))
    ratio    = mean(got_λ_h ./ want_λ_h)

    results = Any[]
    push!(results, (
        name   = "unmapped team keeps global home advantage",
        pass   = drop <= 1e-12,
        detail = drop <= 1e-12 ?
                 "γ_global retained for unmapped home side" :
                 @sprintf("γ_global DROPPED: λ_h is %.3fx the population value (max |Δ| %.3e)",
                          ratio, drop),
    ))

    # Season / month invariance. With GlobalInterception, μ is shared across seasons
    # and δ_month is identically zero, so neither index may change the price.
    df2 = copy(df)
    df2.match_date = [Date(2025, 11, 2) for _ in 1:nrow(df2)]
    priced2 = PGm.extract_parameters(model, df2, fs, chain)
    mid     = Int(df[1, :match_id])
    month_δ = maximum(abs.(priced[mid].λ_h .- priced2[mid].λ_h))

    push!(results, (
        name   = "month index inert for this config",
        pass   = month_δ <= 1e-12,
        detail = @sprintf("max |Δλ_h| = %.3e when match month changes (GlobalInterception ⇒ δ_month ≡ 0)",
                          month_δ),
    ))

    return results
end


# ==============================================================================
# 3. GATE 4b — Real chain, real loader
# ==============================================================================

"""
    tp_load_smoke(path) -> ExperimentResults

Load the gate 3c artifact back off disk.

Deliberately loads from PATH rather than reusing the in-memory object the smoke
returned. The saved artifact is what gates 5-7 will consume, so it is the thing
that has to be readable — an object that only exists in a REPL session is not a
result.
"""
tp_load_smoke(path::AbstractString) = Experiments.load_experiment(path)

"""
    tp_gate_extraction_real(ds, results, contract) -> (Vector, LatentStates)

Run the ordinary `Experiments.extract_oos_predictions` path and check what comes
back.

This cannot verify arithmetic — the true λ is unknown. It verifies PLUMBING: that
the splitter reproduces the same boundaries the experiment was trained on, that
the OOS fixtures at t+1 are found, that every one is priced, and that nothing is
NaN, negative, or absurd.

Note it re-derives features via the splitter-aware `create_features`
(post_processing.jl:150-155), the same overload gate 2 checks. A caller passing a
bare symbol there would silently rebuild features on a different clock.
"""
function tp_gate_extraction_real(ds, results, contract::SLContract)
    latents = Experiments.extract_oos_predictions(ds, results; force = true)
    df      = latents.df

    chain   = results.training_results.items[1][1]
    n_draws = size(chain, 1) * size(chain, 3)

    splitter   = tp_smoke_splitter(contract)
    boundaries = Data.create_id_boundaries(ds, splitter)
    oos        = DataFrame(Data.get_next_matches(ds, boundaries[1], splitter))

    out = Any[]

    push!(out, (
        name   = "OOS fixtures priced",
        pass   = nrow(df) == nrow(oos) && nrow(df) > 0,
        detail = "$(nrow(df)) rows priced, $(nrow(oos)) OOS fixtures at t+1",
    ))

    push!(out, (
        name   = "match ids match the OOS set",
        pass   = Set(df.match_id) == Set(oos.match_id),
        detail = "$(length(setdiff(Set(oos.match_id), Set(df.match_id)))) missing, " *
                 "$(length(setdiff(Set(df.match_id), Set(oos.match_id)))) unexpected",
    ))

    push!(out, (
        name   = "posterior depth preserved",
        pass   = all(length(v) == n_draws for v in df.λ_h),
        detail = "$(n_draws) draws per fixture (chain is $(size(chain,1)) x $(size(chain,3)))",
    ))

    allλ = vcat(vcat(df.λ_h...), vcat(df.λ_a...))
    allr = vcat(vcat(df.r_h...), vcat(df.r_a...))

    push!(out, (
        name   = "λ finite and positive",
        pass   = all(isfinite, allλ) && all(>(0), allλ),
        detail = @sprintf("%d values, range [%.3f, %.3f]", length(allλ), minimum(allλ), maximum(allλ)),
    ))

    push!(out, (
        name   = "r finite and positive",
        pass   = all(isfinite, allr) && all(>(0), allr),
        detail = @sprintf("range [%.2f, %.2f] — clamp at exp(±10) never binds", minimum(allr), maximum(allr)),
    ))

    # A structural model of Scottish lower-league goals that prices a side outside
    # roughly [0.2, 5] goals is not wrong arithmetically, but it is not credible.
    med = median(allλ)
    push!(out, (
        name   = "λ plausible for the league",
        pass   = 0.2 <= minimum(allλ) && maximum(allλ) <= 6.0,
        detail = @sprintf("median %.3f goals/side (league average is ~1.3)", med),
    ))

    return (out, latents)
end
