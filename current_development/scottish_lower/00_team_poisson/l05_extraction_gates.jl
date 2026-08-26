# ==============================================================================
# Model 00 — GATE 4 : EXTRACTION (Pure Poisson)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Gate 4 verifies that the priced model matches the fitted model:
#   4a: Synthetic-chain parity against l02_equations.jl (exact, ~1e-10)
#   4b: Real-chain loading and plumbing via Experiments.extract_oos_predictions
#   4c: Extraction fallbacks for unmapped teams
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
# 1. Synthetic Chain Construction
# ==============================================================================

function tp00_synthetic_chain(draws::Vector{TP00Params}; n_chains::Int = 2)
    n_draws = length(draws)
    n_draws % n_chains == 0 ||
        error("n_draws ($n_draws) must divide evenly into n_chains ($n_chains)")
    n_iter  = n_draws ÷ n_chains
    n_teams = length(first(draws).raw_a)

    names = tp00_sampled_sites(n_teams)
    arr   = zeros(Float64, n_iter, length(names), n_chains)

    for (k, p) in enumerate(draws)
        it = ((k - 1) % n_iter) + 1
        ch = ((k - 1) ÷ n_iter) + 1
        row = vcat(p.μ, p.γ, p.σ_a, p.σ_d, p.raw_a, p.raw_d)
        arr[it, :, ch] = row
    end

    return Chains(arr, Symbol.(names))
end

function tp00_synthetic_draws(n_teams::Int, n_draws::Int; seed::Int = 20260826)
    rng = Random.MersenneTwister(seed)
    return [TP00Params(
                μ     = 0.1  + 0.3 * randn(rng),
                γ     = 0.2  + 0.2 * randn(rng),
                σ_a   = 0.15 + 0.10 * rand(rng),
                σ_d   = 0.15 + 0.10 * rand(rng),
                raw_a = randn(rng, n_teams),
                raw_d = randn(rng, n_teams),
            ) for _ in 1:n_draws]
end


# ==============================================================================
# 2. GATE 4a — Synthetic Chain Parity
# ==============================================================================

function tp00_gate_extraction_synthetic(model::DynamicPoissonGoalsTimeDecayModel, fs; n_draws = 8, n_chains = 2)
    tp00_assert_default(model)
    n_teams = Int(fs.data[:n_teams])
    draws   = tp00_synthetic_draws(n_teams, n_draws)
    chain   = tp00_synthetic_chain(draws; n_chains = n_chains)

    # Test fixtures spanning various teams
    team_names = sort(collect(keys(fs.data[:team_map])))
    test_df = DataFrame(
        match_id   = [101, 102, 103, 104, 105, 106],
        home_team  = [team_names[1], team_names[2], team_names[3], team_names[4], team_names[5], team_names[6]],
        away_team  = [team_names[2], team_names[1], team_names[4], team_names[3], team_names[6], team_names[5]],
        match_date = fill(Date(2024, 10, 19), 6),
        season_idx = fill(1, 6),
    )

    extracted = PG.extract_parameters(model, test_df, fs, chain)

    results = []
    max_diff_λ = 0.0

    for row in eachrow(test_df)
        mid = row.match_id
        res = extracted[mid]
        h_idx = fs.data[:team_map][row.home_team]
        a_idx = fs.data[:team_map][row.away_team]

        for k in 1:n_draws
            p = draws[k]
            λ_h_ref, λ_a_ref = tp00_intensities(p, [h_idx], [a_idx])
            max_diff_λ = max(max_diff_λ, abs(res.λ_h[k] - λ_h_ref[1]))
            max_diff_λ = max(max_diff_λ, abs(res.λ_a[k] - λ_a_ref[1]))
        end
    end

    push!(results, (
        name   = "λ parity vs l02",
        pass   = max_diff_λ <= 1e-10,
        detail = @sprintf("max |Δλ| = %.3e over %d draws x %d fixtures", max_diff_λ, n_draws, nrow(test_df)),
    ))

    # Check distinct draws
    first_res = extracted[test_df.match_id[1]]
    distinct_λ = length(unique(first_res.λ_h))
    push!(results, (
        name   = "draws not collapsed",
        pass   = distinct_λ == n_draws,
        detail = "$distinct_λ distinct λ_h across $n_draws draws",
    ))

    return results
end


# ==============================================================================
# 3. GATE 4b — Real Chain Plumbing
# ==============================================================================

function tp00_gate_extraction_real(ds, loaded_results, contract::SLContract)
    latents = Experiments.extract_oos_predictions(ds, loaded_results; force = true)
    df = latents.df
    results = []

    push!(results, (
        name   = "OOS fixtures priced",
        pass   = nrow(df) > 0,
        detail = "$(nrow(df)) rows priced",
    ))

    all_finite = true
    all_positive = true
    for row in eachrow(df)
        if any(!isfinite, row.λ_h) || any(!isfinite, row.λ_a)
            all_finite = false
        end
        if any(x -> x <= 0, row.λ_h) || any(x -> x <= 0, row.λ_a)
            all_positive = false
        end
    end

    push!(results, (
        name   = "λ finite and positive",
        pass   = all_finite && all_positive,
        detail = all_finite && all_positive ? "all posterior λ > 0 and finite" : "non-positive or non-finite detected",
    ))

    med_λ_h = median(vcat(df.λ_h...))
    push!(results, (
        name   = "λ plausible for the league",
        pass   = 0.5 <= med_λ_h <= 3.0,
        detail = @sprintf("median λ_h = %.3f goals (expected ~1.0-2.0)", med_λ_h),
    ))

    return (results, latents)
end


# ==============================================================================
# 4. GATE 4c — Fallbacks
# ==============================================================================

function tp00_gate_extraction_fallbacks(model::DynamicPoissonGoalsTimeDecayModel, fs)
    test_df = DataFrame(
        match_id   = [9999],
        home_team  = ["non_existent_home_team"],
        away_team  = ["non_existent_away_team"],
        match_date = [Date(2024, 10, 19)],
        season_idx = [1],
    )

    n_teams = Int(fs.data[:n_teams])
    draws   = tp00_synthetic_draws(n_teams, 4)
    chain   = tp00_synthetic_chain(draws; n_chains = 2)

    extracted = PG.extract_parameters(model, test_df, fs, chain)
    res = extracted[9999]

    results = []
    push!(results, (
        name   = "unmapped team produces finite output",
        pass   = all(isfinite, res.λ_h) && all(isfinite, res.λ_a),
        detail = "population fallback generates finite λ",
    ))

    return results
end
