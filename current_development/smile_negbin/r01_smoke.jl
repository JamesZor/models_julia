# current_development/smile_negbin/r01_smoke.jl
#
# RUNNER. AD-safety + round-trip smoke test for the smile-NegBin engine.
#
# No database, no `.cache/`, no DataStore. Everything is hand-rolled, in the spirit of
# `orderbook_layer2/r01_apparatus_smoke.jl` and `test/portfolio_tests.jl`: the point is to prove
# the plumbing works before spending hours of MCMC on it.
#
#   S1  the Turing model COMPILES under ReverseDiff(compile=true) — the production AD backend —
#       and samples without NaN / -Inf lp.
#   S2  `extract_parameters` returns finite, positive r_h / r_a (i.e. the dispersion submodel is
#       genuinely wired in, not silently defaulted).
#   S3  `extract_params -> compute_score_matrix -> compute_market_probs` produces valid
#       probabilities for 1X2, BTTS, correct-score and O/U, and returns a `SmileScoreMatrix` (if
#       it returned a plain `ScoreMatrix`, O/U would have silently de-smiled).
#   S4  POISSON-LIMIT CHECK: with r -> 1e6 the NegBin grid must converge to the Poisson parent's
#       grid on identical λ. This is what distinguishes "a genuine generalization" from "a
#       divergent reimplementation that happens to run".
#
# NAMING NOTE: globals are `sn_`-prefixed. This is normally noise, but these runners are executed
# in a LONG-LIVED shared REPL, where a bare `params = ...` collides with the const binding
# `using Distributions` puts in Main and a bare `rate(...)` definition fails outright.
#
# USAGE
#   julia --project -t auto
#   julia> using BayesianFootball
#   julia> include("current_development/smile_negbin/r01_smoke.jl")

using BayesianFootball
using Random, Distributions, DataFrames, Statistics, Printf, LinearAlgebra
using Turing, ReverseDiff        # AutoReverseDiff is re-exported by Turing (see Samplers)

include(joinpath(@__DIR__, "l01_smile_negbin_engine.jl"))
include(joinpath(@__DIR__, "l02_smile_negbin_predict.jl"))

Random.seed!(20260813)

# ===================================================================
# 0. Tiny assertion harness (collect, don't abort — one run reports everything)
# ===================================================================
const SN_FAILURES = String[]
function sn_check(ok::Bool, msg::AbstractString)
    @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg)
    ok || push!(SN_FAILURES, msg)
    return ok
end
empty!(SN_FAILURES)

# ===================================================================
# 1. Synthetic corpus
# ===================================================================
# Drawn from the model's OWN generative story (ratings -> log λ -> NegBin goals, Gamma xG,
# noisy market λ, smile-shaped per-strike intensities) so that sampling has something coherent to
# find. It is a plumbing test, not a recovery study — but incoherent data would make an
# AD/geometry failure indistinguishable from a misspecification failure.

const N_TEAMS   = 8
const N_SEASONS = 2
const N_MATCHES = 140
const KMAX      = 4
const NK        = KMAX + 1
const R_TRUE_H  = 9.0            # mild overdispersion; the thing the Poisson parent cannot fit
const R_TRUE_A  = 6.0
const PHI_TRUE  = [0.88, 0.93, 1.00, 1.09, 1.22]   # rising per-strike intensity smile
const BASE_RATING = 6.5

sn_teams = ["T$(i)" for i in 1:N_TEAMS]
team_map = Dict(sn_teams[i] => i for i in 1:N_TEAMS)
att_true = 0.30 .* randn(N_TEAMS)
def_true = 0.25 .* randn(N_TEAMS)

home_ids = Int[]; away_ids = Int[]
for _ in 1:N_MATCHES
    h = rand(1:N_TEAMS); a = rand(setdiff(1:N_TEAMS, h))
    push!(home_ids, h); push!(away_ids, a)
end
season_ids  = rand(1:N_SEASONS, N_MATCHES)
month_ids   = rand(1:12, N_MATCHES)
date_deltas = rand(0:420, N_MATCHES)
match_ids   = collect(1001:(1000 + N_MATCHES))

# Ratings live around the tracker's prior mean (6.5); deviations carry the team signal.
_rating_draw(t, s) = BASE_RATING + s * (att_true[t] + def_true[t]) + 0.10 * randn()
hG = [_rating_draw(home_ids[m], 0.5) for m in 1:N_MATCHES]
hD = [_rating_draw(home_ids[m], 0.5) for m in 1:N_MATCHES]
hM = [_rating_draw(home_ids[m], 0.5) for m in 1:N_MATCHES]
hF = [_rating_draw(home_ids[m], 0.5) for m in 1:N_MATCHES]
aG = [_rating_draw(away_ids[m], 0.5) for m in 1:N_MATCHES]
aD = [_rating_draw(away_ids[m], 0.5) for m in 1:N_MATCHES]
aM = [_rating_draw(away_ids[m], 0.5) for m in 1:N_MATCHES]
aF = [_rating_draw(away_ids[m], 0.5) for m in 1:N_MATCHES]

# True intensities: league base + home edge + team attack/defence.
λ_h_true = [exp(0.25 + 0.20 + att_true[home_ids[m]] - def_true[away_ids[m]]) for m in 1:N_MATCHES]
λ_a_true = [exp(0.25        + att_true[away_ids[m]] - def_true[home_ids[m]]) for m in 1:N_MATCHES]

home_goals = [rand(NegativeBinomial(R_TRUE_H, R_TRUE_H / (R_TRUE_H + λ_h_true[m]))) for m in 1:N_MATCHES]
away_goals = [rand(NegativeBinomial(R_TRUE_A, R_TRUE_A / (R_TRUE_A + λ_a_true[m]))) for m in 1:N_MATCHES]

# xG present on ~85% of matches (exercises xg_mask, incl. the `missing` -> NaN coalesce path).
ν_true = 3.0
home_xg = Union{Float64,Missing}[rand() < 0.85 ? rand(Gamma(ν_true, λ_h_true[m] / ν_true)) : missing
                                 for m in 1:N_MATCHES]
away_xg = Union{Float64,Missing}[rand() < 0.85 ? rand(Gamma(ν_true, λ_a_true[m] / ν_true)) : missing
                                 for m in 1:N_MATCHES]
# A present-but-zero xG must survive the builder's ε-floor rather than sending Gamma logpdf to -Inf.
home_xg[1] = 0.0

# Market λ: noisy view of the truth, absent on ~10% (exercises market_mask), with one degenerate
# value the builder's plausibility filter (0.02 < λ < 20) is supposed to reject.
_market_draw(λ) = rand() < 0.90 ? λ * exp(0.08 * randn()) : missing
market_λ_h = Union{Float64,Missing}[_market_draw(λ_h_true[m]) for m in 1:N_MATCHES]
market_λ_a = Union{Float64,Missing}[_market_draw(λ_a_true[m]) for m in 1:N_MATCHES]
market_λ_h[2] = 357.0

# Smile: per-strike intensity Λ(K) = λ_tot · φ(K), a couple of strikes dropped per match.
smile_logΛ = zeros(Float64, N_MATCHES, NK)
smile_mask = zeros(Float64, N_MATCHES, NK)
for m in 1:N_MATCHES
    λ_tot = λ_h_true[m] + λ_a_true[m]
    for k in 1:NK
        rand() < 0.12 && continue
        smile_logΛ[m, k] = log(λ_tot * PHI_TRUE[k]) + 0.05 * randn()
        smile_mask[m, k] = 1.0
    end
end

# `player_ratings_map`: match_id -> ("home"/"away", position) -> rating, as the extractor reads it.
ratings_map = Dict{Int, Dict{Tuple{String,String}, Float64}}()
for m in 1:N_MATCHES
    ratings_map[match_ids[m]] = Dict(
        ("home","G") => hG[m], ("home","D") => hD[m], ("home","M") => hM[m], ("home","F") => hF[m],
        ("away","G") => aG[m], ("away","D") => aD[m], ("away","M") => aM[m], ("away","F") => aF[m],
    )
end

sn_fs = BayesianFootball.FeatureSet(Dict{Symbol,Any}(
    :n_teams            => N_TEAMS,
    :n_seasons          => N_SEASONS,
    :dates              => date_deltas,
    :flat_home_ids      => home_ids,
    :flat_away_ids      => away_ids,
    :season_indices     => season_ids,
    :flat_months        => month_ids,
    :flat_home_goals    => home_goals,
    :flat_away_goals    => away_goals,
    :flat_home_G_rating => hG, :flat_home_D_rating => hD,
    :flat_home_M_rating => hM, :flat_home_F_rating => hF,
    :flat_away_G_rating => aG, :flat_away_D_rating => aD,
    :flat_away_M_rating => aM, :flat_away_F_rating => aF,
    :flat_home_xg       => home_xg,
    :flat_away_xg       => away_xg,
    :flat_market_λ_home => market_λ_h,
    :flat_market_λ_away => market_λ_a,
    :flat_smile_logΛ    => smile_logΛ,
    :flat_smile_mask    => smile_mask,
    :smile_Kmax         => KMAX,
    :team_map           => team_map,
    :player_ratings_map => ratings_map,
))

# ===================================================================
# 2. The engine — r02's `src_sup40_sw40` cell, NegBin sibling
# ===================================================================
smile_negbin_model() = DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    dispersion_config      = PreGame.HomeAwayDispersion(),
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = Features.PlayerRatingsFeature(
                                 Features.BayesianTracker(BASE_RATING, 1.0, 0.5, 0.01)),
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax = KMAX),
    market_on              = true,
    supremacy_weight       = 0.4,
    smile_weight           = 0.4,
)

sn_model = smile_negbin_model()

println("\n", "="^90)
println("S0  required_features / build_turing_model")
println("="^90)

sn_req = Features.required_features(sn_model)
sn_check(length(sn_req) == 9, "required_features returns 9 configs (got $(length(sn_req)))")
sn_check(any(f -> f isa Features.MarketSmileFeature, sn_req),
         "required_features includes MarketSmileFeature")

sn_turing_model = PreGame.build_turing_model(sn_model, sn_fs)
sn_check(sn_turing_model isa Turing.DynamicPPL.Model, "build_turing_model returns a DynamicPPL.Model")

# ===================================================================
# 3. S1 — compile under ReverseDiff and sample
# ===================================================================
println("\n", "="^90)
println("S1  NUTS under AutoReverseDiff(compile=true)  —  the production AD backend")
println("="^90)

const N_WARMUP  = 150
const N_SAMPLES = 150

sn_t0 = time()
sn_chain = sample(
    sn_turing_model,
    NUTS(N_WARMUP, 0.65, max_depth = 8),
    N_SAMPLES;
    progress = false,
    adtype   = AutoReverseDiff(compile = true),
)
@printf("  sampled %d draws in %.1f s\n", N_SAMPLES, time() - sn_t0)

sn_lp = vec(Array(sn_chain[:lp]))
sn_check(all(isfinite, sn_lp), "lp is finite on every draw (no NaN / -Inf rejections)")
sn_check(std(sn_lp) > 0.0, "lp actually moved (chain is not stuck at the init point)")

sn_par_names = string.(names(sn_chain))
sn_check("disp.log_r" in sn_par_names,
         "chain carries disp.log_r    (dispersion submodel is wired in)")
sn_check("disp.δ_r_home" in sn_par_names,
         "chain carries disp.δ_r_home (home/away split is wired in)")
sn_check(any(startswith.(sn_par_names, "log_φ")),
         "chain carries log_φ (smile pillar survived the edit)")

# Divergence rate is diagnostic, not a gate — 150 draws on synthetic data proves nothing about
# real-fold geometry. Printed so a pathological value is visible rather than silent.
if :numerical_error in Symbol.(sn_par_names)
    @printf("  (divergences: %.1f%%)\n", 100 * mean(vec(Array(sn_chain[:numerical_error]))))
end

# ===================================================================
# 4. S2 — extract_parameters, and the dispersion it must now emit
# ===================================================================
println("\n", "="^90)
println("S2  extract_parameters")
println("="^90)

sn_df_predict = DataFrame(
    match_id   = match_ids,
    home_team  = [sn_teams[home_ids[m]] for m in 1:N_MATCHES],
    away_team  = [sn_teams[away_ids[m]] for m in 1:N_MATCHES],
    season_idx = season_ids,
    month_idx  = month_ids,
)

sn_latents = PreGame.extract_parameters(sn_model, sn_df_predict, sn_fs, sn_chain)
sn_check(length(sn_latents) == N_MATCHES, "one latent NamedTuple per match")

sn_nt1 = sn_latents[match_ids[1]]
for k in (:λ_h, :λ_a, :λ_tot, :φ, :r_h, :r_a)
    sn_check(haskey(sn_nt1, k), "latent NamedTuple carries :$k")
end
sn_check(all(isfinite, sn_nt1.r_h) && all(>(0), sn_nt1.r_h), "r_h finite and strictly positive")
sn_check(all(isfinite, sn_nt1.r_a) && all(>(0), sn_nt1.r_a), "r_a finite and strictly positive")
sn_check(size(sn_nt1.φ) == (N_SAMPLES, NK),
         "φ is [n_samples × nK] = ($N_SAMPLES, $NK), got $(size(sn_nt1.φ))")
sn_check(all(isfinite, sn_nt1.λ_h) && all(>(0), sn_nt1.λ_h), "λ_h finite and strictly positive")
sn_check(maximum(sn_nt1.λ_tot .- (sn_nt1.λ_h .+ sn_nt1.λ_a)) < 1e-12, "λ_tot == λ_h + λ_a")
@printf("  posterior mean r_h = %.2f   r_a = %.2f   (true %.1f / %.1f)\n",
        mean(sn_nt1.r_h), mean(sn_nt1.r_a), R_TRUE_H, R_TRUE_A)

# The latent DataFrame the simulator actually consumes, built by the real function.
sn_latent_df = BayesianFootball.Experiments._latent_state_dict_to_df(sn_latents)
sn_check(:r_h in propertynames(sn_latent_df), "latent DataFrame carries :r_h")
sn_row = first(eachrow(sn_latent_df))

# ===================================================================
# 5. S3 — the prediction round trip
# ===================================================================
println("\n", "="^90)
println("S3  extract_params -> compute_score_matrix -> compute_market_probs")
println("="^90)

sn_params = Pred.extract_params(sn_model, sn_row)
sn_check(haskey(sn_params, :r_h),
         "extract_params passes r_h through (our method won dispatch, not the generic one)")

sn_S = Pred.compute_score_matrix(sn_model, sn_params)
sn_check(sn_S isa Pred.SmileScoreMatrix,
         "compute_score_matrix returns a SmileScoreMatrix (a bare ScoreMatrix would de-smile O/U)")
sn_check(size(sn_S.Λ) == (NK, N_SAMPLES), "Λ is [nK × n_samples], got $(size(sn_S.Λ))")

sn_grid = sn_S.grid.data
sn_check(all(isfinite, sn_grid) && all(>=(0), sn_grid), "score grid finite and non-negative")
sn_grid_mass = [sum(view(sn_grid, :, :, k)) for k in 1:size(sn_grid, 3)]
sn_check(minimum(sn_grid_mass) > 0.999,
         @sprintf("grid mass >= 0.999 on every draw (min %.6f) — max_goals=12 truncation negligible",
                  minimum(sn_grid_mass)))

sn_markets = Any[
    Data.Market1X2(),
    Data.MarketBTTS(),
    Data.MarketCorrectScore(),
    Data.MarketOverUnder(2.5),
    Data.MarketOverUnder(0.5),
    Data.MarketOverUnder(7.5),   # beyond the learned smile -> grid fall-back
]

for mkt in sn_markets
    probs = Pred.compute_market_probs(sn_S, mkt)
    label = string(mkt)
    allp  = reduce(vcat, values(probs))
    ok_rng = all(isfinite, allp) && all(p -> -1e-12 <= p <= 1 + 1e-12, allp)
    sn_check(ok_rng, "$label: every probability finite and in [0,1]")

    tot = reduce(.+, values(probs))
    if mkt isa Data.MarketCorrectScore
        # correct-score is a partial partition (scores beyond max_goals are not enumerated)
        sn_check(all(t -> t <= 1 + 1e-9, tot) && minimum(tot) > 0.9,
                 @sprintf("%s: outcome mass <= 1 and > 0.9 (min %.6f)", label, minimum(tot)))
    else
        sn_check(maximum(abs.(tot .- 1.0)) < 2e-3,
                 @sprintf("%s: outcomes sum to 1 (max dev %.2e)", label, maximum(abs.(tot .- 1.0))))
    end
end

# The O/U route must price off the smile, not off the grid: those two disagree by construction
# whenever φ(K) != 1, and agreeing exactly would mean the SmileScoreMatrix dispatch never fired.
sn_ou = Data.MarketOverUnder(2.5)
sn_p_smile = Pred.compute_market_probs(sn_S, sn_ou)[Data.outcomes(sn_ou).under]
sn_p_grid  = Pred.compute_market_probs(sn_S.grid, sn_ou)[Data.outcomes(sn_ou).under]
@printf("  O/U 2.5 under: smile %.4f vs grid %.4f (mean over draws)\n",
        mean(sn_p_smile), mean(sn_p_grid))
sn_check(mean(abs.(sn_p_smile .- sn_p_grid)) > 1e-6,
         "O/U priced through the smile, not the grid (the two differ, as they must)")

# ===================================================================
# 6. S4 — Poisson-limit check: r -> ∞ must reproduce the parent's grid
# ===================================================================
println("\n", "="^90)
println("S4  Poisson limit (r = 1e6) vs src's _smile_poisson_grid on identical λ")
println("="^90)

λ_h_chk = collect(sn_nt1.λ_h)
λ_a_chk = collect(sn_nt1.λ_a)
r_big   = fill(1e6, length(λ_h_chk))

g_negbin  = _smile_negbin_grid(λ_h_chk, λ_a_chk, r_big, r_big; max_goals = 12).data
g_poisson = Pred._smile_poisson_grid(λ_h_chk, λ_a_chk; max_goals = 12).data

max_abs = maximum(abs.(g_negbin .- g_poisson))
sn_check(max_abs < 1e-6,
         @sprintf("NegBin grid -> Poisson grid as r -> ∞ (max abs diff %.3e)", max_abs))

# And with the FITTED r it must NOT match — otherwise the dispersion is decorative again.
g_fitted = _smile_negbin_grid(λ_h_chk, λ_a_chk,
                              collect(sn_nt1.r_h), collect(sn_nt1.r_a); max_goals = 12).data
@printf("  fitted-r vs Poisson: max abs diff %.3e\n", maximum(abs.(g_fitted .- g_poisson)))
sn_check(maximum(abs.(g_fitted .- g_poisson)) > 1e-4,
         "fitted r produces a materially different grid from Poisson (dispersion is load-bearing)")

# ===================================================================
# 7. Verdict
# ===================================================================
println("\n", "="^90)
if isempty(SN_FAILURES)
    println("SMOKE TEST PASSED — all checks green.")
else
    @printf("SMOKE TEST FAILED — %d check(s):\n", length(SN_FAILURES))
    for f in SN_FAILURES
        println("  - ", f)
    end
end
println("="^90)

nothing
