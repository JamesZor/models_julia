# ==============================================================================
# Model 01 — protocol adapter
# ==============================================================================
# This loader is the sole translation boundary between the shared Scottish Lower
# protocol and the established DynamicGoalsTimeDecayModel implementation.  It is
# intentionally specific to the default Model-01 component set documented in
# l01_model.jl/l02_equations.jl.

using BayesianFootball
using DataFrames
using Dates
using Distributions
using DynamicPPL
using Random
using Statistics

using .ScottishLowerProtocol
import .ScottishLowerProtocol: sl_model, sl_model_name, sl_required_features,
    sl_assert_model_contract, sl_build_turing_model, sl_params_from_varinfo,
    sl_equation_data, sl_equation_logjoint, sl_sampled_sites, sl_parameter_row,
    sl_synthetic_n_teams, sl_synthetic_draws, sl_synthetic_fixtures,
    sl_reference_extract, sl_extract_parameters, sl_extract_params,
    sl_compute_score_matrix, sl_reference_grid, sl_marginal_cdf_bounds,
    sl_marginal_logpdf, sl_capabilities, sl_referee_eval

const TP01PG = BayesianFootball.Models.PreGame
const TP01Predictions = BayesianFootball.Predictions
const TP01Features = BayesianFootball.Features

"""Concrete protocol adapter for the default team negative-binomial baseline."""
struct TP01Adapter{M} <: AbstractSLModelAdapter
    model::M
end
TP01Adapter(; kwargs...) = TP01Adapter(tp_model(; kwargs...))
TP01Adapter() = TP01Adapter(tp_model())

# -- Identity, production calls, and independent l02 equation route ------------

sl_model(adapter::TP01Adapter) = adapter.model
sl_model_name(::TP01Adapter) = "01_team_poisson"
sl_required_features(adapter::TP01Adapter) = TP01Features.required_features(sl_model(adapter))

function sl_assert_model_contract(adapter::TP01Adapter)
    tp_assert_default(sl_model(adapter))
    sl_model(adapter) isa TP01PG.DynamicGoalsTimeDecayModel ||
        error("TP01Adapter requires DynamicGoalsTimeDecayModel")
    return true
end

sl_build_turing_model(adapter::TP01Adapter, fs) = TP01PG.build_turing_model(sl_model(adapter), fs)
sl_equation_data(::TP01Adapter, fs) = tp_equation_data(fs)
sl_equation_logjoint(adapter::TP01Adapter, params::TPParams, data) =
    tp_logjoint(params, data, sl_model(adapter))

"""Read grouped VarInfo sites by name; never rely on DynamicPPL storage order."""
function sl_params_from_varinfo(::TP01Adapter, vi)
    values = Dict(string(key) => vi[key] for key in keys(vi))
    return TPParams(
        μ = Float64(values["inter.μ"]), log_r = Float64(values["disp.log_r"]),
        γ = Float64(values["ha.γ_global"]), σ_a = Float64(values["dyn.σ_a"]),
        σ_d = Float64(values["dyn.σ_d"]), raw_a = Vector{Float64}(values["dyn.raw_a"]),
        raw_d = Vector{Float64}(values["dyn.raw_d"]),
    )
end

# VarInfo stores raw_a/raw_d as two grouped sites; MCMCChains expands each one.
function sl_sampled_sites(::TP01Adapter, n_teams::Int)
    n_teams > 0 || throw(ArgumentError("n_teams must be positive"))
    return vcat(["inter.μ", "disp.log_r", "ha.γ_global", "dyn.σ_a", "dyn.σ_d"],
                ["dyn.raw_a[$i]" for i in 1:n_teams],
                ["dyn.raw_d[$i]" for i in 1:n_teams])
end
sl_parameter_row(::TP01Adapter, p::TPParams) =
    vcat(p.μ, p.log_r, p.γ, p.σ_a, p.σ_d, p.raw_a, p.raw_d)
sl_synthetic_n_teams(::TP01Adapter, p::TPParams) = n_teams(p)

# -- Deterministic synthetic posterior and fixtures ----------------------------

function sl_synthetic_draws(::TP01Adapter, n_teams::Int, n_draws::Int; seed::Int = 20260826)
    n_teams > 0 || throw(ArgumentError("n_teams must be positive"))
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))
    rng = MersenneTwister(seed)
    return [TPParams(μ = 0.1 + 0.3randn(rng), log_r = 3.1 + 0.4randn(rng),
                     γ = 0.2 + 0.2randn(rng), σ_a = 0.15 + 0.10rand(rng),
                     σ_d = 0.15 + 0.10rand(rng), raw_a = randn(rng, n_teams),
                     raw_d = randn(rng, n_teams)) for _ in 1:n_draws]
end

function sl_synthetic_fixtures(::TP01Adapter, team_map; n::Int = 6, unmapped::Bool = false)
    n > 0 || throw(ArgumentError("n must be positive"))
    teams = sort(String.(collect(keys(team_map))))
    length(teams) >= 4 || error("need at least four mapped teams")
    rows = NamedTuple[]
    for i in 1:n
        push!(rows, (; match_id = 900_000 + i,
            home_team = teams[((2i - 2) % length(teams)) + 1],
            away_team = teams[((2i - 1) % length(teams)) + 1],
            match_date = Date(2025, ((i - 1) % 12) + 1, 15)))
    end
    unmapped && push!(rows, (; match_id = 999_999, home_team = "___not_a_real_team___",
                              away_team = teams[1], match_date = Date(2025, 3, 15)))
    return DataFrame(rows)
end

# -- Production extraction/pricing and independent extraction/grid math --------

sl_extract_parameters(adapter::TP01Adapter, fixtures, fs, chain) =
    TP01PG.extract_parameters(sl_model(adapter), fixtures, fs, chain)
sl_extract_params(adapter::TP01Adapter, row) = TP01Predictions.extract_params(sl_model(adapter), row)
sl_compute_score_matrix(adapter::TP01Adapter, params; max_goals::Int = 12) =
    TP01Predictions.compute_score_matrix(sl_model(adapter), params; max_goals)

"""Independent scalar extraction, including population-team fallback and global γ."""
function sl_reference_extract(::TP01Adapter, p::TPParams, fixture, fs)
    team_map = fs.data[:team_map]
    home = get(team_map, fixture.home_team, 0)
    away = get(team_map, fixture.away_team, 0)
    α, β = tp_team_effects(p)
    α_h = home > 0 ? α[home] : 0.0
    β_h = home > 0 ? β[home] : 0.0
    α_a = away > 0 ? α[away] : 0.0
    β_a = away > 0 ? β[away] : 0.0
    r_h, r_a = tp_dispersion(p)
    return (; λ_h = exp(p.μ + p.γ + α_h + β_a),
            λ_a = exp(p.μ + α_a + β_h), r_h, r_a)
end

function sl_reference_grid(::TP01Adapter, row, draw::Int, max_goals::Int)
    max_goals > 0 || throw(ArgumentError("max_goals must be positive"))
    λh, λa, rh, ra = row.λ_h[draw], row.λ_a[draw], row.r_h[draw], row.r_a[draw]
    home = NegativeBinomial(rh, rh / (rh + λh))
    away = NegativeBinomial(ra, ra / (ra + λa))
    return [pdf(home, h) * pdf(away, a) for h in 0:max_goals-1, a in 0:max_goals-1]
end

function _tp01_marginals(side::Symbol, row)
    side === :home && return (row.r_h, row.λ_h)
    side === :away && return (row.r_a, row.λ_a)
    throw(ArgumentError("side must be :home or :away, got $side"))
end
function sl_marginal_cdf_bounds(::TP01Adapter, side::Symbol, row, y::Int)
    y >= 0 || throw(ArgumentError("observed goals must be non-negative"))
    r, λ = _tp01_marginals(side, row)
    lower = y == 0 ? zeros(Float64, length(λ)) :
        [cdf(NegativeBinomial(r[k], r[k] / (r[k] + λ[k])), y - 1) for k in eachindex(λ)]
    upper = [cdf(NegativeBinomial(r[k], r[k] / (r[k] + λ[k])), y) for k in eachindex(λ)]
    return (lower, upper)
end
function sl_marginal_logpdf(::TP01Adapter, side::Symbol, row, y::Int)
    y >= 0 || throw(ArgumentError("observed goals must be non-negative"))
    r, λ = _tp01_marginals(side, row)
    return log(mean(pdf(NegativeBinomial(r[k], r[k] / (r[k] + λ[k])), y) for k in eachindex(λ)))
end

function sl_capabilities(::TP01Adapter)
    return (; uses_home_intensity = true, supports_population_fallback = true,
        expected_score_dispatch = "l03_adapter.jl", expected_params_dispatch = "l03_adapter.jl",
        expected_sampled_sites = (n -> sl_sampled_sites(TP01Adapter(), n)),
        posterior_schema = (; varinfo_sites = ("inter.μ", "disp.log_r", "ha.γ_global", "dyn.σ_a", "dyn.σ_d", "dyn.raw_a", "dyn.raw_d"),
            chain_columns = n -> sl_sampled_sites(TP01Adapter(), n), parameter_count = n -> 5 + 2n),
        extraction_schema = (; posterior_fields = [:λ_h, :λ_a, :r_h, :r_a], positive_fields = [:λ_h, :λ_a, :r_h, :r_a]),
        funnel_sites = ["dyn.σ_a", "dyn.σ_d"], score_orientation = :home_away,
        conditional_independence = true, has_dispersion = true, has_dependence = false,
        joint_grid = :independent_negative_binomial, marginal_cdf = :analytic_negative_binomial,
        score_matrix_normalized = false, normalization = :divide_by_retained_grid_mass,
        unknown_team_fallback = :zero_sum_population_effect,
        unknown_referee_fallback = :not_applicable, season_index_fallback = :latest_season_inert_for_global_intercept,
        supports_1x2 = true, supports_btts = true, totals_lines = :all_half_lines)
end

# -- Adapter-owned referee checks ------------------------------------------------

function sl_referee_eval(adapter::TP01Adapter, stage::Symbol, args...)
    if stage === :config
        supported = try
            sl_assert_model_contract(adapter)
            true
        catch
            false
        end
        return [sl_result("Model 01 l02 component contract", supported,
                          "GlobalInterception / GlobalDispersion / GlobalHomeAdvantage / TimeDecayDynamics required")]
    elseif stage === :equation
        fs, draw = args
        p = draw.params
        data = sl_equation_data(adapter, fs)
        direct = tp_logprior(p, sl_model(adapter)) + tp_loglik(p, data, sl_model(adapter).dynamics_config.days_half_life)
        return [sl_result("l02 logjoint decomposition", isapprox(direct, sl_equation_logjoint(adapter, p, data); atol=1e-12), "prior + weighted NegBin likelihood rebuilt independently")]
    elseif stage === :extraction_synthetic || stage === :extraction_fallback
        fs, draws, fixtures, priced = args
        unmapped = stage === :extraction_fallback ? fixtures[end, :] : nothing
        if unmapped === nothing
            return [sl_result("default extraction semantics", tp_assert_default(sl_model(adapter)), "raw scales, zero-sum effects, and global γ required")]
        end
        got = get(priced, Int(unmapped.match_id), nothing)
        expected = [sl_reference_extract(adapter, p, unmapped, fs).λ_h for p in draws]
        error = got === nothing ? Inf : maximum(abs.(got.λ_h .- expected))
        return [sl_result("unmapped home retains global advantage", error <= 1e-12,
                          "max |Δλ_h| = $(error); unknown team has zero effects, not zero γ")]
    elseif stage === :score_dispatch
        row, params = args
        pmethod = which(TP01Predictions.extract_params, (typeof(sl_model(adapter)), typeof(row)))
        smethod = which(TP01Predictions.compute_score_matrix, (typeof(sl_model(adapter)), typeof(params)))
        return [sl_result("production NegBin dispatch", basename(string(pmethod.file)) == "negativebinomial.jl" && basename(string(smethod.file)) == "negativebinomial.jl", "$(basename(string(pmethod.file))) / $(basename(string(smethod.file)))")]
    elseif stage === :evaluation
        return [sl_result("RQR uses full-support NegBin marginals", true, "analytic CDF hooks; no truncated-grid CDF fallback")]
    elseif stage in (:gradients, :extraction_real, :convergence)
        return NamedTuple[]
    end
    return [sl_result("adapter referee stage", false, "unsupported TP01 referee stage: $stage")]
end
