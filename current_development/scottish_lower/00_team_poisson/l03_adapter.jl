# ==============================================================================
# Model 00 — Scottish Lower protocol adapter (pure team Poisson)
# ==============================================================================
# This loader intentionally translates the legacy Model-00 implementation to the
# protocol boundary only.  l02 remains the independent density/equation referee.

using BayesianFootball
using DataFrames
using Dates
using Distributions
using Statistics
using Random

# `using ScottishLowerProtocol` copies exported bindings only when the caller
# explicitly asks for them; extending a same-named local generic would leave the
# protocol fallback untouched.  Import the hook generics we extend.
import .ScottishLowerProtocol: AbstractSLModelAdapter, sl_model, sl_model_name,
    sl_required_features, sl_assert_model_contract, sl_build_turing_model,
    sl_params_from_varinfo, sl_equation_data, sl_equation_logjoint,
    sl_sampled_sites, sl_parameter_row, sl_synthetic_n_teams,
    sl_synthetic_draws, sl_synthetic_fixtures, sl_reference_extract,
    sl_extract_parameters, sl_extract_params, sl_compute_score_matrix,
    sl_reference_grid, sl_marginal_cdf_bounds, sl_marginal_logpdf,
    sl_capabilities, sl_referee_eval, sl_result

"""Protocol adapter for Model 00's default Global/Global/TimeDecay configuration."""
struct TP00Adapter{M<:DynamicPoissonGoalsTimeDecayModel} <: AbstractSLModelAdapter
    model::M
end

TP00Adapter(; kwargs...) = TP00Adapter(tp00_model(; kwargs...))

# ------------------------------------------------------------------------------
# Identity, production path, and posterior layout
# ------------------------------------------------------------------------------

sl_model(adapter::TP00Adapter) = adapter.model
sl_model_name(::TP00Adapter) = "00_team_poisson"
sl_required_features(adapter::TP00Adapter) = BayesianFootball.Features.required_features(adapter.model)
sl_assert_model_contract(adapter::TP00Adapter) = tp00_assert_default(adapter.model)
sl_build_turing_model(adapter::TP00Adapter, fs) =
    BayesianFootball.Models.PreGame.build_turing_model(adapter.model, fs)

function sl_params_from_varinfo(::TP00Adapter, vi)
    values = Dict(string(key) => vi[key] for key in keys(vi))
    TP00Params(
        μ = Float64(values["inter.μ"]),
        γ = Float64(values["ha.γ_global"]),
        σ_a = Float64(values["dyn.σ_a"]),
        σ_d = Float64(values["dyn.σ_d"]),
        raw_a = Float64.(values["dyn.raw_a"]),
        raw_d = Float64.(values["dyn.raw_d"]),
    )
end

sl_equation_data(::TP00Adapter, fs) = tp00_equation_data(fs)
sl_equation_logjoint(adapter::TP00Adapter, params::TP00Params, data) =
    tp00_logjoint(params, data, adapter.model)

sl_sampled_sites(::TP00Adapter, n_teams::Int) = tp00_sampled_sites(n_teams)
sl_parameter_row(::TP00Adapter, p::TP00Params) =
    Float64[p.μ, p.γ, p.σ_a, p.σ_d, p.raw_a..., p.raw_d...]
function sl_synthetic_draws(::TP00Adapter, n_teams::Int, n_draws::Int; seed::Int = 20260826)
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

function sl_synthetic_fixtures(::TP00Adapter, team_map; n::Int = 6, unmapped::Bool = false)
    n > 0 || throw(ArgumentError("n must be positive"))
    teams = sort!(collect(keys(team_map)))
    isempty(teams) && throw(ArgumentError("team_map must not be empty"))
    home = [teams[mod1(i, length(teams))] for i in 1:n]
    away = [teams[mod1(i + 1, length(teams))] for i in 1:n]
    if unmapped
        home[end] = "__tp00_unknown_home__"
        away[end] = "__tp00_unknown_away__"
    end
    DataFrame(match_id = collect(90_001:(90_000 + n)), home_team = home, away_team = away,
              match_date = fill(Date(2024, 10, 19), n), season_idx = fill(1, n))
end

# Production extraction/pricing calls are deliberately direct package dispatches.
sl_extract_parameters(adapter::TP00Adapter, fixtures, fs, chain) =
    BayesianFootball.Models.PreGame.extract_parameters(adapter.model, fixtures, fs, chain)
sl_extract_params(adapter::TP00Adapter, row) =
    BayesianFootball.Predictions.extract_params(adapter.model, row)
sl_compute_score_matrix(adapter::TP00Adapter, params; max_goals::Int = 12) =
    BayesianFootball.Predictions.compute_score_matrix(adapter.model, params; max_goals)

# ------------------------------------------------------------------------------
# Independent extraction and distribution referee
# ------------------------------------------------------------------------------

"""Independent l02 arithmetic for exactly one draw and one prediction fixture."""
function sl_reference_extract(::TP00Adapter, p::TP00Params, fixture, fs)
    team_map = fs.data[:team_map]
    home_idx = get(team_map, fixture.home_team, 0)
    away_idx = get(team_map, fixture.away_team, 0)
    α, β = tp00_team_effects(p)

    # Model 00's production fallback is population-zero team effects, while the
    # global home advantage remains present for an unknown home side.
    α_h = home_idx > 0 ? α[home_idx] : 0.0
    β_h = home_idx > 0 ? β[home_idx] : 0.0
    α_a = away_idx > 0 ? α[away_idx] : 0.0
    β_a = away_idx > 0 ? β[away_idx] : 0.0
    λ_h = exp(p.μ + p.γ + α_h + β_a)
    λ_a = exp(p.μ + α_a + β_h)
    (; λ_h, λ_a, true_xg_h = λ_h, true_xg_a = λ_a)
end

function sl_reference_grid(::TP00Adapter, row, draw::Int, max_goals::Int)
    max_goals > 0 || throw(ArgumentError("max_goals must be positive"))
    λ_h, λ_a = row.λ_h[draw], row.λ_a[draw]
    home = [pdf(Poisson(λ_h), goals) for goals in 0:(max_goals - 1)]
    away = [pdf(Poisson(λ_a), goals) for goals in 0:(max_goals - 1)]
    home * away'
end

function _tp00_adapter_marginal(side::Symbol, row)
    side === :home && return row.λ_h
    side === :away && return row.λ_a
    throw(ArgumentError("side must be :home or :away, got $side"))
end

function sl_marginal_cdf_bounds(::TP00Adapter, side::Symbol, row, y::Int)
    y >= 0 || throw(ArgumentError("goal count must be non-negative"))
    λ = _tp00_adapter_marginal(side, row)
    lower = y == 0 ? 0.0 : mean(cdf(Poisson(x), y - 1) for x in λ)
    upper = mean(cdf(Poisson(x), y) for x in λ)
    (lower, upper)
end

function sl_marginal_logpdf(::TP00Adapter, side::Symbol, row, y::Int)
    y >= 0 || throw(ArgumentError("goal count must be non-negative"))
    λ = _tp00_adapter_marginal(side, row)
    log(mean(pdf(Poisson(x), y) for x in λ))
end

function sl_capabilities(::TP00Adapter)
    (; uses_home_intensity = true,
       supports_population_fallback = true,
       expected_score_dispatch = "l03_adapter.jl",
       expected_params_dispatch = "l03_adapter.jl",
       expected_sampled_sites = ("inter.μ", "ha.γ_global", "dyn.σ_a", "dyn.σ_d", "dyn.raw_a", "dyn.raw_d"),
       posterior_schema = (; varinfo_sites = ("inter.μ", "ha.γ_global", "dyn.σ_a", "dyn.σ_d", "dyn.raw_a", "dyn.raw_d"),
                             chain_columns = n -> tp00_sampled_sites(n),
                             parameter_count = n -> 4 + 2n),
       extraction_schema = (; posterior_fields = (:λ_h, :λ_a, :true_xg_h, :true_xg_a),
                              positive_fields = (:λ_h, :λ_a, :true_xg_h, :true_xg_a)),
       funnel_sites = ("dyn.σ_a", "dyn.σ_d"),
       score_matrix_normalized = false,
       conditional_independence = true,
       has_dispersion = false,
       has_dependence = false,
       score_orientation = :home_away,
       score_support = :zero_to_max_goals_minus_one,
       unknown_team_fallback = :zero_sum_population_effect,
       marginal_distribution = :Poisson,
       normalization = :raw_truncated_mass,
       production_extraction = :PreGame_extract_parameters,
       production_score_dispatch = :Predictions_compute_score_matrix)
end

# ------------------------------------------------------------------------------
# Model-local referee checks.  These complement generic gates without reusing
# production extraction/scoring calculations.
# ------------------------------------------------------------------------------
function sl_referee_eval(adapter::TP00Adapter, stage::Symbol, args...)
    if stage === :config
        model = adapter.model
        default_components = model.interception_config isa BayesianFootball.Models.PreGame.GlobalInterception &&
                             model.homeadvantage_config isa BayesianFootball.Models.PreGame.GlobalHomeAdvantage &&
                             model.dynamics_config isa BayesianFootball.Models.PreGame.TimeDecayDynamics
        return [sl_result("Model 00 l02 component contract", default_components,
                          "GlobalInterception / GlobalHomeAdvantage / TimeDecayDynamics required")]
    elseif stage === :equation
        fs, draw = args
        n = Int(fs.data[:n_teams])
        p = draw.params
        return [sl_result("Model 00 grouped versus expanded schema",
                          length(p.raw_a) == n && length(p.raw_d) == n && length(draw.θ) == 4 + 2n,
                          "4 scalar coordinates + two $n-team raw vectors")]
    elseif stage === :extraction_synthetic || stage === :extraction_fallback
        fs, draws, fixtures, priced = args
        expected = length(draws)
        fields_ok = all(haskey(priced, Int(r.match_id)) &&
                        all(hasproperty(priced[Int(r.match_id)], f) && length(getproperty(priced[Int(r.match_id)], f)) == expected
                            for f in (:λ_h, :λ_a, :true_xg_h, :true_xg_a)) for r in eachrow(fixtures))
        return [sl_result("Model 00 extraction preserves λ and true-xG draws", fields_ok,
                          "$expected draws; true_xg is identically the Poisson intensity")]
    elseif stage === :score_dispatch
        row, params = args
        return [sl_result("Model 00 pure-Poisson score semantics", adapter.model isa BayesianFootball.TypesInterfaces.AbstractPoissonModel,
                          "independent Poisson [home, away] grid; no dispersion or dependence")]
    elseif stage === :gradients || stage === :convergence || stage === :extraction_real || stage === :evaluation
        return NamedTuple[]
    end
    throw(ArgumentError("unsupported TP00Adapter referee stage: $stage"))
end
