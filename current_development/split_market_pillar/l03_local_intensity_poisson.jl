# current_development/split_market_pillar/l03_local_intensity_poisson.jl
#
# LOADER (temporary module). Single self-contained model:
#   LocalIntensitySmileDoublePoissonModel — double-Poisson {goals + xG + SUPREMACY + SMILE + outfield}.
#
# Extends l02 by REPLACING the isotropic level pillar with a per-strike LOCAL-INTENSITY SMILE pillar
# (Dupire / "Local Intensity model", docs/bets_multi/unified_kelly_postgrad_notes.md Ch.4). The
# supremacy (who-wins) pillar is KEPT (r06 showed it helps 1X2). Motivation: the per-line eval
# proved total goals N=h+a ~ Poisson(λ_h+λ_a) is too thin-tailed; the market prices a RISING
# per-strike implied intensity Λ(K) (the "intensity smile"). One λ for the whole O/U ladder cannot
# fit it.
#
# MATHS
#  Market smile target (precompute, OFF the AD path): for each match, each O/U strike K
#  (line K+0.5, under = N≤K), invert the de-vigged fair F^mkt(K)=P_mkt(N≤K)=prob_fair_close[under_K5]
#  to the implied Poisson rate Λ^mkt(K) solving  cdf(Poisson(Λ),K)=F^mkt(K)  (Def 25; 1-D root).
#  Model pillar: global shape log_φ ∈ ℝ^{Kmax+1} (φ(K)=exp(log_φ(K)); φ≡1 ⇒ Poisson). Per match m,
#  per present strike K:  log Λ^model_m(K) = log(λ_h,m+λ_a,m) + log_φ(K), anchored
#     Σ_m Σ_K w_m·mask_{m,K}·logpdf(Normal(log Λ^model_m(K), σ_smile), log Λ^mkt_m(K)).
#  σ_smile is SAMPLED (l02 release-valve). φ does NOT enter the goals likelihood — it is a
#  market-calibration/pricing object (local-vol analogy). Centre strikes subsume the old level
#  anchor; the rising tail is the new signal.
#  Per-line inference pricing: under-K priced with its OWN intensity, P(N≤K)=cdf(Poisson(λ_tot·φ(K)),K);
#  1X2/BTTS/correct-score still come from the unchanged (λ_h,λ_a) grid.
#
# DISPATCH: subtypes AbstractTimeDecayPlayerModel (NegBin default route is harmless); ships
# loader-local extract_params / compute_score_matrix + a SmileScoreMatrix and per-line O/U
# compute_market_probs override.

using Turing
using Distributions
using DataFrames

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

# Default sampled-σ priors (wide enough to converge; mean sets anchor strength).
const SUP_PRIOR   = truncated(Normal(0.10, 0.10), lower=0.02)
const SMILE_PRIOR = truncated(Normal(0.15, 0.10), lower=0.02)

_market_active(config) = config.market_on ? 1.0 : 0.0
_mok(x) = !ismissing(x) && (xf = Float64(x); !isnan(xf) && 0.02 < xf < 20.0)

# ==========================================
# 0. MARKET-SMILE TARGET (off-AD inversion)
# ==========================================
# Solve cdf(Poisson(Λ),K)=F for Λ. cdf is strictly decreasing in Λ ⇒ bisection.
function _smile_intensity(F::Float64, K::Int; lo::Float64=1e-4, hi::Float64=60.0,
                          tol::Float64=1e-9, iters::Int=80)::Float64
    Fc = clamp(F, 1e-6, 1.0 - 1e-6)
    K == 0 && return -log(Fc)               # cdf(Poisson(Λ),0)=exp(-Λ)=F  (closed form)
    a, b = lo, hi
    # f(a)=cdf(a)-F > 0 (cdf≈1 at Λ→0); f(b)=cdf(b)-F < 0 (cdf→0 at large Λ)
    for _ in 1:iters
        m = 0.5 * (a + b)
        if cdf(Poisson(m), K) - Fc > 0.0
            a = m
        else
            b = m
        end
        b - a < tol && break
    end
    return 0.5 * (a + b)
end

# Loader-local feature: per match × strike K=0..Kmax, market-implied log Λ^mkt(K) + presence mask.
Base.@kwdef struct MarketSmileFeature <: Features.AbstractFeatureConfig
    Kmax::Int = 6
end

function Features.add_feature!(F_data::Dict, config::MarketSmileFeature,
                               ordered_ids, team_map::Dict, ds::Data.DataStore)
    Kmax = config.Kmax
    nK   = Kmax + 1
    id_set = Set(ordered_ids)
    odds = subset(ds.odds, :match_id => ByRow(in(id_set)))

    # match_id -> Dict(selection_symbol => prob_fair_close)
    target = Dict{Int, Dict{Symbol, Float64}}()
    for g in groupby(odds, :match_id)
        mid = Int(first(g.match_id))
        d = Dict{Symbol, Float64}()
        for r in eachrow(g)
            ismissing(r.prob_fair_close) && continue
            d[Symbol(r.selection)] = Float64(r.prob_fair_close)
        end
        target[mid] = d
    end

    n     = length(ordered_ids)
    logΛ  = zeros(Float64, n, nK)
    mask  = zeros(Float64, n, nK)
    for (i, mid) in enumerate(ordered_ids)
        d = get(target, Int(mid), Dict{Symbol, Float64}())
        for K in 0:Kmax
            key = Symbol("under_$(K)5")
            haskey(d, key) || continue
            F = d[key]
            (1e-4 < F < 1.0 - 1e-4) || continue
            Λ = _smile_intensity(F, K)
            (isfinite(Λ) && Λ > 1e-4) || continue
            logΛ[i, K + 1] = log(Λ)
            mask[i, K + 1] = 1.0
        end
    end
    F_data[:flat_smile_logΛ] = logΛ
    F_data[:flat_smile_mask] = mask
    F_data[:smile_Kmax]      = Kmax
end

# Builder-side unpacking (mirrors l02).
function _unpack_market(data)
    market_mask  = Float64.(_mok.(data[:flat_market_λ_home]) .& _mok.(data[:flat_market_λ_away]))
    market_log_h = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_home]]
    market_log_a = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_away]]
    return market_log_h, market_log_a, market_mask
end

function _unpack_xg(data)
    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    away_xg_raw = coalesce.(data[:flat_away_xg], NaN)
    xg_mask = Float64.(.!isnan.(home_xg_raw) .& .!isnan.(away_xg_raw))
    home_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in home_xg_raw]
    away_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in away_xg_raw]
    return home_xg, away_xg, xg_mask
end

function _centre_ratings(hG, hD, hM, hF, aG, aD, aM, aF, base_rating)
    h_G_c = hG .- base_rating
    h_O_c = (hD .+ hM .+ hF) .- (10.0 * base_rating)
    a_G_c = aG .- base_rating
    a_O_c = (aD .+ aM .+ aF) .- (10.0 * base_rating)
    return h_G_c, h_O_c, a_G_c, a_O_c
end

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct LocalIntensitySmileDoublePoissonModel{
    I<:PreGame.AbstractInterceptionConfig,
    P<:PreGame.OutfieldPlayerDynamicsConfig,
    D<:PreGame.AbstractDispersionConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    K<:PreGame.AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig,
    M<:Features.AbstractMarketFeatureConfig
  } <: PreGame.AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P
      dispersion_config::D            # config-compat; unused by the Poisson likelihood
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      smile_feature::MarketSmileFeature = MarketSmileFeature()
      ν_xg::Distribution               = truncated(Normal(3.0, 0.5), lower=0.5)
      σ_supremacy_prior::Distribution  = SUP_PRIOR
      σ_smile_prior::Distribution      = SMILE_PRIOR
      smile_shape_sd::Float64          = 0.5   # prior sd on the global log_φ(K) shape
      market_on::Bool                  = true  # false => BOTH market pillars OFF (control)
      supremacy_weight::Float64        = 1.0
      smile_weight::Float64            = 1.0
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_local_intensity_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    home_G_ratings::Vector{Float64}, home_D_ratings::Vector{Float64},
    home_M_ratings::Vector{Float64}, home_F_ratings::Vector{Float64},
    away_G_ratings::Vector{Float64}, away_D_ratings::Vector{Float64},
    away_M_ratings::Vector{Float64}, away_F_ratings::Vector{Float64},
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    xg_mask::Vector{Float64},
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    market_mask::Vector{Float64},
    smile_logΛ::Matrix{Float64},     # [n_matches × nK]
    smile_mask::Matrix{Float64},     # [n_matches × nK]
    n_strikes::Int,
    market_active::Float64,
    supremacy_weight::Float64,
    smile_weight::Float64,
    smile_shape_sd::Float64,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::LocalIntensitySmileDoublePoissonModel
)
    # --- 1. LOAD COMPONENTS ---
    ν_xg    ~ config.ν_xg
    σ_sup   ~ config.σ_supremacy_prior
    σ_smile ~ config.σ_smile_prior
    log_φ   ~ filldist(Normal(0.0, smile_shape_sd), n_strikes)   # global smile shape

    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(PreGame.build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(PreGame.build_dynamics(config.player_dynamics_config, n_teams))

    # --- 2. VECTORIZED INDEXING & MATH ---
    base_rating = config.player_ratings_feature.tracker.prior_mean
    h_G_c, h_O_c, a_G_c, a_O_c = _centre_ratings(
        home_G_ratings, home_D_ratings, home_M_ratings, home_F_ratings,
        away_G_ratings, away_D_ratings, away_M_ratings, away_F_ratings, base_rating)

    att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
    def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)
    att_a = (p_dyn.w_G_att .* a_G_c) .+ (p_dyn.w_Outfield_att .* a_O_c)
    def_a = (p_dyn.w_G_def .* a_G_c) .+ (p_dyn.w_Outfield_def .* a_O_c)

    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    log_λ_h = clamp.(int_m .+ view(ha, home_team_indices) .+ att_h .+ def_a, -20.0, 20.0)
    log_λ_a = clamp.(int_m                                .+ att_a .+ def_h, -20.0, 20.0)

    kap_h = view(kap, home_team_indices)
    kap_a = view(kap, away_team_indices)
    λ_h = kap_h .* exp.(log_λ_h) .+ 1e-6
    λ_a = kap_a .* exp.(log_λ_a) .+ 1e-6

    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- Pillar B: Actual Goals (Poisson) ---
    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar A: xG (Gamma) ---
    xg_rate_h = exp.(log_λ_h) .+ 1e-6
    xg_rate_a = exp.(log_λ_a) .+ 1e-6
    xg_rate_h = ifelse.(isnan.(xg_rate_h) .| isinf.(xg_rate_h), one.(xg_rate_h), xg_rate_h)
    xg_rate_a = ifelse.(isnan.(xg_rate_a) .| isinf.(xg_rate_a), one.(xg_rate_a), xg_rate_a)
    ll_xg_h = logpdf.(Gamma.(ν_xg, xg_rate_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, xg_rate_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C1: SUPREMACY (who-wins), σ SAMPLED, gated by market_active ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)
    model_sup = market_rate_h .- market_rate_a
    m_sup     = market_log_λ_h .- market_log_λ_a
    ll_sup    = logpdf.(Normal.(model_sup, σ_sup), m_sup)
    Turing.@addlogprob! market_active * supremacy_weight *
        sum(ll_sup .* match_weights .* market_mask)

    # --- Pillar C2: LOCAL-INTENSITY SMILE (per-strike totals), σ SAMPLED ---
    # log Λ^model_m(K) = log(λ_h+λ_a)_m + log_φ(K)  anchored to market log Λ^mkt_m(K).
    log_λ_tot  = log.(λ_h .+ λ_a)                          # [n_matches]
    model_logΛ = log_λ_tot .+ reshape(log_φ, 1, n_strikes) # [n_matches × nK]
    ll_smile   = logpdf.(Normal.(model_logΛ, σ_smile), smile_logΛ)   # [n_matches × nK]
    Turing.@addlogprob! market_active * smile_weight *
        sum(ll_smile .* smile_mask .* match_weights)
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::LocalIntensitySmileDoublePoissonModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.XGFeature(),
        model.market_feature_config,
        model.smile_feature,
        model.player_ratings_feature,
        Features.TimeIndicesFeature()
    ]
end

function PreGame.build_turing_model(config::LocalIntensitySmileDoublePoissonModel, feature_set)
    data = feature_set.data

    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12

    date_deltas   = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.player_dynamics_config.days_half_life)

    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    month_idx  = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    hG = Vector{Float64}(data[:flat_home_G_rating]); hD = Vector{Float64}(data[:flat_home_D_rating])
    hM = Vector{Float64}(data[:flat_home_M_rating]); hF = Vector{Float64}(data[:flat_home_F_rating])
    aG = Vector{Float64}(data[:flat_away_G_rating]); aD = Vector{Float64}(data[:flat_away_D_rating])
    aM = Vector{Float64}(data[:flat_away_M_rating]); aF = Vector{Float64}(data[:flat_away_F_rating])

    home_xg, away_xg, xg_mask = _unpack_xg(data)
    mlh, mla, mmask           = _unpack_market(data)

    smile_logΛ = Matrix{Float64}(data[:flat_smile_logΛ])
    smile_mask = Matrix{Float64}(data[:flat_smile_mask])
    n_strikes  = size(smile_logΛ, 2)

    return build_local_intensity_engine(
        home_ids, away_ids, season_ids, month_idx,
        home_goals, away_goals, match_weights,
        hG, hD, hM, hF, aG, aD, aM, aF,
        home_xg, away_xg, xg_mask,
        mlh, mla, mmask,
        smile_logΛ, smile_mask, n_strikes,
        _market_active(config), config.supremacy_weight, config.smile_weight,
        config.smile_shape_sd,
        n_teams, n_seasons, n_months, config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function PreGame.extract_parameters(model::LocalIntensitySmileDoublePoissonModel, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams])
    team_map    = data[:team_map]
    ratings_map = data[:player_ratings_map]
    nK = Int(data[:smile_Kmax]) + 1

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3)
    base_r = model.player_ratings_feature.tracker.prior_mean

    # Global smile shape φ(K): [n_samples × nK] (same for every match, per posterior draw).
    φ_mat = Matrix{Float64}(undef, n_samples, nK)
    for k in 1:nK
        φ_mat[:, k] = exp.(vec(Array(chain[Symbol("log_φ[$k]")])))
    end

    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        h_id = get(team_map, row.home_team, -1)
        a_id = get(team_map, row.away_team, -1)

        m_ratings = get(ratings_map, Int(row.match_id), Dict())
        h_G = get(m_ratings, ("home","G"), 0.0); h_D = get(m_ratings, ("home","D"), 0.0)
        h_M = get(m_ratings, ("home","M"), 0.0); h_F = get(m_ratings, ("home","F"), 0.0)
        a_G = get(m_ratings, ("away","G"), 0.0); a_D = get(m_ratings, ("away","D"), 0.0)
        a_M = get(m_ratings, ("away","M"), 0.0); a_F = get(m_ratings, ("away","F"), 0.0)

        h_G_c = h_G - base_r; h_O_c = (h_D + h_M + h_F) - (10.0 * base_r)
        a_G_c = a_G - base_r; a_O_c = (a_D + a_M + a_F) - (10.0 * base_r)

        att_h = (p_dyn_nt.w_G_att .* h_G_c) .+ (p_dyn_nt.w_Outfield_att .* h_O_c)
        def_h = (p_dyn_nt.w_G_def .* h_G_c) .+ (p_dyn_nt.w_Outfield_def .* h_O_c)
        att_a = (p_dyn_nt.w_G_att .* a_G_c) .+ (p_dyn_nt.w_Outfield_att .* a_O_c)
        def_a = (p_dyn_nt.w_G_def .* a_G_c) .+ (p_dyn_nt.w_Outfield_def .* a_O_c)

        γ_h = h_id > 0 ? ha_mat[:, h_id] : zeros(n_samples)
        κ_h = h_id > 0 ? kap_mat[:, h_id] : ones(n_samples)
        κ_a = a_id > 0 ? kap_mat[:, a_id] : ones(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = hasproperty(row, :month_idx) ? Int(row.month_idx) : 1
        μ_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        att_a .+ def_h, -20.0, 20.0)

        λ_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_a = κ_a .* exp.(log_λ_a) .+ 1e-6

        results[Int(row.match_id)] = (;
            λ_h, λ_a,
            λ_tot = λ_h .+ λ_a,
            φ = φ_mat,                 # [n_samples × nK] global smile
            θ_1 = log.(λ_h), θ_2 = log.(λ_a), θ_3 = zeros(n_samples), ρ = zeros(n_samples),
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a),
        )
    end
    return results
end

# ==========================================
# 5. PREDICTION OVERRIDES (loader-local)
# ==========================================
# Carry the (λ_h,λ_a) grid PLUS per-sample per-strike model intensities Λ^model(K)=λ_tot·φ(K).
struct SmileScoreMatrix <: Pred.AbstractScoreMatrix
    grid::Pred.ScoreMatrix          # [max_goals × max_goals × n_samples] for 1X2/BTTS/correct-score
    Λ::Matrix{Float64}              # [nK × n_samples] per-strike total intensity (K = row-1)
end

function _poisson_score(λ_h, λ_a; max_goals::Int=12)
    n = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)
    @inbounds for k in 1:n
        @. p_h = pdf(Poisson(λ_h[k]), goals)
        @. p_a = pdf(Poisson(λ_a[k]), goals)
        for j in 1:max_goals, i in 1:max_goals
            S[i, j, k] = p_h[i] * p_a[j]
        end
    end
    return Pred.ScoreMatrix(S)
end

Pred.extract_params(::LocalIntensitySmileDoublePoissonModel, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ)

function Pred.compute_score_matrix(::LocalIntensitySmileDoublePoissonModel, params; max_goals::Int=12)
    grid = _poisson_score(params.λ_h, params.λ_a; max_goals)
    # Λ^model(K) = λ_tot · φ(K), shape [nK × n_samples]
    Λ = transpose(params.λ_tot .* params.φ)          # (n_samples × nK)' -> (nK × n_samples)
    return SmileScoreMatrix(grid, Matrix{Float64}(Λ))
end

# Per-line O/U pricing via Def 25: P(N≤K)=cdf(Poisson(Λ^model(K)),K). Fall back to the grid for
# strikes beyond the learned smile range.
function Pred.compute_market_probs(S::SmileScoreMatrix, m::Data.MarketOverUnder)
    K = Int(floor(m.line))
    nK = size(S.Λ, 1)
    if K < 0 || K + 1 > nK
        return Pred.compute_market_probs(S.grid, m)   # outside learned smile -> grid
    end
    n = size(S.Λ, 2)
    under = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        under[s] = cdf(Poisson(S.Λ[K + 1, s]), K)
    end
    over = 1.0 .- under
    keys = Data.outcomes(m)
    return Dict(keys.over => over, keys.under => under)
end

# Everything else (1X2 / BTTS / correct-score) prices from the unchanged grid.
Pred.compute_market_probs(S::SmileScoreMatrix, m::Data.AbstractMarket) =
    Pred.compute_market_probs(S.grid, m)

println("[l03] local-intensity loader ready: LocalIntensitySmileDoublePoissonModel " *
        "{goals + xG + supremacy + per-strike SMILE + outfield}; knobs market_on / supremacy_weight / smile_weight")
