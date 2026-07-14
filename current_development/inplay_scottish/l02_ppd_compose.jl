#=
l02_ppd_compose.jl — WP3 loader: compose pregame λ posterior with the NHPP multiplier
posterior into full-market PPDs at any in-play state.

Cut-posterior draw pairing (RESEARCH.md §3): pair pregame draw s with multiplier draw
m(s). Because log λ_i(t) = log pgλ_i + (α + β z + state + δ_time), the pregame rate is
a pure multiplier, so per multiplier draw we precompute the integration KERNEL
  K_side^m = Σ_slices exp(α^m + β^m z + γ^m·state + δ^m_time) Δt   (pg = 1)
and the paired remaining intensity is Λ_i^s = pgλ_i^s · K_i^{m(s)}.

Final-score matrix per pair (independent double-Poisson remaining, shifted by the
current score) is wrapped in Predictions.ScoreMatrix ([H, A, samples]) so EVERY
existing market implementation (1X2, OU ladder, BTTS, CS, DNB, DC, AH) prices the
in-play book unchanged — no line shifting needed, the matrix is over FINAL scores.

Requires l01_nhpp_scottish.jl loaded (for _cv/_cm/_has and NHPPXConfig).
=#

using Distributions, Random
const Pred = BayesianFootball.Predictions
const DataM = BayesianFootball.Data   # market structs: Market1X2, MarketOverUnder, ...

# ---------------------------------------------------------------------------
# 1. Per-multiplier-draw integration kernels
# ---------------------------------------------------------------------------

"""
    intensity_kernels(chain, c; gh, ga, reds_h, reds_a, t_now, Tend)
        -> (K_h::Vector, K_a::Vector)   one entry per multiplier draw

Remaining-intensity integral for a UNIT pregame rate, holding the current game state
fixed. Λ_side for a paired draw = pgλ_side_draw × K_side.
"""
function intensity_kernels(chain, c::NHPPXConfig; gh, ga, reds_h = 0, reds_a = 0,
                           t_now, Tend = c.Tend)
    remaining_intensity(chain, c; pg_h = 1.0, pg_a = 1.0, gh = gh, ga = ga,
                        reds_h = reds_h, reds_a = reds_a, t_now = t_now, Tend = Tend)
end

# ---------------------------------------------------------------------------
# 2. Composed final-score matrix (the in-play P_t)
# ---------------------------------------------------------------------------

"""
    compose_score_matrix(λh_draws, λa_draws, K_h, K_a; gh, ga, n_pairs, max_goals, rng)
        -> Pred.ScoreMatrix

Pairs pregame draws with multiplier-kernel draws (independent uniform pairing — the
two posteriors are fit on different data, RESEARCH.md §3) and builds the final-score
matrix per pair: current score (gh, ga) ⊕ double-Poisson remaining goals.
"""
function compose_score_matrix(λh_draws::AbstractVector, λa_draws::AbstractVector,
                              K_h::AbstractVector, K_a::AbstractVector;
                              gh::Int, ga::Int, n_pairs::Int = 2000,
                              max_goals::Int = 12, rng = Xoshiro(1))
    npg, nm = length(λh_draws), length(K_h)
    G = max_goals + 1
    data = zeros(Float64, G, G, n_pairs)
    for s in 1:n_pairs
        i = rand(rng, 1:npg); j = rand(rng, 1:nm)
        Λh = λh_draws[i] * K_h[j]; Λa = λa_draws[i] * K_a[j]
        dh = Poisson(max(Λh, 1e-9)); da = Poisson(max(Λa, 1e-9))
        for a in 0:(max_goals - ga), h in 0:(max_goals - gh)
            data[gh + h + 1, ga + a + 1, s] = pdf(dh, h) * pdf(da, a)
        end
        # renormalise the truncated grid so each draw's matrix sums to 1
        tot = sum(view(data, :, :, s))
        tot > 0 && (view(data, :, :, s) ./= tot)
    end
    return Pred.ScoreMatrix(data)
end

"""
    inplay_ppd(chain, c, λh_draws, λa_draws; gh, ga, reds_h, reds_a, t_now, markets, kw...)
        -> Dict{Symbol, Vector{Float64}}   selection => per-pair probability draws

One call = the full in-play book PPD at a given match state.
"""
function inplay_ppd(chain, c::NHPPXConfig, λh_draws, λa_draws;
                    gh, ga, reds_h = 0, reds_a = 0, t_now,
                    markets = default_markets(), n_pairs = 2000, max_goals = 12,
                    rng = Xoshiro(1))
    K_h, K_a = intensity_kernels(chain, c; gh = gh, ga = ga, reds_h = reds_h,
                                 reds_a = reds_a, t_now = t_now)
    S = compose_score_matrix(λh_draws, λa_draws, K_h, K_a;
                             gh = gh, ga = ga, n_pairs = n_pairs,
                             max_goals = max_goals, rng = rng)
    out = Dict{Symbol, Vector{Float64}}()
    for m in markets
        merge!(out, Pred.compute_market_probs(S, m))
    end
    return out
end

default_markets() = vcat([DataM.Market1X2(), DataM.MarketBTTS()],
                         [DataM.MarketOverUnder(k + 0.5) for k in 0:5])
