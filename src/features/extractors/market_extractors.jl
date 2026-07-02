# src/features/extractors/market_extractors.jl

function add_feature!(F_data::Dict, config::AbstractMarketFeatureConfig, ordered_ids, team_map::Dict, ds::Data.DataStore)
    id_set = Set(ordered_ids)
    filtered_odds = subset(ds.odds, :match_id => ByRow(in(id_set)))
    odds_by_match = groupby(filtered_odds, :match_id)
    n_matches = length(odds_by_match)
    
    # We use Any/NamedTuple because keys vary based on config
    thread_results = Vector{Tuple{Int, NamedTuple}}(undef, n_matches)
    
    @threads for i in 1:n_matches
        match_df = odds_by_match[i]
        res = fit_market_implied_parameters(match_df, config)
        params = extract_parameters(config, res.minimizer)
        thread_results[i] = (res.match_id, params)
    end
    
    market_map = Dict{Int, NamedTuple}(r[1] => r[2] for r in thread_results)

    # Dynamically unroll keys and apply NaN padding
    dummy_params = extract_parameters(config, get_initial_guess(config))
    for key in keys(dummy_params)
        dict_key = Symbol("flat_market_", key)
        F_data[dict_key] = [haskey(market_map, id) ? market_map[id][key] : NaN for id in ordered_ids]
    end
end

# ------------------------------------------------------------------
# Local-intensity SMILE target (off-AD-path market inversion)
# ------------------------------------------------------------------
# Per match × strike K=0..Kmax, invert the de-vigged fair under-probability
# F^mkt(K)=P_mkt(N≤K)=prob_fair_close[:under_K5] to the implied Poisson rate Λ^mkt(K)
# solving cdf(Poisson(Λ),K)=F^mkt(K) (a 1-D root). cdf is strictly decreasing in Λ, so a
# closed form at K=0 and bisection otherwise. This is a PRICING target for the smile pillar
# (φ anchors log(λ_tot)+log_φ(K) to it) — it does NOT enter the goals likelihood and stays
# entirely off the AD path. Kept POISSON-referenced on purpose: pregame totals are ~Poisson
# while the market prices over-dispersion; that gap is the edge, an NB inversion would absorb it.
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

function add_feature!(F_data::Dict, config::MarketSmileFeature,
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
