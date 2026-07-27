# src/predictions/market_inference/draw_no_bet.jl

using ..Data: MarketDrawNoBet, outcomes

# Draw No Bet: the draw voids, so the fair probability space is the two non-draw
# outcomes renormalised (matches the de-vigged Betfair price, which sums the two
# selections to 1.0).
function compute_market_probs(S::ScoreMatrix, market::MarketDrawNoBet)
    (max_h, max_a, n_samples) = size(S.data)

    home_prob = zeros(Float64, n_samples)
    away_prob = zeros(Float64, n_samples)

    @inbounds for k in 1:n_samples
        ph = 0.0
        pa = 0.0
        for c in 1:max_a
            limit_away = min(c - 1, max_h)
            for r in 1:limit_away
                pa += S.data[r, c, k]      # away wins (home < away)
            end
            for r in (c + 1):max_h
                ph += S.data[r, c, k]      # home wins (home > away)
            end
        end
        denom = ph + pa
        # Guard the (degenerate) all-draw sample.
        if denom > 0
            home_prob[k] = ph / denom
            away_prob[k] = pa / denom
        end
    end

    keys = outcomes(market)
    return Dict(keys.home => home_prob, keys.away => away_prob)
end
