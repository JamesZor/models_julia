# src/predictions/market_inference/asian_handicap.jl

using ..Data: MarketAsianHandicap, outcomes

# Asian Handicap (whole/half lines). Returns the push-adjusted win probability
# P(win) / (P(win) + P(loss)), i.e. renormalised over the non-push outcomes so it
# is directly comparable to the de-vigged Betfair price. Quarter lines are not
# priced here (config should use whole/half lines - see MarketAsianHandicap).
function compute_market_probs(S::ScoreMatrix, market::MarketAsianHandicap)
    (max_h, max_a, n_samples) = size(S.data)
    side = market.side
    L = market.line

    win_prob = zeros(Float64, n_samples)

    @inbounds for k in 1:n_samples
        win = 0.0
        loss = 0.0
        for c in 1:max_a
            a_goals = c - 1
            for r in 1:max_h
                m = (r - 1) - a_goals          # home-minus-away margin
                p = S.data[r, c, k]
                # Bet wins iff side_margin + L > 0.
                sm = side === :home ? m : -m
                adj = sm + L
                if adj > 0
                    win += p
                elseif adj < 0
                    loss += p
                end
                # adj == 0 is a push: excluded from both.
            end
        end
        denom = win + loss
        if denom > 0
            win_prob[k] = win / denom
        end
    end

    sel = outcomes(market).bet
    return Dict(sel => win_prob)
end
