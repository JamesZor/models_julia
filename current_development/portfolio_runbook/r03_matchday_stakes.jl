# r03_matchday_stakes.jl -- pricing fixtures you have NOT played yet.
#
# Everything so far has been backtesting: `build_books` needs a final score, because a MatchBook
# carries the settlement vector so `simulate` can resolve it.
#
#   *** LIMITATION, READ THIS ***
#   `PF.build_books` SKIPS any match without a result (`haskey(scores, m_id) || return nothing`).
#   So you cannot point it at Saturday's fixtures and get a stake sheet.
#
# That is a deliberate v1 boundary, not an oversight: the module was built to be audited against
# history first. For live use you compose the same primitives yourself, which is what this file
# shows. If match-day becomes the primary use, the clean fix is to make `MatchBook.settle`
# a `Union{Nothing,Vector{Float64}}` and have `simulate` refuse unsettled books -- about ten
# lines, but it changes a public type, so it is not being done casually.
#
# The pipeline below is IDENTICAL to what build_books/stake_slate do internally. Compare it
# against src/Portfolio/book.jl and src/Portfolio/stake.jl side by side; that is the point.

include("_setup.jl")

spec = PF.BookSpec(markets = MARKETS)
policy = PF.PolicySpec(trust = PF.FlatTrust(0.25), risk = PF.SlateDrawdown(23.0),
                       cap = PF.FixedCap(0.25))

# ===================================================================
# 1. Pick the fixtures
# ===================================================================
#
# In production this is "today's card". Here we take one real match day and pretend we do not
# know the results, so the numbers are checkable against r01.

_have = Set(latents_df.match_id)
_by_date = Dict{Date,Vector{Int}}()
for r in eachrow(ds.matches)
    r.match_id in _have || continue
    push!(get!(_by_date, Date(r.match_date), Int[]), Int(r.match_id))
end
# pick the busiest day -- a full Saturday card is the case worth looking at, because that is
# where simultaneous exposure actually bites
target_date = argmax(d -> length(_by_date[d]), collect(keys(_by_date)))
fixture_ids = _by_date[target_date]

@info "match day" date = target_date fixtures = length(fixture_ids)

# ===================================================================
# 2. Price each fixture  (= build_book, minus settlement)
# ===================================================================

priced = NamedTuple[]
for m_id in fixture_ids
    row = latents_df[findfirst(==(m_id), latents_df.match_id), :]

    # (a) posterior score grid from the L1 model
    sm = Predictions.compute_score_matrix(expr.config.model,
                                          Predictions.extract_params(expr.config.model, row))
    max_h, max_a, _ = size(sm.data)
    p = vec(mean(sm.data, dims = 3)[:, :, 1]); p ./= sum(p)

    # (b) model probability of every market we price
    model_probs = Dict(string(m) => Predictions.compute_market_probs(sm, m)
                       for m in spec.markets.markets)

    # (c) quotes -> priced selections (de-arb + completeness guard live in here)
    sels = PF.extract_selections(odds, m_id, spec, model_probs)
    isempty(sels) && continue

    # (d) payoff geometry, then the Kelly portfolio on the posterior mean
    R = PF.payoff_matrix(sels, max_h, max_a, spec.exec.commission)
    a = PF.allocate(spec.allocator, p, R, spec.exec)

    # (e) Baker-McHale shrinkage -- re-solves on 128 posterior draws
    k = PF.shrink_factor(spec.shrink, sm, R, p, spec.allocator, spec.exec; seed_offset = m_id)

    push!(priced, (m_id = m_id, sels = sels, p = p, R = R, a_kelly = a.a, k = k, kkt = a.kkt))
end

@printf("\npriced %d of %d fixtures (the rest had no usable quotes)\n", length(priced), length(fixture_ids))

# ===================================================================
# 3. Apply the policy  (= stake_slate)
# ===================================================================
#
# The order matters and is the same everywhere in the module:
#     a_kelly -> x trust -> x shrink -> x risk -> cap

ctx = PF.SlateContext(1, Date(target_date), 1.0)

# trust and shrinkage, per match
staked = [begin
    a = copy(f.a_kelly)
    for j in eachindex(a)
        a[j] *= PF.trust_for(policy.trust, f.sels[j], ctx)
    end
    a .*= f.k
    a
end for f in priced]

# the drawdown budget, solved across the WHOLE day at once
kr = PF.risk_factor(policy.risk, [f.p for f in priced],
                    [priced[i].R * staked[i] for i in eachindex(staked)])
for s in staked; s .*= kr; end

# the hard cap, applied last
staked, was_capped = PF.apply_cap(policy.cap, staked)

@printf("\nrisk factor for the day: %.4f   capped: %s\n", kr, was_capped)

# ===================================================================
# 4. The stake sheet
# ===================================================================

BANKROLL = 1_000.0     # your actual bankroll, in your currency

sheet = DataFrame(match_id = Int[], market = String[], selection = Symbol[],
                  odds = Float64[], p_model = Float64[], p_market = Float64[],
                  edge = Float64[], frac = Float64[], stake = Float64[])

for (f, a) in zip(priced, staked), j in eachindex(a)
    a[j] > 0 || continue
    s = f.sels[j]
    push!(sheet, (f.m_id, s.family, s.selection, s.odds_used, s.p_model, s.p_market,
                  s.p_model - s.p_market, a[j], a[j] * BANKROLL))
end
sort!(sheet, :stake, rev = true)
for c in (:p_model, :p_market, :edge); sheet[!, c] = round.(sheet[!, c], digits = 3); end
sheet.frac  = round.(sheet.frac .* 100, digits = 2)
sheet.stake = round.(sheet.stake, digits = 2)

println("\n", "="^90, "\n=== STAKE SHEET  (bankroll $(BANKROLL)) ===\n", "="^90)
println(sheet)
@printf("\n  %d bets across %d fixtures | total staked %.2f (%.1f%% of bankroll)\n",
        nrow(sheet), length(unique(sheet.match_id)), sum(sheet.stake),
        100 * sum(sheet.stake) / BANKROLL)

println("""

  You will see rows with NEGATIVE edge that still carry a stake. That is not a bug.

  This is a PORTFOLIO Kelly solve, not a list of independent value bets: the allocator
  maximises expected log-growth over the whole 144-state score grid at once, so it will happily
  take a small negative-edge position when it hedges a larger correlated one in the same match
  (typically a draw against a big home/away position). Judge the sheet per match, not per row.
  If you want only standalone value, add a `MinEdge` filter to the policy -- but expect growth
  to fall, because you have removed the hedges.""")

println("""

  Before acting on a sheet like this:
   * `odds` is the de-arbed TRADED price from a 20-minute pre-kick-off window. On this league
     the median O/U and BTTS market has ONE trade in that window. It is not a quote you can
     necessarily get filled at, and there is no back/lay spread modelled anywhere.
   * `edge` is against the vig-removed market price. It is not a forecast of profit.
   * the risk factor above was solved under the MODEL's probabilities. If the model is
     miscalibrated, the drawdown guarantee is decorative.""")
