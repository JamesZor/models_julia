# current_development/matchday_2026_08_08/r06_level_bias.jl
#
# ═══════════════════════════════════════════════════════════════════════════════════════════
#  IS THE GOAL-LEVEL BIAS REAL?  Measured on the full walk-forward, not on one slate.
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
#   julia --project -t 16 current_development/matchday_2026_08_08/r06_level_bias.jl
#
# ───────────────────────────────────────────────────────────────────────────────────────────
# THE CLAIM UNDER TEST
# ───────────────────────────────────────────────────────────────────────────────────────────
#
# Two match-day slates on 2026-08-08 both showed the model's expected total goals sitting ABOVE
# the market's, on every single fixture:
#
#     ScottishLower  funnel engine          10/10 fixtures, mean +0.23 goals
#     ScottishUpper  player-level xG engine  6/6  fixtures, mean +0.25 goals
#
# 16 of 16, two leagues, two engines that share almost no code. A sign test gives p ~ 1.5e-5.
# But 16 fixtures from one weekend is still one weekend, and both slates came in low-scoring, so
# the honest question is whether this is a property of the MODEL or of that Saturday.
#
# This file answers it on the walk-forward instead: every out-of-sample match the engine has
# ever predicted, each with a market price alongside. Hundreds of matches, dozens of rounds,
# multiple seasons.
#
# ───────────────────────────────────────────────────────────────────────────────────────────
# WHY THE GAP IS MEASURED AGAINST THE MARKET, NOT AGAINST GOALS
# ───────────────────────────────────────────────────────────────────────────────────────────
#
# Realised goals are Poisson-noisy: at ~2.7 goals/match you need hundreds of matches before a
# 0.24 shift is visible above the noise. The MARKET's implied total is not noisy in that way —
# it is a forecast, like ours. So `model - market` isolates a systematic disagreement and
# converges far faster than `model - actual`.
#
# It does assume the market is roughly unbiased on the level. That is the assumption most likely
# to be true of anything in this system: it is the most heavily traded number on the card.
#
# ───────────────────────────────────────────────────────────────────────────────────────────
# A PLAUSIBLE MECHANISM, FROM THE PROJECT'S OWN NOTES
# ───────────────────────────────────────────────────────────────────────────────────────────
#
# `compute_score_matrix` integrates the scoreline grid over the λ POSTERIOR. For a right-skewed
# posterior `E[λ] > median(λ)`, so averaging over draws pushes expected goals up, and pushes
# harder the wider the posterior is — i.e. worst exactly where the data is thinnest.
#
# The `totals-compression-is-denoising` note already concluded "use median(λ) not mean". This is
# the same observation arriving from the other direction. If the mechanism is right, the bias
# should be LARGER where the posterior is wider: early season, and thin leagues.
#
# Section 4 tests that prediction directly, because a mechanism that survives its own falsifier
# is worth acting on and a bare correlation is not.

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Distributions

const DD  = BayesianFootball.Data
const EXP = BayesianFootball.Experiments
const PF  = BayesianFootball.Portfolio
const MD  = BayesianFootball.MatchDay

# The graduated src engine, 40 folds of walk-forward on ScottishLower.
#
# NOT `data/funnel_full/…`: those were saved with `TeamFunnelFlexDPGoalsModel`, a prototype type
# that only ever existed in a loader file, so JLD2 cannot reconstruct them into a session where
# that type is absent. Anything you want to re-analyse later has to be trained against a type
# that lives in `src`.
const EXP_PATH = "./data/experiments/plus_minus_biweek/funnel_winner_20260729_074452"

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 1. Out-of-sample latents for every fold
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# `extract_oos_predictions` walks every fold and predicts that fold's NEXT round via
# `get_next_matches` — the same call `MatchDay.select_split` now uses to choose a serving fold.
# So these latents are produced exactly the way match-day latents are, which is what makes the
# comparison meaningful rather than approximate.

ds   = DD.load_datastore_cached(DD.ScottishLower())
expr = EXP.load_experiment(EXP_PATH)

@info "experiment" model = nameof(typeof(expr.config.model)) folds = length(expr.training_results)

oos = EXP.extract_oos_predictions(ds, expr)
lat = oos.df
@info "out-of-sample latents" matches = nrow(lat)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 2. Build the books, so model and market sit on the same rows
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# `build_books` gives, per match, the posterior score grid AND the de-vigged market probability
# of every quoted selection. Both halves of the comparison from one object, priced by the same
# code that stakes on match day.

spec  = PF.BookSpec(markets = MD.MatchDaySpec().markets)
books = PF.build_books(spec, lat, expr, ds.odds, ds; require_result = true)
@info "books" n = length(books)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 3. Model total vs market total, per match
# ═══════════════════════════════════════════════════════════════════════════════════════════

"Least-squares Poisson total against the de-vigged Over probabilities of the whole O/U ladder."
function fit_market_total(b)
    over = Dict{Float64,Float64}()
    for l in unique(s.line for s in b.sels if s.group == "OverUnder")
        grp = [s for s in b.sels if s.group == "OverUnder" && s.line == l]
        length(grp) == 2 || continue
        tot = sum(s.p_market for s in grp)
        tot > 0 || continue
        over[l] = only(s.p_market for s in grp if startswith(String(s.selection), "over_")) / tot
    end
    length(over) >= 2 || return NaN            # one line is not a ladder; refuse to fit
    best, bl = Inf, NaN
    for lam in 0.40:0.01:6.50
        e = sum((ccdf(Poisson(lam), floor(Int, l)) - p)^2 for (l, p) in over)
        e < best && (best = e; bl = lam)
    end
    return bl
end

md_dates = Dict(ds.matches.match_id[i] => ds.matches.match_date[i] for i in 1:nrow(ds.matches))
md_tourn = Dict(ds.matches.match_id[i] => ds.matches.tournament_id[i] for i in 1:nrow(ds.matches))
md_seas  = Dict(ds.matches.match_id[i] => ds.matches.season[i] for i in 1:nrow(ds.matches))

rows = NamedTuple[]
for b in books
    n = 12
    eh = sum(h * b.p_grid[PF.grid_index(h, a, n)] for h in 0:n-1, a in 0:n-1)
    ea = sum(a * b.p_grid[PF.grid_index(h, a, n)] for h in 0:n-1, a in 0:n-1)
    lm = fit_market_total(b)
    isnan(lm) && continue
    d = get(md_dates, b.m_id, nothing); d === nothing && continue
    push!(rows, (match_id = b.m_id, date = Date(d), season = get(md_seas, b.m_id, "?"),
                 tournament = get(md_tourn, b.m_id, 0), month = month(d),
                 model_tot = eh + ea, market_tot = lm, gap = eh + ea - lm,
                 goals = b.settle === nothing ? missing : missing))
end
G = DataFrame(rows)

# attach the realised total, for the third column of the comparison
sc = Dict(ds.matches.match_id[i] => (ds.matches.home_score[i], ds.matches.away_score[i])
          for i in 1:nrow(ds.matches))
G.actual = [haskey(sc, m) && !ismissing(sc[m][1]) ? sc[m][1] + sc[m][2] : missing for m in G.match_id]

println("\n", "="^96, "\n  MODEL TOTAL vs MARKET TOTAL — full walk-forward\n", "="^96)
n   = nrow(G)
mg  = mean(G.gap); sdg = std(G.gap)
pos = count(>(0), G.gap)
t   = mg / (sdg / sqrt(n))
@printf("\n  matches                    %d\n", n)
@printf("  mean gap (model − market)  %+.3f goals   (sd %.3f)\n", mg, sdg)
@printf("  median gap                 %+.3f\n", median(G.gap))
@printf("  model above market on      %d of %d  (%.1f%%)\n", pos, n, 100pos/n)
@printf("  t-statistic                %+.1f\n", t)
@printf("\n  mean model %.2f   mean market %.2f   mean ACTUAL %.2f\n",
        mean(G.model_tot), mean(G.market_tot), mean(skipmissing(G.actual)))
println("""
  A t-stat past about 3 means the level offset is real and not a story about one weekend.
  Compare `mean market` against `mean ACTUAL`: if the MARKET is also above the realised total,
  the shortfall is the sample being low-scoring and only the MODEL−MARKET column is diagnostic.""")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 4. Does the bias behave the way the mechanism predicts?
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# If the cause is posterior spread (E[λ] > median λ), the gap must be LARGER where the posterior
# is wider — early in a season, and in whichever division is more thinly observed. A constant
# gap everywhere is a different animal: a fixed intercept error, which would point at the prior
# or at the shot→goal conversion rather than at Jensen.
#
# Both are fixable. They are fixed DIFFERENTLY, which is why it is worth knowing which.

println("\n--- by season ---")
show(sort(combine(groupby(G, :season), nrow => :n,
                  :gap => (x -> round(mean(x), digits = 3)) => :mean_gap), :season),
     allrows = true, allcols = true); println()

println("\n--- by tournament (56 = League One, 57 = League Two) ---")
show(combine(groupby(G, :tournament), nrow => :n,
             :gap => (x -> round(mean(x), digits = 3)) => :mean_gap,
             :model_tot => (x -> round(mean(x), digits = 2)) => :model,
             :market_tot => (x -> round(mean(x), digits = 2)) => :market),
     allrows = true, allcols = true); println()

println("\n--- by month of season (Aug = early, posterior widest) ---")
show(sort(combine(groupby(G, :month), nrow => :n,
                  :gap => (x -> round(mean(x), digits = 3)) => :mean_gap), :month),
     allrows = true, allcols = true); println()

println("""
  READ THE MONTH TABLE AGAINST THE MECHANISM. A gap that SHRINKS as the season fills in is the
  posterior-spread story, and the fix is to use the posterior median (free, principled). A gap
  that is FLAT across months is an intercept error, and the fix is a constant offset or a
  retrained prior. A gap that GROWS is neither and means this analysis is wrong.""")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 5. Would correcting it have helped?
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# Scale each match's λ so the model's total lands on `model_tot - offset`, rebuild, and rescore.
# Totals only — this correction cannot touch 1X2, which is a supremacy question, not a level one.
#
# The offset is the MEAN GAP MEASURED ABOVE, so this is in-sample and therefore optimistic. It
# is a sanity check that the correction moves the right things in the right direction, not an
# estimate of what it is worth. A walk-forward offset (fit on months 1..t, applied at t+1) is
# the honest version and the obvious next step if this looks promising.

OFFSET = round(mg, digits = 2)
@printf("\n  applying a −%.2f goal offset (in-sample; treat as directional only)\n", OFFSET)

function totals_logloss(books, ds_scores; offset = 0.0, lat = nothing, expr = nothing, spec = nothing)
    tot = Float64[]
    for b in books
        haskey(ds_scores, b.m_id) || continue
        s = ds_scores[b.m_id]
        (ismissing(s[1]) || ismissing(s[2])) && continue
        g = s[1] + s[2]
        for l in unique(x.line for x in b.sels if x.group == "OverUnder")
            grp = [x for x in b.sels if x.group == "OverUnder" && x.line == l]
            length(grp) == 2 || continue
            pm = [x.p_model for x in grp]; pm ./= sum(pm)
            i_over = findfirst(x -> startswith(String(x.selection), "over_"), grp)
            p_over = pm[i_over]
            # shift the implied total by `offset` using a Poisson re-read of the same line
            if offset != 0.0
                lam  = max(0.05, _lambda_from_over(p_over, l) - offset)
                p_over = ccdf(Poisson(lam), floor(Int, l))
            end
            y = g > l
            push!(tot, -log(max(y ? p_over : 1 - p_over, 1e-12)))
        end
    end
    return mean(tot), length(tot)
end

"Invert P(Over line) back to the Poisson rate that produced it."
function _lambda_from_over(p, line)
    lo, hi = 0.05, 8.0
    for _ in 1:60
        mid = (lo + hi) / 2
        ccdf(Poisson(mid), floor(Int, line)) < p ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

ll0, n0 = totals_logloss(books, sc; offset = 0.0)
ll1, _  = totals_logloss(books, sc; offset = OFFSET)
@printf("\n  totals log loss   before %.4f   after %.4f   change %+.4f   (%d O/U groups)\n",
        ll0, ll1, ll1 - ll0, n0)
println(ll1 < ll0 ?
    "  ← the offset IMPROVES totals calibration. Next: fit it walk-forward, not in-sample." :
    "  ← the offset does NOT improve it, so the gap is not costing accuracy where it is priced.\n" *
    "    That would mean the level disagreement lives somewhere the O/U ladder does not see.")

println("\n  n = $(nrow(G)) matches. This is the sample the one-slate numbers did not have.")
