#=
l01_inplay_inverse.jl  —  Loader for the in-play market-implied λ decay study.

Stage 1: clock alignment (wall-clock betfair ticks -> match-minute) + game state.
Stage 2: per-bin vig stripping of in-play traded prices -> fair probabilities.
Stage 2b: score-conditioned inversion (remaining-λ) + naive full-match baseline,
          reusing the existing scoreline-matrix engine in
          src/features/market_inverse_utils.jl.

Reuses (do NOT modify src/ during prototyping):
  - Features.build_probability_matrix(config, θ, max_goals)
  - Features.get_initial_guess(config)
  - Features.extract_parameters(config, θ)
  - config structs: DoublePoissonMarketFeature (default), DixonColesMarketFeature, ...

Conventions verified in Stage 0 (Ireland .cache):
  - in-play  <=>  ds.betfair_odds.minutes_to_kickoff (mtk) in (0, ~130]  (positive = AFTER kickoff)
  - betfair last-traded overround ~ 1.0 (tiny vig), but we still normalise per market
  - match-min 90 lands at wall-clock ~105-118 (HT ~15min + stoppage baked into mtk)
=#

using DataFrames
using Distributions
using Optim
using LinearAlgebra
using Statistics

const Features = BayesianFootball.Features

# ---------------------------------------------------------------------------
# 0. Market / selection bookkeeping
# ---------------------------------------------------------------------------

# Over/Under integer thresholds present in the data (over_05 .. over_55 -> k = 0..5)
const OU_KS = 0:5

"Map a market group symbol -> the selection symbols that make up that (fair=1) market."
function market_groups()
    groups = Dict{Symbol,Vector{Symbol}}()
    groups[:result_1x2] = [:home, :draw, :away]
    groups[:btts]       = [:btts_yes, :btts_no]
    for k in OU_KS
        groups[Symbol("ou_$(k)5")] = [Symbol("over_$(k)5"), Symbol("under_$(k)5")]
    end
    # Correct-score: all cs_* selections (incl. cs_any_other_*) form one exhaustive market.
    groups[:cs] = Symbol[]  # filled dynamically from the data per match
    return groups
end

is_cs(sel::Symbol) = startswith(String(sel), "cs_")

# ---------------------------------------------------------------------------
# 1a. Game state from incidents (match-minute step functions)
# ---------------------------------------------------------------------------

"Match minute for an incident row: time (+ added_time when present & not the 999 sentinel)."
function _incident_minute(r)::Int
    at = (ismissing(r.added_time) || r.added_time == 999) ? 0 : Int(r.added_time)
    return Int(r.time) + at
end

"Goals for a match, sorted by match-minute. Columns: mm, is_home, incident_class."
function match_goals(ds, mid::Integer)
    g = subset(ds.incidents, :match_id => ByRow(==(mid)),
                              :incident_type => ByRow(==("goal")))
    isempty(g) && return DataFrame(mm=Int[], is_home=Bool[], incident_class=String[])
    g = transform(g, AsTable(:) => ByRow(_incident_minute) => :mm)
    sort!(g, :mm)
    return select(g, :mm, :is_home, :incident_class)
end

"Red cards (straight red or second yellow) for a match, sorted by match-minute."
function match_reds(ds, mid::Integer)
    c = subset(ds.incidents, :match_id => ByRow(==(mid)),
                              :incident_type => ByRow(==("card")))
    isempty(c) && return DataFrame(mm=Int[], is_home=Bool[])
    isred(x) = !ismissing(x) && (occursin("red", lowercase(String(x))) ||
                                 occursin("yellowred", lowercase(String(x))) ||
                                 occursin("secondyellow", lowercase(String(x))))
    c = subset(c, :incident_class => ByRow(isred))
    isempty(c) && return DataFrame(mm=Int[], is_home=Bool[])
    c = transform(c, AsTable(:) => ByRow(_incident_minute) => :mm)
    sort!(c, :mm)
    return select(c, :mm, :is_home)
end

# ---------------------------------------------------------------------------
# 1b. Goal-jump anchoring: locate each goal's wall-clock time in the feed
# ---------------------------------------------------------------------------

"Sorted in-play price series for one selection. Columns: mtk, traded_price."
function _price_series(bf, mid::Integer, sel::Symbol; mtk_max=130.0)
    s = subset(bf, :match_id => ByRow(==(mid)), :selection => ByRow(==(sel)),
                   :minutes_to_kickoff => ByRow(x -> 0.0 < x <= mtk_max))
    sort!(s, :minutes_to_kickoff)
    return select(s, :minutes_to_kickoff => :mtk, :traded_price)
end

"""
    anchor_goals(bf, ds, mid; prior_off1, prior_htgap, win) -> Vector{NamedTuple}

For each goal (in chronological order) find the betfair wall-clock time of its
price jump. A home goal makes the HOME 1X2 price drop (away price rise); an away
goal does the reverse. We search the relevant series for the largest *correctly
signed* log-return inside a window around the goal's expected wall-clock position
(match-min + prior offset). Returns (mm, is_home, g_w) with g_w = wall-clock mtk
of the jump (falls back to the prior-expected position if no jump is found).
"""
function anchor_goals(bf, ds, mid::Integer; prior_off1=3.0, prior_htgap=15.0, win=8.0)
    goals = match_goals(ds, mid)
    home = _price_series(bf, mid, :home)
    away = _price_series(bf, mid, :away)
    anchors = NamedTuple[]
    for r in eachrow(goals)
        prior_off = r.mm <= 45 ? prior_off1 : (prior_off1 + prior_htgap)
        expected_w = r.mm + prior_off
        lo = r.mm - 1.0; hi = expected_w + win   # feed jump can't precede the goal minute
        # choose the series whose price DROPS when this team scores
        ser = r.is_home ? home : away
        g_w = expected_w
        if nrow(ser) >= 2
            lr = diff(log.(ser.traded_price))
            tw = ser.mtk[2:end]
            # candidate jumps: negative log-return (price drop) within window
            inwin = findall(i -> lo <= tw[i] <= hi && lr[i] < -0.05, eachindex(lr))
            if !isempty(inwin)
                best = inwin[argmin(lr[inwin])]   # most negative drop
                g_w = tw[best]
            end
        end
        push!(anchors, (mm=r.mm, is_home=r.is_home, g_w=g_w))
    end
    sort!(anchors, by = a -> a.g_w)
    return anchors
end

"Current score (home, away) at wall-clock time t_w given goal anchors."
function score_at(anchors, t_w::Real)
    gh = ga = 0
    for a in anchors
        if a.g_w <= t_w
            a.is_home ? (gh += 1) : (ga += 1)
        end
    end
    return (gh, ga)
end

"""
    make_clock_map(anchors; htgap, prior_off1) -> function t_w -> t_m

Two-segment wall-clock -> match-minute map with the half-time break baked in
STRUCTURALLY (so it is correct even when all goals are in one half). Within a
half the feed clock runs ~parallel to the match clock (slope 1) with a constant
offset; the half-time break adds `htgap`.

  - off1 = median feed lag of first-half goal anchors (else `prior_off1`)
  - off2 = median feed lag of second-half goal anchors (else off1 + htgap),
           floored at off1 + 0.5*htgap so the HT gap never vanishes
  - H1:   t_w <= 45+off1            -> t_m = t_w - off1
  - HT:   45+off1 < t_w < 45+off2   -> t_m = 45 (frozen during the break)
  - H2:   t_w >= 45+off2            -> t_m = t_w - off2
"""
function make_clock_map(anchors; htgap=15.0, prior_off1=3.0)
    h1 = [Float64(a.g_w - a.mm) for a in anchors if a.mm <= 45]
    h2 = [Float64(a.g_w - a.mm) for a in anchors if a.mm > 45]
    off1 = isempty(h1) ? prior_off1 : median(h1)
    off2 = isempty(h2) ? off1 + htgap : median(h2)
    off2 = max(off2, off1 + 0.5 * htgap)
    function tm(t_w::Real)
        x = Float64(t_w)
        x <= 45 + off1 && return x - off1
        x >= 45 + off2 && return x - off2
        return 45.0   # within the half-time break
    end
    return tm
end

# ---------------------------------------------------------------------------
# 2. Per-bin fair probabilities (LOCF within a staleness window + vig strip)
# ---------------------------------------------------------------------------

"Latest traded price per selection within (t_w - staleness, t_w]. Dict selection => price."
function latest_prices(bf_match::AbstractDataFrame, t_w::Real; staleness=4.0)
    out = Dict{Symbol,Float64}()
    sub = filter(r -> (t_w - staleness) < r.minutes_to_kickoff <= t_w, bf_match)
    isempty(sub) && return out
    for gdf in groupby(sub, :selection)
        r = last(sort(gdf, :minutes_to_kickoff))   # most recent in window
        out[r.selection] = r.traded_price
    end
    return out
end

"""
    fair_match_df(prices) -> DataFrame(:match_id?, :selection, :prob_fair_close)

Strip vig per market group: prob_fair = (1/price) / overround, with overround
summed over the selections of that market that are present. Correct-score is
normalised over ALL cs_* selections (incl. any_other), but only explicit cs_ij
are emitted as fit targets (any_other has no clean model probability).
"""
function fair_match_df(prices::Dict{Symbol,Float64})
    groups = market_groups()
    # build cs group dynamically from whatever cs_* selections are present
    groups[:cs] = [s for s in keys(prices) if is_cs(s)]
    sels = Symbol[]; probs = Float64[]
    for (_, members) in groups
        present = [s for s in members if haskey(prices, s)]
        isempty(present) && continue
        overround = sum(1.0 / prices[s] for s in present)
        overround <= 0 && continue
        for s in present
            pf = (1.0 / prices[s]) / overround
            # drop cs_any_other_* from targets (kept only in the overround)
            (is_cs(s) && !_is_explicit_cs(s)) && continue
            push!(sels, s); push!(probs, pf)
        end
    end
    return DataFrame(selection = sels, prob_fair_close = probs)
end

_is_explicit_cs(sel::Symbol) = (s = String(sel); startswith(s, "cs_") &&
                                length(s) == 5 && all(isdigit, s[4:5]))

# ---------------------------------------------------------------------------
# 2b. Score-conditioned model probabilities + inversion
# ---------------------------------------------------------------------------

"""
    model_prob(sel, P, gh, ga) -> Float64 or missing

Model-implied probability of a market selection given the REMAINING-goals
scoreline matrix `P[i+1,j+1]` (i,j = remaining home/away goals) and the current
score (gh, ga). Final score = (gh+i, ga+j). Returns `missing` for selections
without a clean model prob (e.g. cs_any_other_*), which the fit skips.
"""
function model_prob(sel::Symbol, P::AbstractMatrix, gh::Int, ga::Int)
    mg = size(P, 1) - 1
    s = String(sel)
    if sel === :home
        return sum(P[i+1, j+1] for i in 0:mg, j in 0:mg if (gh + i) > (ga + j); init=0.0)
    elseif sel === :draw
        return sum(P[i+1, j+1] for i in 0:mg, j in 0:mg if (gh + i) == (ga + j); init=0.0)
    elseif sel === :away
        return sum(P[i+1, j+1] for i in 0:mg, j in 0:mg if (gh + i) < (ga + j); init=0.0)
    elseif sel === :btts_yes
        return sum(P[i+1, j+1] for i in 0:mg, j in 0:mg if (gh + i) > 0 && (ga + j) > 0; init=0.0)
    elseif sel === :btts_no
        return sum(P[i+1, j+1] for i in 0:mg, j in 0:mg if !((gh + i) > 0 && (ga + j) > 0); init=0.0)
    elseif startswith(s, "over_") || startswith(s, "under_")
        digits = s[(findfirst('_', s)+1):end]              # e.g. "25"
        k = parse(Int, digits[1:end-1])                    # integer threshold (total >= k+1 == over)
        need = (k + 1) - (gh + ga)                          # remaining goals needed for OVER
        prob_over = sum(P[i+1, j+1] for i in 0:mg, j in 0:mg if (i + j) >= need; init=0.0)
        return startswith(s, "over_") ? prob_over : (1.0 - prob_over)
    elseif _is_explicit_cs(sel)
        xh = parse(Int, s[4:4]); xa = parse(Int, s[5:5])
        i = xh - gh; j = xa - ga
        (i < 0 || j < 0 || i > mg || j > mg) && return 0.0
        return P[i+1, j+1]
    end
    return missing
end

"""
    fit_inplay(match_df, config; current_score=(0,0), max_goals=8)

Invert fair probabilities for (λ_home, λ_away[, ρ]) using the existing scoreline
engine, conditioned on `current_score`. `current_score=(0,0)` reproduces the
naive full-match baseline. Returns (; λ_home, λ_away, ρ, residual, n_used).
"""
function fit_inplay(match_df, config; current_score=(0, 0), max_goals=8)
    gh, ga = current_score
    targets = [(row.selection, row.prob_fair_close) for row in eachrow(match_df)]
    function loss(θ)
        P = Features.build_probability_matrix(config, θ, max_goals)
        sse = 0.0; n = 0
        for (sel, target) in targets
            mp = model_prob(sel, P, gh, ga)
            ismissing(mp) && continue
            sse += (mp - target)^2; n += 1
        end
        return n == 0 ? Inf : sse
    end
    res = optimize(loss, Features.get_initial_guess(config), NelderMead())
    θ = Optim.minimizer(res)
    pars = Features.extract_parameters(config, θ)
    # residual rmse over used selections
    P = Features.build_probability_matrix(config, θ, max_goals)
    n = 0; sse = 0.0
    for (sel, target) in targets
        mp = model_prob(sel, P, gh, ga); ismissing(mp) && continue
        sse += (mp - target)^2; n += 1
    end
    rmse = n == 0 ? NaN : sqrt(sse / n)
    return (; λ_home = pars.λ_home, λ_away = pars.λ_away,
              ρ = get(pars, :ρ, 0.0), residual = rmse, n_used = n)
end

# ---------------------------------------------------------------------------
# 3. Per-match driver: sweep wall-clock bins, fit conditioned + naive
# ---------------------------------------------------------------------------

"""
    inplay_lambda_trace(bf, ds, mid, config; ...) -> DataFrame

Sweep wall-clock bins for one match; at each bin map to match-minute, read the
current score from goal anchors, strip vig, and fit both the score-conditioned
(remaining-λ) and naive inversions. Also derive the detrended per-90 rate
μ = λ_rem * 90/(90 - t_m).
"""
function inplay_lambda_trace(bf, ds, mid::Integer, config = Features.DoublePoissonMarketFeature();
                             bin_minutes = 3.0, staleness = 10.0, min_sel = 6,
                             require_1x2 = true, mtk_max = 130.0, max_goals = 8)
    bf_match = subset(bf, :match_id => ByRow(==(mid)),
                          :minutes_to_kickoff => ByRow(x -> 0.0 < x <= mtk_max))
    nrow(bf_match) == 0 && return DataFrame()
    anchors = anchor_goals(bf, ds, mid)
    tmap = make_clock_map(anchors)
    reds = match_reds(ds, mid)

    rows = NamedTuple[]
    for t_w in (bin_minutes):bin_minutes:mtk_max
        prices = latest_prices(bf_match, t_w; staleness = staleness)
        length(prices) < min_sel && continue
        mdf = fair_match_df(prices)
        nrow(mdf) < min_sel && continue
        # need a full 1X2 to identify the home/away split, not just the total
        require_1x2 && !all(s -> s in mdf.selection, (:home, :draw, :away)) && continue
        gh, ga = score_at(anchors, t_w)
        t_m = tmap(t_w)
        # red-card state by match-minute
        hr = count(r -> r.is_home && r.mm <= t_m, eachrow(reds))
        ar = count(r -> !r.is_home && r.mm <= t_m, eachrow(reds))

        cond  = fit_inplay(mdf, config; current_score = (gh, ga), max_goals = max_goals)
        naive = fit_inplay(mdf, config; current_score = (0, 0),  max_goals = max_goals)

        # Per-90 detrend μ = λ_rem * 90/(90 - t_m). Beyond ~t_m 80 the 1/(90-t_m)
        # factor explodes and amplifies noise, so μ is NaN there (use λ_rem for the
        # tail). rem_frac floored for safety.
        rem_frac = clamp((90.0 - t_m) / 90.0, 1e-2, 1.0)
        μ_h = t_m >= 80 ? NaN : cond.λ_home / rem_frac
        μ_a = t_m >= 80 ? NaN : cond.λ_away / rem_frac

        push!(rows, (match_id = mid, t_w = t_w, t_m = round(t_m, digits = 1),
                     λ_rem_h = cond.λ_home, λ_rem_a = cond.λ_away,
                     μ_h = μ_h, μ_a = μ_a,
                     λ_naive_h = naive.λ_home, λ_naive_a = naive.λ_away,
                     goal_diff = gh - ga, gh = gh, ga = ga,
                     home_reds = hr, away_reds = ar,
                     n_sel = cond.n_used, residual = cond.residual))
    end
    return DataFrame(rows)
end
