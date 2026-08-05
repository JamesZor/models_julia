# l06_portfolio_v2.jl
#
# Corrected portfolio / staking loader. Self-contained: does NOT include l02-l05,
# which are kept as the record of the exploration.
#
# What changed vs l02-l05, and why (see r16 for the assertions that police each one):
#
#  1. Win masks and settlement now come from ONE source of truth (Data.grade_selection),
#     evaluated on hypothetical scorelines for the mask and on the real scoreline for
#     settlement. Kills the string-matching mask builder and makes pushes representable.
#  2. Pushes pay 0 (stake returned), not -1. l02 graded a `missing` outcome as a LOSS.
#  3. Payoffs are built from the traded price, de-arbed ONE-SIDED (a book that prices
#     at overround < 1 -- 45% of O/U groups, a closing-window artifact -- is shrunk
#     back to 1.0; a book at overround >= 1 is left alone). l02 renormalised in both
#     directions, so it settled ABOVE the traded price whenever there was real vig.
#     Vig-removed probabilities are kept only as the market's forecast benchmark.
#  4. Market groups with a missing selection are rejected outright (l02 renormalised
#     over whatever was present, which inflates the surviving legs by up to 20%).
#  5. The Sum(a) <= budget constraint is a log-barrier that the gradient actually sees,
#     so obj and grad are consistent; the solver returns a KKT residual for auditing.
#  6. The market-netting pass is gone. With commission in the payoff, no payoff-
#     preserving reduction of a multi-sided position exists (sum 1/(1+c_i) > 1
#     strictly), so l02's `a1 - a2` netting was always payoff-changing. Once the book
#     is de-arbed, multi-sided positions are strictly dominated and never chosen --
#     an invariant in r16 asserts exactly that instead.
#  7. Baker & McHale (2013) outer shrinkage is implemented: the Kelly portfolio is
#     re-solved on each posterior draw and a single k* is chosen. l02 collapsed the
#     posterior to its mean and threw all parameter uncertainty away.
#  8. Slates are chronological and settle simultaneously, under a hard exposure cap,
#     so the simulated bankroll can never go negative.
#  9. alpha is fitted WALK-FORWARD on decayed log-loss of the blended probability
#     (the recipe in kelly_multi.tex S"Historical Alpha Optimization"), never by
#     maximising a backtest path metric.

using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, Random
using Optim

const BF   = BayesianFootball
const D    = BF.Data
const E    = BF.Experiments
const PRED = BF.Predictions

# ===================================================================
# 1. Configuration
# ===================================================================

Base.@kwdef struct PortfolioConfig
    commission::Float64            = 0.02   # Betfair exchange commission on net win
    max_stake_per_selection::Float64 = 0.50 # box bound on a single a_j
    budget::Float64                = 0.99   # Sum(a) < budget, per match
    min_stake::Float64             = 1e-4   # stakes below this are dropped to 0
    barrier_mu::Float64            = 1e-6   # budget log-barrier weight
    require_complete_markets::Bool = true   # reject partially-quoted market groups
    # How the traded price is turned into the price we settle at:
    #   :dearb     -> d * min(overround, 1)  shrink an impossible book, never inflate (v2)
    #   :normalise -> d * overround          l02 behaviour: also INFLATES a real book
    #   :raw       -> d                      leaves the window arbitrage in place
    price_mode::Symbol             = :dearb
end

"""
Busseti, Ryu & Boyd (2016) drawdown constraint plus a hard slate exposure cap.

`lambda = log(beta)/log(D)`; D is the tolerated bankroll floor (0.8 = 20% drawdown)
and beta the probability of ever breaching it. `slate_cap` bounds the total stake
settled simultaneously, which is what structurally prevents a negative bankroll.
`mode` selects the slate aggregation: `:sequential` reproduces the l05 sum-of-logs
(a ~2-3% conservative approximation) and `:joint` Monte-Carlos the true simultaneous
distribution.

`scope` selects WHERE the drawdown budget is spent:
  :slate -> one k for the whole day, solved against all L matches jointly (v2 default)
  :match -> l03 behaviour: each match gets its own k from its own return distribution,
            which bounds the drawdown of each bet in isolation and therefore does NOT
            bound the drawdown of the ~6 bets that all settle at 3pm.
"""
Base.@kwdef struct RiskConfig
    lambda::Float64      = 20.0
    slate_cap::Float64   = 0.25
    mode::Symbol         = :sequential
    joint_draws::Int     = 50_000
    scope::Symbol        = :slate
end

risk_lambda(Dfloor::Float64, beta::Float64) = log(beta) / log(Dfloor)

"""Baker & McHale (2013) parameter-uncertainty shrinkage."""
Base.@kwdef struct ShrinkConfig
    enabled::Bool     = true
    n_draws::Int      = 128
    k_grid::Vector{Float64} = collect(0.0:0.02:1.0)
    seed::Int         = 20260805
end

# ===================================================================
# 2. Data structures
# ===================================================================

struct Selection
    family::String        # "1X2_home", "O/U 2.5_over_25", ... (l04-compatible key)
    group::String         # "1X2" | "BTTS" | "OverUnder"
    line::Float64
    selection::Symbol
    odds_quoted::Float64  # price as traded in the closing window
    odds_used::Float64    # de-arbed price we settle at (<= odds_quoted, never above)
    p_model::Float64      # posterior-mean model probability
    p_market::Float64     # vig-removed market probability (benchmark only)
end

"""
Everything needed to stake and settle one match. `R` is the Jacot return matrix
(N_states x N_selections) built from raw prices net of commission; `settle` is the
realised per-unit payoff of each selection (win / push / lose) using the same rule.
"""
struct MatchBook
    m_id::Int
    date::Date
    sels::Vector{Selection}
    p_grid::Vector{Float64}      # posterior-mean score grid, length N
    R::Matrix{Float64}           # N x n
    settle::Vector{Float64}      # n
    a_kelly::Vector{Float64}     # full-Kelly stakes on the posterior mean
    k_bm::Float64                # Baker-McHale shrinkage factor (1.0 if disabled)
    kkt::Float64                 # KKT residual of a_kelly (should be ~0)
end

alpha_key(s::Selection) = s.family

# ===================================================================
# 3. Market extraction  (complete groups only, raw prices preserved)
# ===================================================================

_family(group::String, line::Float64, sel::Symbol) =
    group == "OverUnder" ? "O/U $(line)_$(sel)" : "$(group)_$(sel)"

"""
    extract_markets(odds_df, match_id, markets_config, cfg) -> Vector{Selection}

Pulls the closing price for every selection of every configured market. A market
group is kept only if EVERY one of its outcomes is quoted (`require_complete_markets`),
because the vig-removal step below divides by the sum over whatever is present --
with a leg missing that silently manufactures edge on the survivors.
"""
function extract_markets(odds_df::DataFrame, match_id::Int, markets_config,
                         model_probs::Dict, cfg::PortfolioConfig)
    row_odds = view(odds_df, odds_df.match_id .== match_id, :)
    out = Selection[]

    for m in markets_config.markets
        m_str   = string(m)
        group   = D.market_group(m)
        line    = D.market_line(m)
        outs    = D.outcomes(m)
        n_want  = length(outs)

        m_df = view(row_odds, (row_odds.market_name .== group) .&
                              isapprox.(row_odds.market_line, line; atol = 1e-3), :)
        isempty(m_df) && continue

        quoted = Dict{Symbol, Float64}()
        for r in eachrow(m_df)
            (ismissing(r.odds_close) || r.odds_close <= 1.0) && continue
            quoted[r.selection] = r.odds_close
        end

        # completeness guard
        if cfg.require_complete_markets && length(quoted) != n_want
            continue
        end
        isempty(quoted) && continue

        # vig removal over the (now complete) group -> market probability benchmark
        overround = sum(1.0 / o for o in values(quoted))
        haskey(model_probs, m_str) || continue

        # De-arb, ONE-SIDED. ~45% of O/U groups price at overround < 1 because the
        # closing "price" is a time-weighted average of trades that happened at
        # different moments (median 1 tick). Left alone, the Kelly solver treats that
        # as a risk-free arbitrage and levers the whole bankroll into it. Shrinking
        # such a book back to overround 1.0 removes the artifact; books with a real
        # overround >= 1 are left untouched, so we never settle above the traded price.
        # With commission > 0 this also makes any multi-sided position strictly
        # dominated, which is why v2 needs no netting pass (see r16 invariants).
        dearb = cfg.price_mode === :dearb     ? min(overround, 1.0) :
                cfg.price_mode === :normalise ? overround           :
                cfg.price_mode === :raw       ? 1.0                 :
                error("unknown price_mode $(cfg.price_mode)")

        for (sel, o) in quoted
            haskey(model_probs[m_str], sel) || continue
            push!(out, Selection(_family(group, line, sel), group, line, sel,
                                 o, o * dearb,
                                 mean(model_probs[m_str][sel]),
                                 (1.0 / o) / overround))
        end
    end
    return out
end

# ===================================================================
# 4. Payoff matrix  (single source of truth = Data.grade_selection)
# ===================================================================

"""
Per-unit payoff of `sel` if the match ends `h`-`a`, net of commission.
win -> (1-c)(d-1),  push/void -> 0.0 (stake returned),  loss -> -1.0
"""
function payoff(sel::Selection, h::Int, a::Int, commission::Float64)
    g = D.grade_selection(sel.group, sel.line, sel.selection, h, a)
    ismissing(g) && return 0.0
    return g ? (1.0 - commission) * (sel.odds_used - 1.0) : -1.0
end

"""
    build_returns(sels, max_h, max_a, cfg) -> R (N x n)

R[omega, j] is the Jacot r_{omega,j}: wealth after the bet is `1 + R*a`.
Rows are `vec`-ordered over the (home, away) score grid, matching `vec(P_grid)`.
"""
function build_returns(sels::Vector{Selection}, max_h::Int, max_a::Int, cfg::PortfolioConfig)
    n = length(sels)
    R = zeros(Float64, max_h * max_a, n)
    @inbounds for j in 1:n, c in 1:max_a, r in 1:max_h
        R[(c - 1) * max_h + r, j] = payoff(sels[j], r - 1, c - 1, cfg.commission)
    end
    return R
end

settle_vector(sels::Vector{Selection}, h::Int, a::Int, cfg::PortfolioConfig) =
    [payoff(s, h, a, cfg.commission) for s in sels]

# ===================================================================
# 5. Kelly solver  (Jacot eq. 12-14, consistent objective/gradient)
# ===================================================================

"""
    solve_kelly(p, R, cfg) -> (a, kkt_residual)

maximise  G(a) = sum_omega p_omega log(1 + R_omega' a)
s.t.      0 <= a_j <= max_stake,  sum(a) <= budget

The budget is enforced with a log-barrier of weight `barrier_mu` that appears in BOTH
the objective and the gradient (l02 put a hard `Inf` in the objective only, so the
optimiser had no interior signal steering it away from the cliff). `kkt_residual` is
the worst violation of the first-order conditions and should be ~1e-8.
"""
function solve_kelly(p::Vector{Float64}, R::Matrix{Float64}, cfg::PortfolioConfig)
    n = size(R, 2)
    n == 0 && return (Float64[], 0.0)
    B, mu = cfg.budget, cfg.barrier_mu

    function obj(a)
        s = sum(a)
        s >= B && return Inf
        w = 1.0 .+ R * a
        any(w .<= 1e-10) && return Inf
        return -dot(p, log.(w)) - mu * log(B - s)
    end
    function grad!(g, a)
        s = sum(a)
        w = 1.0 .+ R * a
        if s >= B || any(w .<= 1e-10)
            fill!(g, 1e6); return g
        end
        g .= -(R' * (p ./ w)) .+ mu / (B - s)
        return g
    end

    res = optimize(obj, grad!, zeros(n), fill(cfg.max_stake_per_selection, n),
                   fill(1e-3, n), Fminbox(LBFGS()))
    a = copy(Optim.minimizer(res))
    a[a .< cfg.min_stake] .= 0.0

    # KKT audit on the *unbarriered* problem.
    #   min -G(a)  s.t. a >= 0, a <= ub, sum(a) <= B
    #   stationarity: gr_j + nu - mu_lo,j + mu_hi,j = 0, nu >= 0, nu*(sum(a)-B) = 0
    # so every interior coordinate must share the same multiplier nu = -gr_j; the
    # residual is the spread of gr over interior coordinates plus the sign conditions
    # at the bounds. (l02's cliff had no nu at all, which is what this catches.)
    w  = 1.0 .+ R * a
    gr = -(R' * (p ./ w))                      # gradient of -G
    ub = cfg.max_stake_per_selection
    interior = [j for j in 1:n if a[j] > 0.0 && a[j] < ub - 1e-9]
    nu = (sum(a) < B - 1e-6 || isempty(interior)) ? 0.0 : max(0.0, -mean(gr[interior]))

    kkt = 0.0
    for j in 1:n
        if a[j] <= 0.0
            kkt = max(kkt, max(0.0, -(gr[j] + nu)))    # need gr_j + nu >= 0
        elseif a[j] >= ub - 1e-9
            kkt = max(kkt, max(0.0, gr[j] + nu))       # need gr_j + nu <= 0
        else
            kkt = max(kkt, abs(gr[j] + nu))            # interior: gr_j + nu = 0
        end
    end
    return (a, kkt)
end

# ===================================================================
# 6. Baker & McHale outer shrinkage
# ===================================================================

"""
    baker_mchale_k(score_matrix, R, p_true, cfg, shrink) -> k*

Implements kelly_multi.tex eq. (10): re-solve the Kelly portfolio on each posterior
draw q^(j), then pick the single k maximising

    U(k) = 1/m sum_j sum_omega p_true,omega log(1 + k R_omega(a*(q^(j))))

Returns 1.0 when shrinkage is disabled. U(k) is strictly concave in k (the 1-D Hessian
is a sum of negative terms), so the grid argmax is the global optimum.
"""
function baker_mchale_k(score_matrix, R::Matrix{Float64}, p_true::Vector{Float64},
                        cfg::PortfolioConfig, shrink::ShrinkConfig; seed_offset::Int = 0)
    shrink.enabled || return 1.0
    size(R, 2) == 0 && return 1.0

    ns   = size(score_matrix.data, 3)
    rng  = MersenneTwister(shrink.seed + seed_offset)
    draw = randperm(rng, ns)[1:min(shrink.n_draws, ns)]

    port = Vector{Vector{Float64}}(undef, length(draw))
    for (i, j) in enumerate(draw)
        q = vec(score_matrix.data[:, :, j])       # copy: concrete Vector{Float64}
        q ./= sum(q)
        a, _ = solve_kelly(q, R, cfg)
        port[i] = R * a
    end

    best_k, best_u = 0.0, -Inf
    for k in shrink.k_grid
        u = 0.0
        for r in port
            u += dot(p_true, log.(max.(1.0 .+ k .* r, 1e-12)))
        end
        if u > best_u
            best_u, best_k = u, k
        end
    end
    return best_k
end

# ===================================================================
# 7. Per-match book construction
# ===================================================================

"""
    build_book(latents_row, expr, odds_df, markets_config, scores, cfg, shrink) -> MatchBook | nothing

Fast-fails before touching the (expensive) score matrix if the match has no usable
odds. Returns `nothing` for any match we cannot stake or settle.
"""
function build_book(latents_row, expr, odds_df::DataFrame, markets_config,
                    scores::Dict{Int,Tuple{Int,Int,Date}},
                    cfg::PortfolioConfig, shrink::ShrinkConfig)
    m_id = latents_row.match_id
    haskey(scores, m_id) || return nothing
    h, a, dt = scores[m_id]

    # cheap pre-filter: any quotes at all for this match?
    any(odds_df.match_id .== m_id) || return nothing

    param = PRED.extract_params(expr.config.model, latents_row)
    score_matrix = try
        PRED.compute_score_matrix(expr.config.model, param)
    catch
        return nothing
    end

    model_probs = Dict(string(m) => PRED.compute_market_probs(score_matrix, m)
                       for m in markets_config.markets)

    sels = extract_markets(odds_df, m_id, markets_config, model_probs, cfg)
    isempty(sels) && return nothing

    max_h, max_a, _ = size(score_matrix.data)
    p_grid = vec(mean(score_matrix.data, dims = 3)[:, :, 1])
    p_grid ./= sum(p_grid)                                   # kill grid truncation

    R      = build_returns(sels, max_h, max_a, cfg)
    a_k, kkt = solve_kelly(p_grid, R, cfg)
    k_bm   = baker_mchale_k(score_matrix, R, p_grid, cfg, shrink; seed_offset = m_id)

    return MatchBook(m_id, dt, sels, p_grid, R,
                     settle_vector(sels, h, a, cfg), a_k, k_bm, kkt)
end

"""Multithreaded book construction over a latents DataFrame."""
function build_books(latents_df::DataFrame, expr, odds_df::DataFrame, markets_config,
                     ds; cfg::PortfolioConfig = PortfolioConfig(),
                     shrink::ShrinkConfig = ShrinkConfig())
    scores = Dict{Int,Tuple{Int,Int,Date}}()
    for r in eachrow(ds.matches)
        (ismissing(r.home_score) || ismissing(r.away_score)) && continue
        scores[r.match_id] = (Int(r.home_score), Int(r.away_score), Date(r.match_date))
    end

    n   = nrow(latents_df)
    buf = Vector{Union{Nothing,MatchBook}}(undef, n)
    Threads.@threads for i in 1:n
        buf[i] = build_book(latents_df[i, :], expr, odds_df, markets_config, scores, cfg, shrink)
    end

    books = MatchBook[b for b in buf if b !== nothing]
    sort!(books, by = b -> (b.date, b.m_id))          # chronological, once and for all
    return books
end

# ===================================================================
# 8. Slates
# ===================================================================

struct Slate
    date::Date
    books::Vector{MatchBook}
end

function build_slates(books::Vector{MatchBook})
    @assert issorted(books, by = b -> b.date) "books must be chronological"
    slates = Slate[]
    for b in books
        if !isempty(slates) && slates[end].date == b.date
            push!(slates[end].books, b)
        else
            push!(slates, Slate(b.date, [b]))
        end
    end
    return slates
end

# ===================================================================
# 9. Risk manager
# ===================================================================

"""
Busseti constraint for a single simultaneous slate, solved by bisection.

`:sequential` is the l05 form  sum_t log E[(1 + k R_t)^-lambda] <= 0, which assumes the
matches compound one after another. `:joint` Monte-Carlos the true simultaneous
return sum_t R_t and solves E[(1 + k sum_t R_t)^-lambda] <= 1. Measured difference on
real slates: ~2-3% in k, sequential being the looser of the two.
"""
function slate_shrinkage(probs::Vector{Vector{Float64}}, rets::Vector{Vector{Float64}},
                         rc::RiskConfig; rng = MersenneTwister(1))
    rc.lambda <= 0 && return 1.0
    isempty(probs) && return 1.0
    lam = rc.lambda

    f = if rc.mode === :joint
        cums  = [cumsum(p ./ sum(p)) for p in probs]
        draws = zeros(rc.joint_draws)
        for m in 1:rc.joint_draws
            s = 0.0
            for t in eachindex(rets)
                idx = searchsortedfirst(cums[t], rand(rng))
                s += rets[t][min(idx, length(rets[t]))]
            end
            draws[m] = s
        end
        k -> mean((1.0 .+ k .* draws) .^ (-lam)) - 1.0
    else
        function (k)
            tot = 0.0
            for t in eachindex(probs)
                tot += log(sum(probs[t][i] * (1.0 + k * rets[t][i])^(-lam)
                               for i in eachindex(probs[t])))
            end
            return tot
        end
    end

    f(1.0) <= 0.0 && return 1.0
    lo, hi = 0.0, 1.0
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        f(mid) > 0.0 ? (hi = mid) : (lo = mid)
    end
    return lo
end

# ===================================================================
# 10. Staking policy  (alpha -> Kelly -> Baker-McHale -> Busseti -> cap)
# ===================================================================

"""
    stake_slate(slate, alphas, cfg, rc; use_bm) -> (stakes_per_book, k_risk, exposure)

`alphas` maps an alpha_key to a per-selection trust weight. NOTE: because the market
probabilities are vig-removed (p_market * d_fair == 1), blending
`alpha*p_model + (1-alpha)*p_market` scales the marginal Kelly edge by exactly alpha,
so applying alpha to the solved stake is exact at the selection level and first-order
for the joint portfolio. This is the same arithmetic l04 used -- what changes in v2 is
where `alphas` comes from (walk-forward log-loss, not backtest Martin).

The hard `slate_cap` is applied last and is what guarantees `day_pl > -1`.
"""
function stake_slate(slate::Slate, alphas::Dict{String,Float64},
                     cfg::PortfolioConfig, rc::RiskConfig;
                     use_bm::Bool = true, global_scale::Float64 = 1.0)
    stakes = Vector{Vector{Float64}}(undef, length(slate.books))
    for (i, b) in enumerate(slate.books)
        a = copy(b.a_kelly)
        for j in eachindex(a)
            a[j] *= get(alphas, alpha_key(b.sels[j]), 0.0)
        end
        use_bm && (a .*= b.k_bm)
        a .*= global_scale
        stakes[i] = a
    end

    probs = [b.p_grid for b in slate.books]
    rets  = [slate.books[i].R * stakes[i] for i in eachindex(stakes)]

    # risk budget: one k for the day, or one k per match
    k, ks = if rc.scope === :match
        per = [slate_shrinkage([probs[i]], [rets[i]], rc) for i in eachindex(stakes)]
        for i in eachindex(stakes); stakes[i] .*= per[i]; end
        (1.0, per)
    else
        kk = slate_shrinkage(probs, rets, rc)
        for s in stakes; s .*= kk; end
        (kk, [kk])
    end

    # hard exposure cap, applied last, in BOTH scopes
    exposure = sum(sum(s) for s in stakes)
    capped   = false
    if exposure > rc.slate_cap && exposure > 0
        sc = rc.slate_cap / exposure
        for s in stakes; s .*= sc; end
        exposure = rc.slate_cap
        capped   = true
    end

    return (stakes = stakes, k_risk = mean(ks), exposure = exposure, capped = capped)
end

# ===================================================================
# 11. Simulation + metrics
# ===================================================================

"""
    simulate(slates, alphas, cfg, rc; use_bm) -> NamedTuple

Chronological, simultaneous same-day settlement, compounding once per slate.
Asserts the bankroll can never be driven non-positive.
"""
function simulate(slates::Vector{Slate}, alphas::Dict{String,Float64},
                  cfg::PortfolioConfig, rc::RiskConfig;
                  use_bm::Bool = true, global_scale::Float64 = 1.0)
    @assert issorted(slates, by = s -> s.date) "slates must be chronological"
    @assert rc.slate_cap < 1.0 "slate_cap must be < 1 or the bankroll can go negative"

    bank    = 1.0
    hist    = Float64[1.0]
    dates   = Date[]
    day_pl  = Float64[]
    stake_t = 0.0
    pl_t    = 0.0
    ks      = Float64[]
    expo    = Float64[]
    ncap    = 0
    bet_rows = NamedTuple[]

    for sl in slates
        st = stake_slate(sl, alphas, cfg, rc; use_bm = use_bm, global_scale = global_scale)
        push!(ks, st.k_risk); push!(expo, st.exposure); st.capped && (ncap += 1)

        dpl, dst = 0.0, 0.0
        for (i, b) in enumerate(sl.books), j in eachindex(b.sels)
            s = st.stakes[i][j]
            s > 0 || continue
            dst += s
            dpl += s * b.settle[j]
            push!(bet_rows, (match_id = b.m_id, date = b.date, key = alpha_key(b.sels[j]),
                             stake = s, odds = b.sels[j].odds_used,
                             pl = s * b.settle[j], payoff = b.settle[j],
                             p_model = b.sels[j].p_model, p_market = b.sels[j].p_market))
        end

        @assert dpl > -1.0 "slate $(sl.date) lost more than the bankroll (dpl=$dpl)"
        bank *= (1.0 + dpl)
        push!(hist, bank); push!(dates, sl.date); push!(day_pl, dpl)
        stake_t += dst; pl_t += dpl
    end

    return (bankroll = hist, dates = dates, day_pl = day_pl, k_risk = ks,
            exposure = expo, n_capped = ncap,
            total_stake = stake_t, total_pl = pl_t, bets = DataFrame(bet_rows))
end

"""
    calibrate_scale(slates, alphas, cfg, rc; target_exposure) -> scale

Binary-searches the global stake multiplier so that MEAN realised slate exposure equals
`target_exposure`.

WARNING: only usable while lambda is SLACK. An active Busseti constraint solves k against
the stakes it is handed, so doubling the input halves k and the realised exposure barely
moves -- exposure saturates at the lambda-implied level and the search runs away to absurd
scales until `slate_cap` starts binding. That scale-invariance is exactly why lambda
subsumes alpha. To move exposure ABOVE the lambda level, move lambda: see
`calibrate_lambda`.
"""
function calibrate_scale(slates::Vector{Slate}, alphas::Dict{String,Float64},
                         cfg::PortfolioConfig, rc::RiskConfig;
                         target_exposure::Float64 = 0.15, use_bm::Bool = true,
                         iters::Int = 40)
    f(sc) = mean(simulate(slates, alphas, cfg, rc; use_bm = use_bm, global_scale = sc).exposure)
    lo, hi = 1e-4, 1.0
    while f(hi) < target_exposure && hi < 1e4; hi *= 2; end
    for _ in 1:iters
        mid = sqrt(lo * hi)
        f(mid) < target_exposure ? (lo = mid) : (hi = mid)
    end
    return sqrt(lo * hi)
end

"""
    calibrate_lambda(slates, alphas, cfg, rc; target_exposure) -> lambda

Bisects lambda so MEAN realised slate exposure equals `target_exposure`. lambda is the
dial that actually moves exposure once the drawdown constraint is active (the stake
multiplier does not -- see `calibrate_scale`). Exposure is monotone decreasing in lambda.
"""
function calibrate_lambda(slates::Vector{Slate}, alphas::Dict{String,Float64},
                          cfg::PortfolioConfig, rc::RiskConfig;
                          target_exposure::Float64 = 0.15, use_bm::Bool = true,
                          iters::Int = 30)
    f(lam) = mean(simulate(slates, alphas, cfg,
                           RiskConfig(lambda = lam, slate_cap = rc.slate_cap,
                                      mode = rc.mode, joint_draws = rc.joint_draws,
                                      scope = rc.scope);
                           use_bm = use_bm).exposure)
    lo, hi = 0.5, 200.0                       # low lambda = high exposure
    f(lo) < target_exposure && return lo      # cannot reach it, lambda is already slack
    f(hi) > target_exposure && return hi
    for _ in 1:iters
        mid = sqrt(lo * hi)
        f(mid) > target_exposure ? (lo = mid) : (hi = mid)
    end
    return sqrt(lo * hi)
end

"""Path metrics. Only valid on a chronologically ordered series -- callers must not shuffle."""
function path_metrics(sim)
    bk = sim.bankroll
    rm = accumulate(max, bk)
    dd = (bk .- rm) ./ rm .* 100
    ui = max(sqrt(mean(dd .^ 2)), 1e-9)
    tr = (bk[end] - 1.0) * 100
    n  = max(length(sim.day_pl), 1)
    return (final = bk[end],
            roi = sim.total_stake > 0 ? 100 * sim.total_pl / sim.total_stake : 0.0,
            growth_per_slate = mean(log.(1.0 .+ sim.day_pl)),
            mdd = minimum(dd), ulcer = ui,
            calmar = minimum(dd) < 0 ? tr / abs(minimum(dd)) : 0.0,
            martin = tr / ui, n_slates = n,
            avg_k = mean(sim.k_risk), worst_slate_pl = minimum(sim.day_pl))
end

"""Flat-ROI bootstrap CI, resampled by MATCH (bets inside a match are correlated)."""
function bootstrap_roi(bets::DataFrame; B::Int = 4000, seed::Int = 1)
    isempty(bets) && return (lo = 0.0, hi = 0.0)
    rng  = MersenneTwister(seed)
    mids = unique(bets.match_id)
    idxs = Dict(m => findall(==(m), bets.match_id) for m in mids)
    v = Vector{Float64}(undef, B)
    for b in 1:B
        sel = Int[]
        for _ in eachindex(mids)
            append!(sel, idxs[mids[rand(rng, 1:length(mids))]])
        end
        v[b] = 100 * sum(bets.pl[sel]) / sum(bets.stake[sel])
    end
    return (lo = quantile(v, 0.025), hi = quantile(v, 0.975), sd = std(v))
end

# ===================================================================
# 12. Walk-forward alpha  (kelly_multi.tex, "Historical Alpha Optimization")
# ===================================================================

"""
    fit_alpha_logloss(obs; half_life) -> alpha in [0,1]

`obs` is a vector of (p_model, p_market, y, age_days). Minimises the exponentially
decayed log-loss of the blend `alpha*p_model + (1-alpha)*p_market`. Convex in alpha
(log-loss is convex in p, p is affine in alpha), so Brent is exact.
"""
function fit_alpha_logloss(obs::Vector{<:NamedTuple}; half_life::Float64 = 120.0)
    isempty(obs) && return 0.0
    w = [0.5 ^ (o.age_days / half_life) for o in obs]
    function loss(a)
        t = 0.0
        for (i, o) in enumerate(obs)
            p = clamp(a * o.p_model + (1 - a) * o.p_market, 1e-6, 1 - 1e-6)
            t -= w[i] * (o.y ? log(p) : log1p(-p))
        end
        return t / sum(w)
    end
    res = optimize(loss, 0.0, 1.0, Brent())
    return clamp(Optim.minimizer(res), 0.0, 1.0)
end

"""
    walkforward_alphas(slates; lookback_days, half_life, min_obs, default) -> Vector{Dict}

For each slate, returns the alpha map fitted using ONLY matches that had already been
settled before that slate's date. Families with too little history fall back to
`default`. This is the leakage-free replacement for r08.
"""
function walkforward_alphas(slates::Vector{Slate}; lookback_days::Int = 365,
                            half_life::Float64 = 120.0, min_obs::Int = 40,
                            default::Float64 = 0.25)
    hist = Dict{String,Vector{NamedTuple{(:p_model, :p_market, :y, :date),
                                          Tuple{Float64,Float64,Bool,Date}}}}()
    out = Vector{Dict{String,Float64}}(undef, length(slates))

    for (i, sl) in enumerate(slates)
        amap = Dict{String,Float64}()
        for (k, v) in hist
            obs = [(p_model = o.p_model, p_market = o.p_market, y = o.y,
                    age_days = Float64(Dates.value(sl.date - o.date)))
                   for o in v if 0 <= Dates.value(sl.date - o.date) <= lookback_days]
            amap[k] = length(obs) >= min_obs ? fit_alpha_logloss(obs; half_life = half_life) : default
        end
        out[i] = amap

        for b in sl.books, j in eachindex(b.sels)
            s = b.sels[j]
            push!(get!(hist, alpha_key(s),
                       NamedTuple{(:p_model, :p_market, :y, :date),
                                  Tuple{Float64,Float64,Bool,Date}}[]),
                  (p_model = s.p_model, p_market = s.p_market,
                   y = b.settle[j] > 0.0, date = b.date))
        end
    end
    return out
end

"""Simulate with a per-slate alpha map (the walk-forward case)."""
function simulate_walkforward(slates::Vector{Slate}, alpha_maps::Vector{Dict{String,Float64}},
                              cfg::PortfolioConfig, rc::RiskConfig;
                              use_bm::Bool = true, default::Float64 = 0.25)
    @assert length(alpha_maps) == length(slates)
    bank = 1.0; hist = Float64[1.0]; day_pl = Float64[]; ks = Float64[]
    stake_t = 0.0; pl_t = 0.0; bet_rows = NamedTuple[]; dates = Date[]

    for (i, sl) in enumerate(slates)
        amap = Dict{String,Float64}(k => v for (k, v) in alpha_maps[i])
        for b in sl.books, s in b.sels
            haskey(amap, alpha_key(s)) || (amap[alpha_key(s)] = default)
        end
        st = stake_slate(sl, amap, cfg, rc; use_bm = use_bm)
        push!(ks, st.k_risk)

        dpl, dst = 0.0, 0.0
        for (bi, b) in enumerate(sl.books), j in eachindex(b.sels)
            s = st.stakes[bi][j]
            s > 0 || continue
            dst += s; dpl += s * b.settle[j]
            push!(bet_rows, (match_id = b.m_id, date = b.date, key = alpha_key(b.sels[j]),
                             stake = s, odds = b.sels[j].odds_used,
                             pl = s * b.settle[j], payoff = b.settle[j],
                             p_model = b.sels[j].p_model, p_market = b.sels[j].p_market))
        end
        @assert dpl > -1.0 "slate $(sl.date) lost more than the bankroll"
        bank *= (1.0 + dpl)
        push!(hist, bank); push!(day_pl, dpl); push!(dates, sl.date)
        stake_t += dst; pl_t += dpl
    end

    return (bankroll = hist, dates = dates, day_pl = day_pl, k_risk = ks,
            total_stake = stake_t, total_pl = pl_t, bets = DataFrame(bet_rows))
end
