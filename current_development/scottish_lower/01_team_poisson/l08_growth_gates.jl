# ==============================================================================
# Model 01 — GATE 7 : GROWTH
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Gate 6 established that the probabilities are sound. Gate 7 asks the only
# question that pays: staked into a real book at real prices with real commission,
# does this model make money?
#
# ASSEMBLES src/Portfolio. It writes no staking mathematics. That module was
# audited and graduated, and re-implementing Kelly here would repeat exactly the
# mistake this protocol exists to prevent.
#
#   BookSpec   -> build_books -> group(DailySlate) -> simulate -> path_metrics
#
# Three properties of that module drive everything and are worth stating up front,
# because each is counter-intuitive:
#
#   1. `risk_factor` is HOMOGENEOUS OF DEGREE 0 in the stakes handed to it. Once the
#      drawdown constraint binds, trust and shrinkage can only RESHAPE the book, not
#      resize it. To move exposure you move λ, not trust.
#
#   2. `DeArb` settles at `d * min(overround, 1)`. With real vig that is the quoted
#      price; where the recorded book implies an arbitrage it is shaved. So the
#      backtest cannot harvest data artifacts, and ROI is not inflated by them.
#
#   3. `require_complete_markets` is on by default, so partial markets — the ones
#      that de-vig to p = 1.0 and wrecked gate 6 until they were caught — are
#      dropped inside `extract_selections` rather than needing handling here.
#
# Settlement uses the real score via `settle_vector`, NOT the `is_winner` column, so
# the grading defect in T004 cannot reach these numbers.
#
# ==============================================================================
# WHAT IS DECLARED IN ADVANCE
#
# Prior work on this book is unambiguous that CURATION dominates the model: on the
# full book every Kelly variant lost, and curating to totals + BTTS inverted the
# result. That makes the eligible-line set the single most overfittable knob here.
#
# It is therefore fixed in `tp_growth_policies` BEFORE any result is read, and the
# full book is always reported alongside, so a curated number can never be quoted
# without the thing it was selected from.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Printf

const Pf = BayesianFootball.Portfolio


# ==============================================================================
# 1. The odds table
# ==============================================================================

"""
    tp_betfair_odds_df(ds, contract; ids, window) -> DataFrame

Betfair closing prices in the `ds.odds` schema that `Portfolio.extract_selections`
expects: `match_id`, `market_name`, `market_line`, `selection`, `odds_close`.

Close window only — see `tp_betfair_book` in l07 and ticket T005 for why
`summarize_betfair_market`'s open window is not used.

`extract_selections` recomputes the overround from the quotes it finds, so no
de-vigged probability is passed in. That is deliberate on the package's part: the
overround must be consistent with the exact set of legs actually usable.
"""
function tp_betfair_odds_df(ds, contract::SLContract;
                            ids::Union{Nothing,AbstractSet} = nothing,
                            window = (-20.0, 0.0))
    D   = BayesianFootball.Data
    raw = D.summarize_odds(ds.betfair_odds, D.TWAEstimator(); window = window)
    isempty(raw) && return DataFrame()

    wanted = Set((D.market_group(m), Float64(D.market_line(m))) for m in tp_book_markets(contract))
    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, raw)
    ids === nothing || (df = filter(r -> Int(r.match_id) in ids, df))
    "is_sane" in names(df) && (df = filter(r -> coalesce(r.is_sane, true), df))

    return DataFrame(
        match_id    = Int.(df.match_id),
        market_name = String.(df.market_name),
        market_line = Float64.(df.market_line),
        selection   = Symbol.(df.selection),
        odds_close  = Float64.(df.odds),
    )
end

"""
    tp_bookmaker_odds_df(ds, contract; ids) -> DataFrame

The same, from `ds.odds`. Bet365 close is the wider book (all six markets on ~360
fixtures against Betfair's thin exchange), so it exists as a coverage comparison —
NOT as an execution venue. Bets are executed on the exchange.
"""
function tp_bookmaker_odds_df(ds, contract::SLContract; ids::Union{Nothing,AbstractSet} = nothing)
    D = BayesianFootball.Data
    wanted = Set((D.market_group(m), Float64(D.market_line(m))) for m in tp_book_markets(contract))
    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, ds.odds)
    ids === nothing || (df = filter(r -> Int(r.match_id) in ids, df))
    return DataFrame(
        match_id    = Int.(df.match_id),
        market_name = String.(df.market_name),
        market_line = Float64.(df.market_line),
        selection   = Symbol.(df.selection),
        odds_close  = Float64.(df.odds_close),
    )
end


# ==============================================================================
# 2. Specs, declared before any result is seen
# ==============================================================================

"""
    tp_book_spec(contract) -> BookSpec

The book side: which markets, how prices settle, how stakes are allocated.

`ExecutionConfig` carries the contract's commission. `KellyLogUtility` allocates on
the posterior-mean grid and `BakerMcHale` supplies the parameter-uncertainty
shrinkage — the point of a Bayesian model is that the posterior width should size
the bet, and that is where it enters.
"""
function tp_book_spec(contract::SLContract;
                      price     = Pf.DeArb(),
                      allocator = Pf.KellyLogUtility(),
                      shrink    = Pf.BakerMcHale())
    return Pf.BookSpec(
        markets   = BayesianFootball.Data.Markets.MarketConfig(tp_book_markets(contract)),
        price     = price,
        allocator = allocator,
        shrink    = shrink,
        exec      = Pf.ExecutionConfig(
            commission               = Pf.PerBetCommission(contract.commission),
            max_selection_stake      = contract.max_selection_stake,
            require_complete_markets = true,
        ),
    )
end

"""
    tp_growth_policies(contract; lambda) -> Vector{(; name, policy)}

The staking policies compared, **fixed before any result is read**.

Curation is the most overfittable choice available, so the eligible line sets are
declared here rather than chosen once the numbers are visible, and the full book is
always carried so a curated result cannot be quoted without its selection pool.

`FixedCap(portfolio_kelly_cap)` bounds simultaneous exposure. Independent per-bet
full Kelly went bankrupt in prior work; the cap is the dominant lever.

`SlateDrawdown(λ)` is the exposure control that actually binds. λ is a parameter of
this gate, not a fitted quantity — `calibrate_lambda` may be used to hit a target
exposure, but it must be calibrated on a rule declared in advance, never on the
result.
"""
function tp_growth_policies(contract::SLContract;
                            lambda::Float64 = contract.drawdown_lambda,
                            trust::Float64  = contract.trust_w)
    cap  = Pf.FixedCap(contract.portfolio_kelly_cap)
    risk = Pf.SlateDrawdown(lambda)
    base(f) = Pf.PolicySpec(trust = Pf.FlatTrust(trust), risk = risk, cap = cap,
                            filter = f, grouping = Pf.DailySlate())

    return [
        (name = "full book",     policy = base(Pf.KeepAll())),
        (name = "totals only",   policy = base(tp_keep_groups(contract, ["OverUnder"]))),
        (name = "totals + BTTS", policy = base(tp_keep_groups(contract, ["OverUnder", "BTTS"]))),
        (name = "1X2 only",      policy = base(tp_keep_groups(contract, ["1X2"]))),
    ]
end

"""
    tp_keep_groups(contract, groups) -> MarketWhitelist

Curate to whole market groups, expressed as the `(group, line, selection)` keys
`Portfolio.MarketWhitelist` expects.

Built from the CONTRACT book rather than from whatever selections happen to appear,
so a curated policy always means the same thing regardless of which lines the
exchange priced on a given day.
"""
function tp_keep_groups(contract::SLContract, groups::Vector{String})
    D = BayesianFootball.Data
    keys = Set{Tuple{String,Float64,Symbol}}()
    for m in tp_book_markets(contract)
        g = D.market_group(m)
        g in groups || continue
        for (_, sel) in pairs(D.outcomes(m))
            push!(keys, (g, Float64(D.market_line(m)), sel))
        end
    end
    return Pf.MarketWhitelist(keys)
end


# ==============================================================================
# 3. GATE 7a — Book construction
# ==============================================================================

"""
    tp_gate_books(books, latents_df, odds_df; label) -> Vector

Did the book build, and what did it drop?

`build_book` returns `nothing` for an unknown fixture, unusable quotes, or a
score-matrix failure — silently, and correctly, because a match you cannot price is
not a match you can bet. But a book that quietly built 30 of 360 fixtures is
indistinguishable in the output from one that built all of them, which is precisely
the failure T005 caused one stage earlier.
"""
function tp_gate_books(books, latents_df::AbstractDataFrame, odds_df::AbstractDataFrame;
                       label::String = "", min_coverage::Float64 = 0.50)
    out = Any[]
    n_want = nrow(latents_df)
    n_have = length(books)
    priced = length(unique(odds_df.match_id))

    cov = n_want == 0 ? 0.0 : n_have / n_want
    push!(out, (
        name   = "books built",
        pass   = cov >= min_coverage,
        detail = @sprintf("%d of %d fixtures (%.1f%%); %d had usable quotes — threshold %.0f%%",
                          n_have, n_want, 100cov, priced, 100min_coverage),
    ))
    n_have == 0 && return out

    push!(out, (
        name   = "Kelly solved",
        pass   = all(b.converged for b in books),
        detail = "$(count(b -> b.converged, books)) of $n_have converged",
    ))

    # KKT is checked only on books that actually ALLOCATE. Where the optimum is
    # "bet nothing" the solution sits on the boundary, every stake is zero, and the
    # residual is neither meaningful nor consequential — no stake is placed from it.
    #
    # Measured: 319 of 320 books sit at ~1.2e-6 as documented, and the single
    # exception (2.7e-4) is exactly such a no-bet book. Taking the max over all books
    # would fail the gate on the one fixture where the answer cannot be wrong.
    betting = [b for b in books if maximum(b.a_kelly) > 0]
    worst_kkt = isempty(betting) ? 0.0 : maximum(b.kkt for b in betting)
    push!(out, (
        name   = "KKT residual (books that allocate)",
        pass   = worst_kkt <= 1e-4,
        detail = @sprintf("max %.3e over %d allocating books (target ~1e-6); %d books stake nothing",
                          worst_kkt, length(betting), n_have - length(betting)),
    ))

    push!(out, (
        name   = "grids normalised",
        pass   = all(abs(sum(b.p_grid) - 1) <= 1e-9 for b in books),
        detail = @sprintf("max |Σp - 1| = %.3e",
                          maximum(abs(sum(b.p_grid) - 1) for b in books)),
    ))

    push!(out, (
        name   = "all books settled",
        pass   = all(Pf.is_settled(b) for b in books),
        detail = "$(count(Pf.is_settled, books)) of $n_have have a result",
    ))

    push!(out, (
        name   = "chronological",
        pass   = issorted(books, by = b -> b.date),
        detail = "$(minimum(b.date for b in books)) → $(maximum(b.date for b in books))",
    ))

    n_sel = sum(length(b.sels) for b in books)
    fams  = unique(s.group for b in books for s in b.sels)
    push!(out, (
        name   = "selections available",
        pass   = n_sel > 0,
        detail = @sprintf("%d selections, %.1f per fixture, groups: %s",
                          n_sel, n_sel / n_have, join(sort(fams), ", ")),
    ))

    return out
end


# ==============================================================================
# 4. GATE 7b — Simulation integrity
# ==============================================================================

"""
    tp_gate_simulation(traj, slates, contract) -> Vector

Did the simulation obey its own constraints?

Prior work shipped three defects that all produce plausible-looking numbers:
a bankroll driven negative by an absent slate cap, path metrics computed on a
scrambled bet order, and an alpha fitted on the data it was evaluated on. The first
two are structural and are asserted here; the third is a design rule, not a check.
"""
function tp_gate_simulation(traj, slates, contract::SLContract)
    out = Any[]

    push!(out, (
        name   = "bankroll never non-positive",
        pass   = all(>(0), traj.bankroll),
        detail = @sprintf("min bankroll %.4f over %d slates", minimum(traj.bankroll), length(traj.slate_pl)),
    ))

    push!(out, (
        name   = "slates chronological",
        pass   = issorted(slates, by = s -> s.window),
        detail = "$(length(slates)) slates, $(slates[1].window) → $(slates[end].window)",
    ))

    mx = isempty(traj.exposure) ? 0.0 : maximum(traj.exposure)
    push!(out, (
        name   = "exposure within cap",
        pass   = mx <= contract.portfolio_kelly_cap + 1e-9,
        detail = @sprintf("max slate exposure %.4f, cap %.2f, %d slates capped",
                          mx, contract.portfolio_kelly_cap, traj.n_capped),
    ))

    push!(out, (
        name   = "no slate loses the bankroll",
        pass   = isempty(traj.slate_pl) || minimum(traj.slate_pl) > -1.0,
        detail = @sprintf("worst slate %+.4f", isempty(traj.slate_pl) ? 0.0 : minimum(traj.slate_pl)),
    ))

    push!(out, (
        name   = "bets recorded",
        pass   = nrow(traj.bets) > 0,
        detail = @sprintf("%d bets, %.2f staked in total", nrow(traj.bets), traj.total_stake),
    ))

    return out
end


# ==============================================================================
# 5. Results
# ==============================================================================

"""
    tp_pnl_concentration(traj) -> (; top1, top5, top10, top20, win_rate)

What share of total profit came from the best handful of bets.

**This is the most important number in gate 7.** A backtest whose entire profit comes
from ten bets has not demonstrated an edge; it has demonstrated that longshots
sometimes land. Measured on this model's full book: the top 10 bets accounted for
108.3% of total P&L, meaning the other 898 were net negative in aggregate.

It is the check that reconciles gate 6 with gate 7. Gate 6 has far more statistical
power to detect an information advantage than a 320-match bankroll path does, so when
gate 6 finds nothing and gate 7 shows a large ROI, concentration is where the
disagreement usually resolves — and it did here.

Reported in the growth table itself rather than as a separate diagnostic, so an ROI
cannot be quoted without it.
"""
function tp_pnl_concentration(traj)
    pnl = sort(traj.bets.pnl; rev = true)
    tot = sum(pnl)
    share(k) = (tot == 0 || length(pnl) < k) ? NaN : 100 * sum(pnl[1:k]) / tot
    return (top1 = share(1), top5 = share(5), top10 = share(10), top20 = share(20),
            win_rate = mean(traj.bets.pnl .> 0))
end

"""
    tp_growth_table(books, contract; lambda) -> DataFrame

Every declared policy on the same books, one row each.

Same books for all rows: policies are pure post-multipliers on a built book, so a
difference between rows can only be the policy. Rebuilding per policy would let a
book difference masquerade as a staking result.

`roi_lo` / `roi_hi` are a percentile interval on flat ROI, bootstrapped **by match**
— several bets on one fixture share a scoreline and are strongly dependent, and
resampling individual bets would understate the interval badly.
"""
function tp_growth_table(books, contract::SLContract;
                         lambda::Float64 = contract.drawdown_lambda,
                         trust::Float64  = contract.trust_w,
                         B::Int = 4000)
    rows = NamedTuple[]
    for (name, policy) in tp_growth_policies(contract; lambda = lambda, trust = trust)
        slates = Pf.group(policy.grouping, books)
        traj   = Pf.simulate(policy, slates)
        m      = Pf.path_metrics(traj)
        ci     = nrow(traj.bets) > 0 ? Pf.bootstrap_roi(traj.bets; B = B) : (lo = NaN, hi = NaN, sd = NaN)

        conc = tp_pnl_concentration(traj)
        push!(rows, (
            policy      = name,
            n_bets      = m.n_bets,
            final       = round(m.final, digits = 3),
            roi_pct     = round(m.roi, digits = 2),
            roi_lo      = round(ci[1], digits = 2),
            roi_hi      = round(ci[2], digits = 2),
            # Carried in the table itself so an ROI can never be read without it.
            top10_pct   = round(conc.top10, digits = 1),
            win_rate    = round(conc.win_rate, digits = 3),
            growth      = round(m.growth_per_slate, digits = 5),
            mdd_pct     = round(m.mdd, digits = 1),
            mean_expo   = round(m.mean_exposure, digits = 4),
        ))
    end
    return DataFrame(rows)
end

"""
    tp_gate_growth(growth) -> Vector

The verdict, stated the way the evidence supports it.

**Passing is not "made money".** 360 fixtures of one season is far too little to
establish an edge: prior work on this repository put the honest interval at
[-1.5%, +20.9%], which contains zero comfortably. What gate 7 must establish is that
the machinery is sound and the result is REPORTED HONESTLY — including when the
interval contains zero, which is the expected outcome at this sample size.

A significant positive ROI is recorded as a finding. So is a significant negative
one, which would be the more informative result.
"""
function tp_gate_growth(growth::AbstractDataFrame)
    out = Any[]
    full = growth[findfirst(==("full book"), growth.policy), :]

    push!(out, (
        name   = "every policy produced bets",
        pass   = all(growth.n_bets .> 0),
        detail = join(["$(r.policy): $(r.n_bets)" for r in eachrow(growth)], ", "),
    ))

    push!(out, (
        name   = "bankroll survived everywhere",
        pass   = all(growth.final .> 0),
        detail = @sprintf("final wealth %.3f to %.3f", minimum(growth.final), maximum(growth.final)),
    ))

    sig = filter(r -> isfinite(r.roi_lo) && (r.roi_lo > 0 || r.roi_hi < 0), growth)
    push!(out, (
        name   = "ROI significance (finding, not a criterion)",
        pass   = true,
        detail = isempty(sig) ? "no policy's ROI interval excludes zero — expected at n = $(full.n_bets)" :
                 join(["$(r.policy) $(r.roi_pct)% [$(r.roi_lo), $(r.roi_hi)]" for r in eachrow(sig)], "; "),
    ))

    worst_conc = maximum(filter(isfinite, growth.top10_pct))
    push!(out, (
        name   = "P&L concentration (read before the ROI)",
        pass   = true,
        detail = @sprintf("top 10 bets are %.0f%%-%.0f%% of total P&L across policies; %s",
                          minimum(filter(isfinite, growth.top10_pct)), worst_conc,
                          worst_conc >= 100 ?
                            "AT OR ABOVE 100% — the profit is a handful of bets, not an edge" :
                            "distributed enough to be worth interpreting"),
    ))

    push!(out, (
        name   = "full book reported alongside curation",
        pass   = "full book" in growth.policy,
        detail = @sprintf("full book: %.2f%% ROI, final %.3f, %d bets",
                          full.roi_pct, full.final, full.n_bets),
    ))

    return out
end


# ==============================================================================
# 6. Policy sweeps  (cheap — the books are cached)
# ==============================================================================
#
# A `PolicySpec` is a pure post-multiplier on an already-built `MatchBook`, so a
# policy sweep costs nothing but the simulation. A `BookSpec` change does not: it
# invalidates every book and forces the ~35s rebuild. That asymmetry is why the two
# functions below are separate.

"""
    tp_sweep_policy(books, contract; trusts, lambdas, policy_name) -> DataFrame

Sweep trust and λ on ONE curation, reusing the cached books.

Expect trust to do almost nothing. `risk_factor` is homogeneous of degree 0 in the
stakes handed to it, so once the drawdown constraint binds it solves for whatever
factor satisfies the constraint and `trust x stakes` comes out unchanged — trust can
reshape the book across selections but cannot resize it. λ is the lever that moves
exposure.

Swept rather than asserted: the claim is in the module's docstring, and a sweep that
confirms it on this data costs seconds.
"""
function tp_sweep_policy(books, contract::SLContract;
                         trusts  = [0.15, 0.3, 0.5, 1.0],
                         lambdas = [15.0, 23.0, 35.0],
                         policy_name::String = "full book",
                         B::Int = 1500)
    rows = NamedTuple[]
    for λ in lambdas, w in trusts
        ps = tp_growth_policies(contract; lambda = λ, trust = w)
        i  = findfirst(p -> p.name == policy_name, ps)
        i === nothing && error("unknown policy $policy_name")
        pol    = ps[i].policy
        slates = Pf.group(pol.grouping, books)
        traj   = Pf.simulate(pol, slates)
        m      = Pf.path_metrics(traj)
        conc   = tp_pnl_concentration(traj)
        ci     = nrow(traj.bets) > 0 ? Pf.bootstrap_roi(traj.bets; B = B) : (NaN, NaN, NaN)
        push!(rows, (
            λ = λ, trust = w, n_bets = m.n_bets,
            final = round(m.final, digits = 3),
            roi_pct = round(m.roi, digits = 2),
            roi_lo = round(ci[1], digits = 2), roi_hi = round(ci[2], digits = 2),
            top10_pct = round(conc.top10, digits = 1),
            mean_expo = round(m.mean_exposure, digits = 4),
            mdd_pct = round(m.mdd, digits = 1),
        ))
    end
    return DataFrame(rows)
end

"""
    tp_sweep_book(ds, contract, latents_df, expr, odds_df; variants) -> DataFrame

Sweep `BookSpec` components. **Rebuilds the books for every variant** (~35s each), so
keep the list short.

`price` is the one worth varying and the one to be careful with. `Normalise` settles
ABOVE the traded price wherever there is real vig, which manufactures edge — the
module's own docstring says not to believe anything it produces. It is included here
only as an ablation, to show how much of a result is the price policy rather than the
model.
"""
function tp_sweep_book(ds, contract::SLContract, latents_df, expr, odds_df;
                       variants = [("DeArb (default)", Pf.DeArb()),
                                   ("RawPrice",        Pf.RawPrice()),
                                   ("Normalise (ABLATION — manufactures edge)", Pf.Normalise())],
                       policy_name::String = "full book")
    rows = NamedTuple[]
    for (name, price) in variants
        spec  = tp_book_spec(contract; price = price)
        books = Pf.build_books(spec, latents_df, expr, odds_df, ds)
        ps    = tp_growth_policies(contract)
        pol   = ps[findfirst(p -> p.name == policy_name, ps)].policy
        traj  = Pf.simulate(pol, Pf.group(pol.grouping, books))
        m     = Pf.path_metrics(traj)
        conc  = tp_pnl_concentration(traj)
        push!(rows, (
            price = name, n_books = length(books), n_bets = m.n_bets,
            final = round(m.final, digits = 3),
            roi_pct = round(m.roi, digits = 2),
            top10_pct = round(conc.top10, digits = 1),
            mean_expo = round(m.mean_exposure, digits = 4),
        ))
    end
    return DataFrame(rows)
end
