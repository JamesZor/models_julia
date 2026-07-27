# current_development/match_day_inference/src/unified_staking.jl
#
# Match-day UNIFIED structural-Kelly staking panel.
#
# Layers the (P) joint log-optimal portfolio + (U-MC) posterior-uncertainty shrinkage k*
# (docs/bets_multi/unified_kelly_postgrad_notes.md) on top of the LIVE Betfair book, as a
# per-match portfolio panel shown next to the existing per-bet Bayesian–McHale dashboard.
#
# Scope (validated by unified_staking/NOTES.md r02 — curated set WINS, full-book Kelly bankrupts):
#   - STAKED families: Over/Under + BTTS only. 1X2/DC/AH/CS are display-only elsewhere.
#   - Fed by the Smile-DP model latents (λ_h, λ_a posterior-draw vectors per match).
#   - cap 0.10 (Σa ≤ 10% bankroll) + 2% Betfair commission baked into payoffs.
#
# The structural primitives (solve_P, state_draws, mask_for, G_growth, proj_cap!, GG/HGRID/AGRID)
# are reused VERBATIM from the research loader — single source of truth. We do NOT use its
# run_match/settle (those read a summarize_betfair_market close book; here the book is live Redis).
#
# CAVEATS (v1): back-only (no exchange lays); no complementary near-arb screen (commission in
# d_eff mitigates most of it); the smile prices O/U off-grid via λ_tot·φ(K) but φ≈1 pregame
# (see memory no-pregame-intensity-smile) so the double-Poisson state grid ≈ the smile's own O/U.

using DataFrames
using Statistics
using Printf
using PrettyTables
using Dates
using Random: Xoshiro
import BayesianFootball.Experiments: LatentStates

# Reuse the structural-Kelly primitives (defines solve_P, state_draws, mask_for, G_growth,
# proj_cap!, GG, HGRID, AGRID). Path is resolved relative to THIS file so it works regardless
# of the runner's cwd.
include(joinpath(@__DIR__, "..", "..", "unified_staking", "l01_structural_kelly.jl"))

const UNIFIED_STAKED_FAMILIES = Set(["OverUnder", "BTTS"])

# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE BOOK ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_live_book(redis_conn, market_id_lookup, home, away, mid; families=UNIFIED_STAKED_FAMILIES)

Assembles the live back-odds book for one match, restricted to the staked `families`
(Over/Under + BTTS). Returns a Vector of NamedTuples
`(market_name::String, market_line::Float64, selection::Symbol, back::Float64)`, one row per
selection that currently has a finite back price > 1.0. Returns an empty vector if fewer than
2 valid rows are found (a portfolio needs ≥ 2 selections to be interesting).
"""
function build_live_book(redis_conn, market_id_lookup, home::String, away::String, mid::Int;
                         families=UNIFIED_STAKED_FAMILIES)
    rows = NamedTuple{(:market_name, :market_line, :selection, :back),
                      Tuple{String, Float64, Symbol, Float64}}[]

    for mkt in get_available_markets_for_match(market_id_lookup, home, away)
        mkt.ppd_market in families || continue
        live_odds = fetch_live_odds_for_market(redis_conn, market_id_lookup, home, away, mkt)
        for sel in _expected_selections(mkt.ppd_market, mkt.ppd_line)
            info = get(live_odds, sel, (back=NaN, lay=NaN, back_size=0.0, lay_size=0.0))
            back = info.back
            (isnan(back) || back <= 1.0) && continue
            push!(rows, (market_name=mkt.ppd_market, market_line=mkt.ppd_line,
                         selection=sel, back=Float64(back)))
        end
    end

    return length(rows) >= 2 ? rows : eltype(rows)[]
end

# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED STRUCTURAL KELLY ON A LIVE BOOK
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_match_live(lat_df, mid, live_book; cap=0.10, commission=0.02, S_dec=200,
                   kgrid=0.01:0.01:1.0, seed=11)

Match-day analogue of `run_match`: builds the payoff matrix from LIVE back odds (commission
baked in via `d_eff = 1 + (back-1)(1-commission)`), solves (P) at the posterior mean p̄,
re-solves per posterior draw, and picks the shrinkage k* by (U-MC).

Returns a NamedTuple:
  - `book`   :: DataFrame of rows with `a_star > 0`
                (market, sel, back, p_model, ev, a_star, exec = k*·a_star)
  - `kstar`, `total = Σa*`, `cash = 1-Σa*`, `G` (expected log-growth at a*).
`lat_df` must carry per-match `λ_h`/`λ_a` posterior-draw vectors (LatentStates.df).
"""
function run_match_live(lat_df::DataFrame, mid::Int, live_book::AbstractVector;
                        cap=0.10, commission=0.02, S_dec=200, kgrid=0.01:0.01:1.0, seed=11)
    masks = [mask_for(r.market_name, r.market_line, r.selection) for r in live_book]
    back  = [r.back for r in live_book]
    d_eff = 1.0 .+ (back .- 1.0) .* (1.0 - commission)
    R = hcat([d_eff[m] .* masks[m] .- 1.0 for m in eachindex(masks)]...)

    P = state_draws(lat_df, mid)
    pbar = vec(mean(P, dims=2))
    astar = solve_P(pbar, R; cap=cap)

    idx = rand(Xoshiro(seed), 1:size(P, 2), S_dec)
    A = Matrix{Float64}(undef, length(astar), S_dec)
    for (j, s) in enumerate(idx)
        A[:, j] = solve_P(view(P, :, s), R; cap=cap, a0=astar, iters=800)
    end
    Ψ(k) = mean(G_growth(k .* view(A, :, j), pbar, R) for j in 1:S_dec)
    ks = collect(kgrid)
    kstar = ks[argmax(Ψ.(ks))]

    pm = [sum(pbar[m]) for m in masks]
    out = DataFrame(
        market = [string(r.market_name) * "_" * @sprintf("%.1f", r.market_line) for r in live_book],
        sel = [r.selection for r in live_book],
        back = round.(back, digits=3),
        p_model = round.(pm, digits=4),
        ev = round.(pm .* d_eff .- 1.0, digits=4),   # EV on commission-adjusted odds
        a_star = round.(astar, digits=4),
        exec = round.(kstar .* astar, digits=4),
    )
    return (book = out[out.a_star .> 0, :], kstar = kstar, total = sum(astar),
            cash = 1 - sum(astar), G = G_growth(astar, pbar, R))
end

# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD PANEL
# ═══════════════════════════════════════════════════════════════════════════════

"""
    print_unified_staking_dashboard(latents, redis_conn, todays_matches, market_id_lookup;
                                    label="Smile-DP", cap=0.10, commission=0.02, bankroll=0.0)

Prints a per-match UNIFIED structural-Kelly portfolio panel (Over/Under + BTTS only) beside the
existing per-bet dashboard. Each panel shows the joint allocation a*, the shrinkage k*, the
executed stakes k*·a*, and (if `bankroll > 0`) the £ stake per selection. `latents` is the
Smile-DP `LatentStates` (its `.df` supplies the λ posterior draws).
"""
function print_unified_staking_dashboard(latents::LatentStates, redis_conn,
                                         todays_matches::AbstractDataFrame, market_id_lookup;
                                         label::String="Smile-DP", cap=0.10, commission=0.02,
                                         bankroll::Float64=0.0)
    lat_df = latents.df
    table_format = PrettyTables.TextTableFormat(borders = PrettyTables.text_table_borders__unicode_rounded)

    width = 118
    println("\n" * "="^width)
    println(" UNIFIED STRUCTURAL KELLY ($label) | Over/Under + BTTS | cap=$cap  commission=$(commission*100)% " *
            (bankroll > 0 ? "| bankroll=£$(bankroll) " : "") * "| $(Dates.format(now(), "HH:MM:SS"))")
    println(" (P) joint log-optimal portfolio + (U-MC) posterior shrinkage k*  —  paper stakes, back-only")
    println("="^width)

    for row in eachrow(todays_matches)
        mid = Int(row.match_id)
        home = String(row.home_team)
        away = String(row.away_team)

        # Skip matches whose latents aren't present (e.g. not in this model's inference set).
        if findfirst(==(mid), lat_df.match_id) === nothing
            continue
        end

        live_book = build_live_book(redis_conn, market_id_lookup, home, away, mid)
        if isempty(live_book)
            println("\n> $home vs $away (ID: $mid)")
            println("   [!] No live Over/Under or BTTS book (need ≥2 priced selections).")
            continue
        end

        res = run_match_live(lat_df, mid, live_book; cap=cap, commission=commission)

        println("\n> $home vs $away (ID: $mid)  " *
                "k*=$(round(res.kstar, digits=3))  Σa*=$(round(res.total, digits=3))  " *
                "cash=$(round(res.cash, digits=3))  G=$(round(res.G, digits=4))")

        if isempty(res.book)
            println("   (no +growth allocation — portfolio holds 100% cash)")
            continue
        end

        n = nrow(res.book)
        ncols = bankroll > 0 ? 7 : 6
        data = Matrix{Any}(undef, n, ncols)
        for (i, r) in enumerate(eachrow(res.book))
            data[i, 1] = r.market
            data[i, 2] = string(r.sel)
            data[i, 3] = @sprintf("%.1f%%", r.p_model * 100)
            data[i, 4] = @sprintf("%.2f", r.back)
            data[i, 5] = r.ev > 0 ? @sprintf("+%.1f%%", r.ev * 100) : @sprintf("%.1f%%", r.ev * 100)
            data[i, 6] = @sprintf("%.2f%%", r.exec * 100)
            if bankroll > 0
                data[i, 7] = @sprintf("£%.2f", bankroll * r.exec)
            end
        end

        labels = bankroll > 0 ?
            ["Market", "Selection", "Model %", "Back", "EV", "Exec (k*·a*)", "£ Stake"] :
            ["Market", "Selection", "Model %", "Back", "EV", "Exec (k*·a*)"]
        aligns = bankroll > 0 ? [:l, :l, :r, :r, :r, :r, :r] : [:l, :l, :r, :r, :r, :r]

        pretty_table(data; column_labels=labels, table_format=table_format, alignment=aligns)
    end

    println("\n" * "="^width)
    println(" Exec = k*·a* = fraction of bankroll to stake on Betfair (back). " *
            "Σexec ≤ cap. EV net of $(commission*100)% commission.")
    println("="^width * "\n")
end

# Convenience overload: build the market-id lookup from Redis.
function print_unified_staking_dashboard(latents::LatentStates, redis_conn,
                                         todays_matches::AbstractDataFrame;
                                         label::String="Smile-DP", cap=0.10, commission=0.02,
                                         bankroll::Float64=0.0)
    market_id_lookup = get_live_market_mappings(redis_conn)
    return print_unified_staking_dashboard(latents, redis_conn, todays_matches, market_id_lookup;
                                           label=label, cap=cap, commission=commission, bankroll=bankroll)
end
