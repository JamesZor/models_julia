#=
r04 — NIGHT 2: market-anchor STRENGTH on the r03 winning family.

Two priors disagree, which is exactly why this is worth a night:
  • Ireland (split_market_pillar): an INTERIOR optimum near mw≈0.25-0.40; raising it backfires
    (totals compression is the market pillar denoising, and over-anchoring destroys the signal).
  • 56/57 (scottish_lower_smile): the mw axis is FLAT on scoring across a 4× range (Δ=0.0001), and
    the only separating criterion was SAMPLER HEALTH — only mw40 cleared the hard gate, and the
    R-hat offenders were ALWAYS the team ratings (dyn.raw_a/raw_d), never the market σ. Heavier
    anchoring stiffens the rating geometry.
Record which pattern 54/55 shows; do not assume either.

⚠ EDIT BEFORE RUNNING — set from the r03 verdict:

    WINNER      :iso | :smile | :structural
                  :iso        → sweep market_weight on TeamIsoDPGoalsModel
                  :smile      → sweep (sup × sw) on the smile engine
                  :structural → the r03 winner has NO market pillar (funnel / rating / none). Then
                                this runner tests the winner ⊕ iso pillar FUSION instead, at
                                mw ∈ {0.25, 0.40}, against the un-fused winner.
                                ⚠ PRE-COMMIT: keep the MODULAR engine unless fusion clearly wins.
                                funnel⊕iso fusion was SOFT-NEGATIVE on 56/57 (r07b) — it regressed
                                the 1X2 edge back toward the market.
    HL          the r03 half-life verdict (365 unless the hl180 control reversed the gradient)

`mw = 0.0` is included on purpose: it is the structural control at otherwise identical spec, so the
sweep contains its own baseline and does not depend on cross-grid comparability.

BETFAIR: still none for 54/55 at time of writing. If the historical odds land before this runs, add
an anchor A/B here — the winning cell duplicated with the pillar built from the Betfair close:

    odds_bf = Data.summarize_betfair_market(ds; open_window=(-100000.0,-10.0), close_window=(-20.0,0.0))
    ds_bf   = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds_bf,
                             ds.lineups, ds.incidents, ds.betfair_odds, ds.bbc, ds.bbc_events)
    # ⚠ ALL NINE FIELDS. The 7-arg positional constructor silently drops bbc + bbc_events, which
    #   would degrade the funnel arm to goals-only and zero the rating/plus-minus features.

and score BOTH arms against the SAME benchmark — anchoring to Betfair and then scoring against
Betfair mechanically compresses the spread, which is what inflated the original Ireland read.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_upper/r04_grid_pillar.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using Dates

pinthreads(:cores)

const Experiments = BayesianFootball.Experiments

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_upper/l01_upper.jl"))

# ---- knobs (edit after r03) ----
const WINNER          = :iso        # :iso | :smile | :structural
const HL              = 365.0
const HISTORY_SEASONS = 2
const SAMPLES         = 800
const WARMUP          = 300
const CHAINS          = 4
const DYN_COL         = :match_biweek
# only used when WINNER == :structural — which engine won r03
const STRUCTURAL_WINNER = :funnel   # :funnel | :rating | :none

println("[INFO] Loading ScottishUpper DataStore...")
ds = Data.load_datastore_cached(Data.ScottishUpper())
save_dir = joinpath(ROOT, "data/scottish_upper_pillar/")
mkpath(save_dir)

season_strings = sort(unique(String.(ds.matches.season)))
const TARGETS = season_strings[max(1, end-1):end]
println("[INFO] targets = ", TARGETS, "  WINNER = ", WINNER, "  HL = ", HL)

# ==========================================
# CELLS
# ==========================================
specs = Tuple{String, Any}[]
if WINNER === :iso
    for mw in (0.0, 0.25, 0.40, 0.70, 1.0)
        push!(specs, ("iso_pois_mw$(Int(round(100mw)))_hl$(Int(HL))",
                      team_iso_dp_goals(mw = mw, hl = HL)))
    end
elseif WINNER === :smile
    for sup in (0.4, 1.0), sw in (0.4, 0.5)
        push!(specs, ("smile_pois_sup$(Int(round(100sup)))_sw$(Int(round(100sw)))_hl$(Int(HL))",
                      team_smile_dp_goals(sup = sup, sw = sw, hl = HL)))
    end
    push!(specs, ("smile_pois_sup100_sw0_hl$(Int(HL))", team_smile_dp_goals(sup = 1.0, sw = 0.0, hl = HL)))
elseif WINNER === :structural
    # The r03 winner has no market pillar. Test FUSION against the un-fused winner, plus the iso
    # pillar alone so the three-way comparison is on one grid.
    base_ctor = STRUCTURAL_WINNER === :funnel ? team_funnel_goals :
                STRUCTURAL_WINNER === :rating ? team_rating_dp_goals : team_dp_goals
    push!(specs, ("$(STRUCTURAL_WINNER)_pois_nofuse_hl$(Int(HL))", base_ctor(hl = HL)))
    for mw in (0.25, 0.40)
        push!(specs, ("iso_pois_mw$(Int(round(100mw)))_hl$(Int(HL))", team_iso_dp_goals(mw = mw, hl = HL)))
    end
    @warn """
    WINNER=:structural — no src engine fuses $(STRUCTURAL_WINNER) with an iso market pillar, so this
    grid compares them SIDE BY SIDE rather than fusing. Building a genuine fused engine is a loader
    change (add an iso pillar term to the $(STRUCTURAL_WINNER) engine) and is NOT done here on
    purpose: 56/57 r07b found funnel⊕iso fusion soft-negative. Only build it if the side-by-side
    shows the two win DIFFERENT families (i.e. genuinely complementary information).
    """
else
    error("WINNER must be :iso, :smile or :structural — got $WINNER")
end

println("[INFO] Pillar grid: $(length(specs)) cells -> ", join(first.(specs), ", "))

# ==========================================
# CONVERGENCE GATE
# ==========================================
function _fold_convergence(res)
    n_ok = 0; worst = 0.0; n = length(res.training_results.items)
    offenders = Dict{String, Int}()
    for it in res.training_results.items
        er = DataFrame(MCMCChains.ess_rhat(it[1]))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
        mr = isempty(vals) ? NaN : maximum(vals)
        isnan(mr) && continue
        worst = max(worst, mr)
        if mr <= 1.01
            n_ok += 1
        else
            # WHICH parameter misses matters: on 56/57 it was always the team ratings, never the
            # market σ — that is the "heavier anchoring stiffens rating geometry" signature.
            i = argmax(coalesce.(replace(er[!, rcol], NaN => -Inf), -Inf))
            p = string(er.parameters[i])
            offenders[p] = get(offenders, p, 0) + 1
        end
    end
    return n, n_ok, worst, offenders
end

# ==========================================
# RUN
# ==========================================
gate_lines = String[]
t_start = time()
for (name, model) in specs
    println("\n", "#"^72,
            "\n# CELL: $name  (elapsed $(round((time()-t_start)/60, digits=1)) min)\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = TARGETS,
            history_seasons = HISTORY_SEASONS,
            warmup_period   = 0,
            dynamics_col    = DYN_COL,
            samples         = SAMPLES,
            warmup          = WARMUP,
            chains          = CHAINS,
            use_queue       = true,
            max_depth       = 10,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)

        n, n_ok, worst, offenders = _fold_convergence(res)
        pct = n == 0 ? 0.0 : round(100n_ok / n, digits=1)
        top = isempty(offenders) ? "-" :
              join(["$(k)×$(v)" for (k, v) in sort(collect(offenders), by = last, rev = true)[1:min(3, end)]], ", ")
        gate = "$name: folds=$n converged=$n_ok ($(pct)%) worst=$(round(worst, digits=4)) offenders=[$top]" *
               (n == 0 ? "  ⚠ SILENT DROP — no items!" : pct < 95 ? "  ⚠ BELOW GATE" : "  ✅")
        println("[GATE] ", gate)
        push!(gate_lines, gate)
    catch e
        msg = "$name: FAILED ($(typeof(e)))"
        println("[GATE] ", msg)
        push!(gate_lines, msg)
        @error "cell failed: $name" exception=(e, catch_backtrace())
    end
end

open(joinpath(ROOT, "current_development/scottish_upper/r04_convergence.txt"), "w") do io
    println(io, "r04 pillar grid convergence gate — ", string(now()))
    println(io, "WINNER=", WINNER, " HL=", HL, " targets=", TARGETS, " hs=", HISTORY_SEASONS)
    foreach(l -> println(io, l), gate_lines)
    println(io, "\nNOTE: if the offenders are team ratings (dyn.raw_a/raw_d) rather than the market")
    println(io, "σ, that reproduces the 56/57 finding that heavier anchoring stiffens rating geometry.")
end

println("\n[INFO] Pillar grid complete in $(round((time()-t_start)/3600, digits=2)) h. ",
        "Gate → r04_convergence.txt. Next: r05_eval_pillar.jl")
