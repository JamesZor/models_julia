# current_development/matchday_2026_08_08/l01_weekend_training.jl
#
# Helpers for the 7-9 Aug 2026 weekend training run.
#
# THE POINT OF THIS FILE is one function: `floor_warmup`. Everything else is a guard around it.
#
# WHY. `Data._process_tournament_group_ids` cuts folds like this:
#
#     season_dynamics = sort(unique(target_pool[!, dynamics_col]))
#     valid_steps     = filter(t -> t >= config.warmup_period, season_dynamics)
#     ...
#     for t in valid_steps                      # expanding target window, <= t
#
# plus one injected baseline fold (target_step = 0, NO target matches) whenever history is
# non-empty. So `warmup_period` is a *floor on the dynamics step*, and setting it near the end of
# the season is how you collapse a 24-fold walk-forward into the one or two folds you actually
# need to serve this week.
#
# The trap: if `warmup_period` EXCEEDS the season's last step, `valid_steps` is empty. That is not
# an error. The loop simply never runs and you are left with the baseline fold alone — a model
# trained on history only, with zero target-season data, which then prices this week's fixtures.
#
# This is not hypothetical. `r05_ireland_03_07_26.jl` hardcoded `warmup_period = 23`; Ireland's
# 2026 season had reached `match_week = 20` on 2026-07-03. Loading the artefact confirms it:
#
#     data/match_day_ireland/july/dixon_coles_2026-07-03_20260703_121847
#         -> length(training_results) == 1        # the baseline fold, target_sizes == [0]
#
# So that model never saw a single 2026 match. `floor_warmup` + `assert_splits` make that
# unrepresentable rather than merely documented.

using DataFrames, Dates

"""
    floor_warmup(ds, target_season; dynamics_col = :match_week) -> Int

The largest `warmup_period` that is guaranteed to keep at least one dynamic fold.

Takes the **minimum across the pooled tournaments** of each one's last dynamics step, not the
maximum. For a pooled segment the two divisions are rarely level: Ireland is at `match_week` 23
(79) and 24 (718); ScottishUpper is at 2 (54) and 1 (55). Using the max would overshoot the
league that is a round behind and, on a single-tournament segment, sits exactly on the cliff
edge — one more round played between the cache build and the run and you silently lose the season.

The floor costs at most one extra fold (the trailing steps each get their own expanding window)
and buys the guarantee. Every fold's target window is cumulative (`<= t`), so the LAST fold always
contains the full season regardless of which step the floor landed on.
"""
function floor_warmup(ds, target_season::String; dynamics_col::Symbol = :match_week)
    tgt = filter(:season => ==(target_season), ds.matches)
    isempty(tgt) && error("floor_warmup: no matches in season $target_season for $(ds.segment)")

    per_tournament = [maximum(g[!, dynamics_col]) for g in groupby(tgt, :tournament_id)]
    return Int(minimum(per_tournament))
end

"""
    assert_splits(ds, cfg; label = "") -> Vector

Refuse to launch a run whose last fold is empty, and say what it will actually train.

`r05`'s failure mode was invisible because an experiment with one history-only fold looks
perfectly healthy from the outside — it converges, it has chains, it makes predictions.
"""
function assert_splits(ds, cfg; label::String = "")
    b = BayesianFootball.Data.create_id_boundaries(ds, cfg)
    sizes = [length(x[1].target_match_ids) for x in b]
    steps = [x[1].target_step for x in b]

    isempty(b) && error("$label: splitter produced NO boundaries")
    last(sizes) == 0 && error(
        "$label: the last fold has ZERO target matches — warmup_period=$(cfg.warmup_period) " *
        "overshot the season's last $(cfg.dynamics_col). This is the r05 failure: the run " *
        "would train on history only and never see the target season.")

    @info "$label: $(length(b)) folds" steps target_sizes=sizes history=length(b[1][1].history_match_ids)
    return b
end

"""
    weekend_task(ds, model, name, save_dir, target_season; kwargs...)

`create_experiment_task` with the warmup derived rather than typed, and asserted before it runs.

`dynamics_col = :match_week` (not the `:match_biweek` the grid runs used) because we want the
finest available cut of "up to now" — with the warmup floor doing the collapsing, a finer column
just means the last fold lands closer to the present.
"""
function weekend_task(ds, model, name::String, save_dir::String, target_season::String;
                      history_seasons::Int = 2,
                      dynamics_col::Symbol = :match_week,
                      samples::Int = 1000, warmup::Int = 500, chains::Int = 8,
                      max_depth::Int = 10, max_concurrent_tasks::Int = 16)

    Experiments = BayesianFootball.Experiments
    Data_       = BayesianFootball.Data

    w = floor_warmup(ds, target_season; dynamics_col = dynamics_col)

    cfg = Data_.GroupedCVConfig(
        tournament_groups = [Data_.tournament_ids(ds.segment)],
        target_seasons    = [target_season],
        history_seasons   = history_seasons,
        dynamics_col      = dynamics_col,
        warmup_period     = w,
        stop_early        = false)

    assert_splits(ds, cfg; label = name)

    return Experiments.create_experiment_task(
        ds, model, name, save_dir;
        target_seasons  = [target_season],
        history_seasons = history_seasons,
        dynamics_col    = dynamics_col,
        warmup_period   = w,
        samples = samples, warmup = warmup, chains = chains,
        max_depth = max_depth,
        use_queue = true,
        max_concurrent_tasks = max_concurrent_tasks)
end

# ===================================================================
# Model builders
# ===================================================================

"""
    poisson_outfield_model(; market_weight = 0.4)

`DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel` with the component configuration the Ireland
runners have used throughout (`r05_ireland_03_07_26.jl`): hierarchical monthly interception,
home/away dispersion, hierarchical team HA and kappa, 45-day player half-life, Bayesian rating
tracker.

Requires per-player ratings and team xG, so it is usable on 79, 718, 54 and 55 — and NOT on
56/57, which have zero of either.
"""
function poisson_outfield_model(; market_weight::Float64 = 0.4, days_half_life::Float64 = 45.0)
    PG = BayesianFootball.Models.PreGame
    F  = BayesianFootball.Features

    return PG.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
        interception_config    = PG.HierarchicalMonthlyInterception(),
        player_dynamics_config = PG.OutfieldPlayerDynamicsConfig(days_half_life = days_half_life),
        dispersion_config      = PG.HomeAwayDispersion(),
        homeadvantage_config   = PG.HierarchicalTeamHomeAdvantage(),
        kappa_config           = PG.HierarchicalTeamKappa(),
        player_ratings_feature = F.PlayerRatingsFeature(F.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
        market_feature_config  = F.DoublePoissonMarketFeature(),
        market_weight          = market_weight)
end

"""
    funnel_model()

`DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel` — the 2-layer shots -> goals winner from the
funnel stream, and the only engine that can price 56/57 at all: it needs no player ratings and no
xG, just BBC shot counts, which is the entire observable in League One and League Two.

It is also the only engine here carrying a zero-sum `δ_league` offset, which is what the
`LeagueFromFixture` materialiser now supplies at inference.
"""
function funnel_model()
    PG = BayesianFootball.Models.PreGame
    return PG.DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel()
end
