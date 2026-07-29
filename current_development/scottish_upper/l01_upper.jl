#=
l01 — loader for the Scottish Upper (54/55) bake-off.

Deliberately THIN. Almost every engine in the ladder is already in src; this file only supplies the
two things that are not:

  1. `TeamDPGoalsModel` / `TeamIsoDPGoalsModel` — the 56/57 structural baseline and the 56/57
     PRODUCTION WINNER (iso market pillar). `TeamIsoDPGoalsModel` never graduated to src; it lives
     only in the sibling stream's loader. We `include` that file rather than copying it, because it
     also ships the `PreGame.extract_parameters` / `Pred.compute_score_matrix` overrides those
     loader-local structs need. Without them `Evaluation.evaluate_experiments` SILENTLY DROPS those
     cells' rows (they eval-fail to NaN rather than erroring).

  2. `team_rating_dp_goals(...)` — the "player ratings" arm. This is NOT a new engine: it is
     `PreGame.DynamicGoalsPlusMinusLeagueTimeDecayModel` configured with `PlayerRatingsFeature`
     instead of a RAPM feature. That engine's body is rating-source-agnostic (it reads the
     `flat_<side>_<pos>_rating` vectors and centres with `Features.rating_base`, which dispatches per
     family), and its type parameter was widened in src for exactly this. Reusing it means we inherit
     its prediction-dispatch registration, its extractor and its sufficient-statistic likelihood for
     free — nothing new to add to the score-computation Union.

⚠ TWO SCALE/COVERAGE HAZARDS ON THE RATINGS ARM (both handled here, both worth re-checking in r01):

  (a) SCALE. RAPM's `w_*_prior = Normal(0, 0.3)` is calibrated to a rating whose 10-man sum sits at
      O(0.3-0.5). A minute-weighted SofaScore sum sits at ~10 × 6.5 ≈ 65, so its CENTRED value has
      sd of order 1-3. At sd 0.3 the pillar alone would swing log-λ by ±0.6+. We pass
      `Normal(0, 0.05)` and r01 prints the realised centred-rating sd so the prior can be sanity
      checked against it.

  (b) COVERAGE. Tournament 55 has NO player ratings and NO positive minutes before 23/24. The
      extractor emits 0.0 for such a side, and an unmasked centring would turn that into −10·6.5 =
      −65 — a fake "ten standard deviations below average" side. src `_pm_outfield` now MASKS this
      (0 total ⇒ contribute 0 = league average). Verify in r01 that the 22/23 history block does not
      produce a bimodal rating distribution.

Include from any runner in this stream:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_upper/l01_upper.jl"))
=#

using BayesianFootball
using Distributions

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

const _ROOT = pkgdir(BayesianFootball)

# --- 1. Bring in the 56/57 team engines (TeamDPGoalsModel, TeamIsoDPGoalsModel, TeamSmileDPGoalsModel)
#        together with their prediction overrides. Guarded so repeated includes are cheap/safe.
if !@isdefined(TeamIsoDPGoalsModel)
    include(joinpath(_ROOT, "current_development/scottish_lower_smile/l01_team_dp_league.jl"))
end

# ==========================================
# 2. SHARED COMPONENT CONFIG
# ==========================================
# Held IDENTICAL across every cell in the grid — comparability is the whole point of the bake-off.
# Only the thing under test may vary between cells.
const HL_DEFAULT = 365.0     # 56/57 verdict: monotone gradient favouring long memory
const KMAX       = 4         # smile ladder depth (r00 confirms u05..u45 density)

_inter() = PreGame.HierarchicalMonthlyInterception()
_ha()    = PreGame.HierarchicalTeamHomeAdvantage()
_disp()  = PreGame.HomeAwayDispersion()
_dyn(hl = HL_DEFAULT) = PreGame.TimeDecayDynamics(days_half_life = hl)

# The tracker used throughout split_market_pillar (Ireland): BayesianTracker(prior_mean, prior_var,
# obs_var, drift). `prior_mean = 6.5` is what `Features.rating_base` returns, i.e. the centring point.
_ratings_feature() = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))

# ==========================================
# 3. CELL CONSTRUCTORS
# ==========================================

"""
    team_dp_goals(; hl) -> TeamDPGoalsModel

`none_pois` — structural baseline: pooled double-Poisson goals + zero-sum δ_league, NO market pillar.
"""
team_dp_goals(; hl = HL_DEFAULT) =
    TeamDPGoalsModel(
        interception_config  = _inter(),
        dynamics_config      = _dyn(hl),
        homeadvantage_config = _ha(),
    )

"""
    team_iso_dp_goals(; mw, hl) -> TeamIsoDPGoalsModel

`iso_pois` — the 56/57 PRODUCTION WINNER: isotropic market pillar with a SAMPLED σ (never fixed) and
a scalar `market_weight`. `mw = 0.0` is the structural control at otherwise identical spec.
"""
team_iso_dp_goals(; mw = 0.40, hl = HL_DEFAULT) =
    TeamIsoDPGoalsModel(
        interception_config   = _inter(),
        dynamics_config       = _dyn(hl),
        homeadvantage_config  = _ha(),
        market_feature_config = Features.DoublePoissonMarketFeature(),
        market_weight         = mw,
        market_on             = mw > 0.0,
    )

"""
    team_smile_dp_goals(; sup, sw, hl) -> DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel

`smile_pois` — Ireland's keeper pillar (supremacy anchor + per-strike local-intensity smile) at the
team/goals level. Built from SRC, not the loader copy.

⚠ EXPENSIVE. On 56/57 this cost ~20× the structural engine (tight sampled σ_smile ⇒ ~8.5× more
leapfrogs/iteration). It is ordered LAST in the grid and is the first cut if r01's runtime
calibration says it will not fit. Depth caps are NOT an escape hatch — all three capped smile cells
failed even the ranking gate on 56/57.
"""
team_smile_dp_goals(; sup = 1.0, sw = 0.5, hl = HL_DEFAULT) =
    PreGame.DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel(
        interception_config   = _inter(),
        dynamics_config       = _dyn(hl),
        homeadvantage_config  = _ha(),
        market_feature_config = Features.DoublePoissonMarketFeature(),
        smile_feature         = Features.MarketSmileFeature(Kmax = KMAX),
        market_on             = true,
        supremacy_weight      = sup,
        smile_weight          = sw,
    )

"""
    team_funnel_goals(; hl) -> DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel

`funnel` — two-layer thinned Poisson, Shots ~ Poisson(λ_s) and Goals ~ Poisson(λ_s·p₂). The 56/57
r06 winner on 1X2. NO market pillar.

Shots come from `Features.ShotsFunnelFeature`, i.e. **`ds.bbc`** — the BBC match pages, which cover
BOTH tiers 100% back to 2020. Do NOT be tempted to swap in SofaScore `ShotsFeature`: tournament 55
has no SofaScore shot stats before 23/24, so half the history would silently vanish behind the mask.
"""
team_funnel_goals(; hl = HL_DEFAULT) =
    PreGame.DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel(
        interception_config  = PreGame.HierarchicalMonthlyInterception(
                                   prior_μ_base = Distributions.Normal(0.0, 0.3)),
        dynamics_config      = _dyn(hl),
        homeadvantage_config = _ha(),
    )

"""
    team_rating_dp_goals(; hl, w_sd) -> DynamicGoalsPlusMinusLeagueTimeDecayModel

`rating` — goals + zero-sum δ_league + a SofaScore player-rating pillar, on top of (not instead of)
the team dynamics α/β.

Keeping the team dynamics alongside the pillar is deliberate: the APM graduation found the
pillar-ONLY variant significantly worse than pillar+dynamics (t +2.3). The Ireland `outfield_*`
engines let ratings REPLACE team dynamics; that is the wrong shape here.

`w_sd` defaults to 0.05, not the engine's RAPM-calibrated 0.3 — see hazard (a) in the header.
"""
team_rating_dp_goals(; hl = HL_DEFAULT, w_sd = 0.05) =
    PreGame.DynamicGoalsPlusMinusLeagueTimeDecayModel(
        interception_config    = _inter(),
        dynamics_config        = _dyn(hl),
        homeadvantage_config   = _ha(),
        player_ratings_feature = _ratings_feature(),
        w_att_prior            = Distributions.Normal(0.0, w_sd),
        w_def_prior            = Distributions.Normal(0.0, w_sd),
    )

"""
    team_nb_goals(; hl) -> DynamicGoalsTimeDecayModel

`none_nb` — NegBin dispersion reference. On 56/57 (V/M ≈ 0.94, sub-Poisson) `r` went inert and NB
never beat Poisson. If NB clearly wins here, STOP and investigate the dispersion regime before
reading anything else — 54/55 could be a different animal (r00 §3 measures V/M).
"""
team_nb_goals(; hl = HL_DEFAULT) =
    PreGame.DynamicGoalsTimeDecayModel(
        interception_config  = _inter(),
        dynamics_config      = _dyn(hl),
        dispersion_config    = _disp(),
        homeadvantage_config = _ha(),
    )

# ==========================================
# 4. THE LADDER
# ==========================================
"""
    family_specs(; hl, include_smile) -> Vector{Tuple{String, Any}}

The Night-1 bake-off ladder, in RUN ORDER: cheap cells first so a blown budget costs the least
informative cell, not the baseline. `smile_sw50` is last and opt-in.

Cell names follow the canonical `<pillar>_<disp>[_<knob><val>]` convention inherited from
split_market_pillar/NOTES.md.
"""
function family_specs(; hl = HL_DEFAULT, include_smile::Bool = true)
    specs = Tuple{String, Any}[
        ("none_pois_hl$(Int(hl))",  team_dp_goals(hl = hl)),
        ("none_pois_hl180",         team_dp_goals(hl = 180.0)),   # half-life gradient control
        ("rating_pois_hl$(Int(hl))", team_rating_dp_goals(hl = hl)),
        ("funnel_pois_hl$(Int(hl))", team_funnel_goals(hl = hl)),
        ("none_nb_hl$(Int(hl))",     team_nb_goals(hl = hl)),
        ("iso_pois_mw40_hl$(Int(hl))", team_iso_dp_goals(mw = 0.40, hl = hl)),
    ]
    include_smile && push!(specs,
        ("smile_pois_sup100_sw50_hl$(Int(hl))", team_smile_dp_goals(sup = 1.0, sw = 0.5, hl = hl)))
    return specs
end

println("[l01_upper] loaded — engines: none/iso/smile (56/57 + src), funnel, rating, nb. ",
        "HL_DEFAULT=$(HL_DEFAULT), KMAX=$(KMAX)")
