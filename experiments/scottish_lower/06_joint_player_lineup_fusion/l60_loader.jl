# ==============================================================================
# Experiment 06 shared loader — canonical two-arm joint + player-lineup recipes
# ==============================================================================
#
# Loader. Definitions and shared experiment state only; the runners execute.
#
# Every candidate holds the two-arm `JointGammaPoissonObservation` fixed and varies
# only what feeds the log-rate: team time-decay dynamics, point-in-time player RAPM
# lineups, age-adjusted production wealth, and static away-travel burden.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const L60_FEATURES = BayesianFootball.Features
const L60_PORTFOLIO = BayesianFootball.Portfolio

const L60_MODEL_NAMES = [
    "m05_joint_production_wealth",
    "m09_joint_player_shots_outfield",
    "m10_joint_player_shots_bench",
    "m11_joint_player_pxg_bench",
    "m12_joint_hybrid_synergy",
    "m13_joint_composite",
]

const L60_MODEL_DESCRIPTIONS = Dict(
    "m05_joint_production_wealth" =>
        "Two-arm joint control: team time decay plus Richards-sigmoid age-adjusted production wealth.",
    "m09_joint_player_shots_outfield" =>
        "Two-arm joint model with shots-RAPM starting outfield lineups.",
    "m10_joint_player_shots_bench" =>
        "Two-arm joint model with shots-RAPM starters and fixed 0.10 bench weight.",
    "m11_joint_player_pxg_bench" =>
        "Two-arm joint model with pxG-RAPM starters and fixed 0.10 bench weight.",
    "m12_joint_hybrid_synergy" =>
        "Master synergy: production wealth plus shots-RAPM starters and bench 0.10.",
    "m13_joint_composite" =>
        "Composite: production wealth, away-travel distance, and shots-RAPM starters with bench 0.10.",
)

const L60_TAGS = ["scottish-lower", "24/25", "25/26", "joint-player", "player-lineup"]

"The common two-arm observation; the Gamma arm uses commentary pxG only."
l60_observation() = JointGammaPoissonObservation(
    feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
)

l60_production_wealth() = ProductionWealthCovariate(
    feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
    prior = truncated(Normal(0.10, 0.05), lower = 0.0),
)

# Static away-travel burden. `:log_dist_z` is the catalog-standardised production
# metric, so the coefficient is read per standard deviation of log travel.
l60_distance() = DistanceCovariate(
    feature = L60_FEATURES.DistanceFeature(metric = :log_dist_z),
    prior = truncated(Normal(0.04, 0.03), lower = 0.0),
)

l60_shots_pillar(aggregation) = PlayerLineupPillar(
    feature = L60_FEATURES.ShotsPlusMinusFeature(
        w_sim = 0.0,
        λ = 1000.0,
        half_life_days = 730.0,
        fit_on = :history,
    ),
    aggregation = aggregation,
    w_att_prior = Normal(0.0, 0.3),
    w_def_prior = Normal(0.0, 0.3),
)

l60_pxg_pillar(aggregation) = PlayerLineupPillar(
    feature = L60_FEATURES.XGPlusMinusFeature(
        w_sim = 0.0,
        λ = 200.0,
        half_life_days = 730.0,
        fit_on = :history,
    ),
    aggregation = aggregation,
    w_att_prior = Normal(0.0, 0.3),
    w_def_prior = Normal(0.0, 0.3),
)

"Build all six candidates in canonical leaderboard order."
function l60_models()
    observation = l60_observation()

    m05 = CountModelBuilder(:m05_joint_production_wealth) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(l60_production_wealth()) |>
        add(observation) |>
        build

    m09 = CountModelBuilder(:m09_joint_player_shots_outfield) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(l60_shots_pillar(OutfieldPlayerAggregation())) |>
        add(observation) |>
        build

    m10 = CountModelBuilder(:m10_joint_player_shots_bench) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(l60_shots_pillar(BenchWeightedPlayerAggregation(w_bench = 0.10))) |>
        add(observation) |>
        build

    m11 = CountModelBuilder(:m11_joint_player_pxg_bench) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(l60_pxg_pillar(BenchWeightedPlayerAggregation(w_bench = 0.10))) |>
        add(observation) |>
        build

    m12 = CountModelBuilder(:m12_joint_hybrid_synergy) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(l60_shots_pillar(BenchWeightedPlayerAggregation(w_bench = 0.10))) |>
        add(l60_production_wealth()) |>
        add(observation) |>
        build

    m13 = CountModelBuilder(:m13_joint_composite) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(l60_shots_pillar(BenchWeightedPlayerAggregation(w_bench = 0.10))) |>
        add(l60_production_wealth()) |>
        add(l60_distance()) |>
        add(observation) |>
        build

    return Tuple{String,Any}[
        (L60_MODEL_NAMES[1], m05),
        (L60_MODEL_NAMES[2], m09),
        (L60_MODEL_NAMES[3], m10),
        (L60_MODEL_NAMES[4], m11),
        (L60_MODEL_NAMES[5], m12),
        (L60_MODEL_NAMES[6], m13),
    ]
end

"Canonical 40-boundary Scottish League One/Two walk-forward split."
l60_splitter() = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25", "25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    end_dynamics = nothing,
    stop_early = true,
)

l60_sampler() = QueuedNUTSConfig(
    n_samples = 800,
    n_warmup = 800,
    n_chains = 4,
    accept_rate = 0.65,
)

l60_book_spec() = BookSpec(
    markets = Data.MarketConfig(Data.AbstractMarket[
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
        Data.MarketBTTS(),
    ]),
    price = DeArb(),
    allocator = KellyLogUtility(),
    shrink = L60_PORTFOLIO.FractionalKelly(0.30),
    exec = ExecutionConfig(
        commission = PerBetCommission(0.02),
        budget = 0.99,
        min_selection_stake = 0.001,
    ),
)

l60_policy_spec() = PolicySpec(
    trust = FlatTrust(1.0),
    risk = SlateDrawdown(23.0),
    cap = FixedCap(0.20),
    grouping = DailySlate(),
)

function l60_fit_configs(models, splitter, sampler; save_root = joinpath(@__DIR__, "results"))
    return Dict(name => FitConfig(
        name = name,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = QueuedExecution(),
        tags = copy(L60_TAGS),
        description = L60_MODEL_DESCRIPTIONS[name],
        save_dir = joinpath(save_root, name),
    ) for (name, model) in models)
end

"Register every canonical component and assembled fit recipe in config_registry."
function l60_register!(db, models, splitter, sampler, configs, book_spec, policy_spec)
    model_ids = Dict{String,Int}()
    fit_hashes = Dict{String,String}()
    for (name, model) in models
        model_ids[name] = save_model(
            db, name, model;
            description = L60_MODEL_DESCRIPTIONS[name],
            tags = L60_TAGS,
        )
    end
    splitter_id = save_splitter(
        db, "scottish_lower_joint_player_40fold", splitter;
        description = "Pooled tournaments 56/57, match-biweek walk-forward over 24/25 and 25/26.",
        tags = L60_TAGS,
    )
    sampler_id = save_sampler(
        db, "queued_nuts_4x800_joint_player", sampler;
        description = "Production queued NUTS: 4 chains, 800 warmup and 800 retained draws.",
        tags = L60_TAGS,
    )
    for name in L60_MODEL_NAMES
        fit_hashes[name] = save_config(
            db, name * "_fit", configs[name];
            description = L60_MODEL_DESCRIPTIONS[name],
            tags = L60_TAGS,
        )
    end
    book_id = save_book_spec(
        db, "fractional_kelly_main_markets", book_spec;
        description = "30% fractional Kelly on 1X2, O/U 2.5 and BTTS with exchange commission.",
        tags = [L60_TAGS; "portfolio"],
    )
    policy_id = save_policy_spec(
        db, "daily_drawdown_cap_joint_player", policy_spec;
        description = "Daily slate drawdown control with 20% exposure cap.",
        tags = [L60_TAGS; "portfolio"],
    )
    return (; model_ids, splitter_id, sampler_id, fit_hashes, book_id, policy_id)
end

# Required shared experiment state.
ds = Data.load_datastore_cached(Data.ScottishLower())
db = PostgresStorage("scottish_lower_joint_player_2426")
ensure_schema!(db)

l60_candidate_models = l60_models()
l60_production_splitter = l60_splitter()
l60_production_sampler = l60_sampler()
l60_book = l60_book_spec()
l60_policy = l60_policy_spec()
l60_configs = l60_fit_configs(
    l60_candidate_models, l60_production_splitter, l60_production_sampler)
l60_registry = l60_register!(
    db, l60_candidate_models, l60_production_splitter, l60_production_sampler,
    l60_configs, l60_book, l60_policy)
