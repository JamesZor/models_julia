# ==============================================================================
# Experiment 05 shared loader — canonical player-lineup grid recipes
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const L50_FEATURES = BayesianFootball.Features
const L50_PORTFOLIO = BayesianFootball.Portfolio

const L50_MODEL_NAMES = [
    "m05_joint_production_wealth",
    "m09_player_shots_rapm_outfield",
    "m10_player_shots_rapm_bench",
    "m11_player_pxg_rapm_bench",
    "m12_hybrid_production_wealth_player_rapm",
]

const L50_MODEL_DESCRIPTIONS = Dict(
    "m05_joint_production_wealth" =>
        "Two-arm joint control with Richards-sigmoid age-adjusted production wealth.",
    "m09_player_shots_rapm_outfield" =>
        "Two-arm joint model with shots-RAPM starting outfield lineups.",
    "m10_player_shots_rapm_bench" =>
        "Two-arm joint model with shots-RAPM starters and fixed 0.10 bench weight.",
    "m11_player_pxg_rapm_bench" =>
        "Two-arm joint model with pxG-RAPM starters and fixed 0.10 bench weight.",
    "m12_hybrid_production_wealth_player_rapm" =>
        "Two-arm joint master synergy: production wealth plus shots-RAPM starters and bench 0.10.",
)

const L50_TAGS = ["scottish-lower", "24/25", "25/26", "player-lineup", "joint-pxg"]

"The common two-arm observation; the Gamma arm uses commentary pxG only."
l50_observation() = JointGammaPoissonObservation(
    feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
)

l50_production_wealth() = ProductionWealthCovariate(
    feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
    prior = truncated(Normal(0.10, 0.05), lower = 0.0),
)

l50_shots_dynamics(aggregation) = PlayerLineupDynamics(
    feature = L50_FEATURES.ShotsPlusMinusFeature(
        w_sim = 0.0,
        λ = 1000.0,
        half_life_days = 730.0,
        fit_on = :history,
    ),
    aggregation = aggregation,
    w_att_prior = Normal(0.0, 0.3),
    w_def_prior = Normal(0.0, 0.3),
)

l50_pxg_dynamics(aggregation) = PlayerLineupDynamics(
    feature = L50_FEATURES.XGPlusMinusFeature(
        w_sim = 0.0,
        λ = 200.0,
        half_life_days = 730.0,
        fit_on = :history,
    ),
    aggregation = aggregation,
    w_att_prior = Normal(0.0, 0.3),
    w_def_prior = Normal(0.0, 0.3),
)

"Build all five candidates in canonical leaderboard order."
function l50_models()
    observation = l50_observation()

    m05 = CountModelBuilder(:m05_joint_production_wealth) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(l50_production_wealth()) |>
        add(observation) |>
        build

    m09 = CountModelBuilder(:m09_player_shots_rapm_outfield) |>
        add(GlobalInterception()) |>
        add(l50_shots_dynamics(OutfieldPlayerAggregation())) |>
        add(GlobalHomeAdvantage()) |>
        add(observation) |>
        build

    m10 = CountModelBuilder(:m10_player_shots_rapm_bench) |>
        add(GlobalInterception()) |>
        add(l50_shots_dynamics(BenchWeightedPlayerAggregation(w_bench = 0.10))) |>
        add(GlobalHomeAdvantage()) |>
        add(observation) |>
        build

    m11 = CountModelBuilder(:m11_player_pxg_rapm_bench) |>
        add(GlobalInterception()) |>
        add(l50_pxg_dynamics(BenchWeightedPlayerAggregation(w_bench = 0.10))) |>
        add(GlobalHomeAdvantage()) |>
        add(observation) |>
        build

    m12 = CountModelBuilder(:m12_hybrid_production_wealth_player_rapm) |>
        add(GlobalInterception()) |>
        add(l50_shots_dynamics(BenchWeightedPlayerAggregation(w_bench = 0.10))) |>
        add(GlobalHomeAdvantage()) |>
        add(l50_production_wealth()) |>
        add(observation) |>
        build

    return Tuple{String,Any}[
        (L50_MODEL_NAMES[1], m05),
        (L50_MODEL_NAMES[2], m09),
        (L50_MODEL_NAMES[3], m10),
        (L50_MODEL_NAMES[4], m11),
        (L50_MODEL_NAMES[5], m12),
    ]
end

"Canonical 40-boundary Scottish League One/Two walk-forward split."
l50_splitter() = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25", "25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    end_dynamics = nothing,
    stop_early = true,
)

l50_sampler() = QueuedNUTSConfig(
    n_samples = 800,
    n_warmup = 800,
    n_chains = 4,
    accept_rate = 0.65,
)

l50_book_spec() = BookSpec(
    markets = Data.MarketConfig(Data.AbstractMarket[
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
        Data.MarketBTTS(),
    ]),
    price = DeArb(),
    allocator = KellyLogUtility(),
    shrink = L50_PORTFOLIO.FractionalKelly(0.30),
    exec = ExecutionConfig(
        commission = PerBetCommission(0.02),
        budget = 0.99,
        min_selection_stake = 0.001,
    ),
)

l50_policy_spec() = PolicySpec(
    trust = FlatTrust(1.0),
    risk = SlateDrawdown(23.0),
    cap = FixedCap(0.20),
    grouping = DailySlate(),
)

function l50_fit_configs(models, splitter, sampler; save_root = joinpath(@__DIR__, "results"))
    return Dict(name => FitConfig(
        name = name,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = QueuedExecution(),
        tags = copy(L50_TAGS),
        description = L50_MODEL_DESCRIPTIONS[name],
        save_dir = joinpath(save_root, name),
    ) for (name, model) in models)
end

"Register every canonical component and assembled fit recipe in config_registry."
function l50_register!(db, models, splitter, sampler, configs, book_spec, policy_spec)
    model_ids = Dict{String,Int}()
    fit_hashes = Dict{String,String}()
    for (name, model) in models
        model_ids[name] = save_model(
            db, name, model;
            description = L50_MODEL_DESCRIPTIONS[name],
            tags = L50_TAGS,
        )
    end
    splitter_id = save_splitter(
        db, "scottish_lower_2426_player_40fold", splitter;
        description = "Pooled tournaments 56/57, match-biweek walk-forward over 24/25 and 25/26.",
        tags = L50_TAGS,
    )
    sampler_id = save_sampler(
        db, "queued_nuts_4x800_player_grid", sampler;
        description = "Production queued NUTS: 4 chains, 800 warmup and 800 retained draws.",
        tags = L50_TAGS,
    )
    for name in L50_MODEL_NAMES
        fit_hashes[name] = save_config(
            db, name * "_fit", configs[name];
            description = L50_MODEL_DESCRIPTIONS[name],
            tags = L50_TAGS,
        )
    end
    book_id = save_book_spec(
        db, "fractional_kelly_main_markets", book_spec;
        description = "30% fractional Kelly on 1X2, O/U 2.5 and BTTS with exchange commission.",
        tags = [L50_TAGS; "portfolio"],
    )
    policy_id = save_policy_spec(
        db, "daily_drawdown_cap_player_grid", policy_spec;
        description = "Daily slate drawdown control with 20% exposure cap.",
        tags = [L50_TAGS; "portfolio"],
    )
    return (; model_ids, splitter_id, sampler_id, fit_hashes, book_id, policy_id)
end

# Required shared experiment state.
ds = Data.load_datastore_cached(Data.ScottishLower())
db = PostgresStorage("scottish_lower_player_grid_2426")
ensure_schema!(db)

l50_candidate_models = l50_models()
l50_production_splitter = l50_splitter()
l50_production_sampler = l50_sampler()
l50_book = l50_book_spec()
l50_policy = l50_policy_spec()
l50_configs = l50_fit_configs(
    l50_candidate_models, l50_production_splitter, l50_production_sampler)
l50_registry = l50_register!(
    db, l50_candidate_models, l50_production_splitter, l50_production_sampler,
    l50_configs, l50_book, l50_policy)
