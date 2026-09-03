# ==============================================================================
# Experiment 06 · l64 — hierarchical team-kappa candidates
# ==============================================================================
#
# Loader. Definitions and shared experiment state only; r64/r65 execute.
#
# WHAT CHANGES FROM l60. Exactly one thing: the two-arm observation's finishing
# factor stops being one number for the league and becomes one per team, partially
# pooled around it.
#
#     l60:  log κ            ~ Normal(0, 0.20)
#     l64:  log κ_t = log κ + σ_κ · (raw_t − mean(raw)),  σ_κ ~ truncated(Normal(0, 0.10), 0, ∞)
#
# Everything else — the Gamma arm, the pxG feature, the 180-day team decay, the
# shots-RAPM pillar, the wealth covariate, the splitter, the book and the policy —
# is REUSED from `l60_loader.jl` rather than restated, so a difference in a measured
# number cannot come from a recipe that quietly drifted. The sampler keeps l60's
# 4 × 800 warmup/retained budget but raises target acceptance from 0.65 to 0.90:
# the first production-settings smoke found one divergence in the weakly identified
# σ_κ geometry at 0.65. This file includes l60; it does not copy the shared recipe.
#
# THE TWO CANDIDATES, AND WHY BOTH.
#   m12_hierarchical_kappa   the full hybrid (wealth + shots-RAPM bench pillar)
#   m05_hierarchical_kappa   the team-state control, no lineup arm
# `m05` is what makes any m12 movement attributable. Team finishing and player
# quality are the same story told twice; if per-team κ only pays in the model that
# already carries a lineup pillar, that is a redundancy finding, not a synergy one.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using SHA
using ThreadPinning

# `ds`, `db`, the splitter/sampler/book/policy builders and the six shared-κ
# candidates all come from here. Re-registering l60's configs is idempotent.
include(joinpath(@__DIR__, "l60_loader.jl"))

const L64_MODEL_NAMES = [
    "m05_hierarchical_kappa",
    "m12_hierarchical_kappa",
]

# The shared-κ run each hierarchical candidate must be read against. Same recipe,
# same fold set, one component different.
const L64_CONTROL_OF = Dict(
    "m05_hierarchical_kappa" => "m05_joint_production_wealth",
    "m12_hierarchical_kappa" => "m12_joint_hybrid_synergy",
)

const L64_MODEL_DESCRIPTIONS = Dict(
    "m05_hierarchical_kappa" =>
        "Two-arm joint control with hierarchical team kappa: team time decay, age-adjusted " *
        "production wealth, per-team finishing factor pooled around the league factor.",
    "m12_hierarchical_kappa" =>
        "Master synergy with hierarchical team kappa: production wealth, shots-RAPM starters " *
        "and bench 0.10, per-team finishing factor pooled around the league factor.",
)

const L64_TAGS = [L60_TAGS; "hierarchical-kappa"]

"""
The prior on the team-finishing spread.

Tight on purpose. A team contributes roughly 40 matches to a Scottish Lower fold, and
at ~1.3 goals a side that is Poisson noise standing about 0.13 in log space per team.
A σ_κ prior wide enough to admit a 30% finishing edge would let the sampler fit that
noise and report it as a finding; `truncated(Normal(0, 0.10), 0, ∞)` puts ~90% of its
mass below a 17% swing and lets the data argue upward if the effect is real.

It is HALF-OPEN, not folded: σ_κ < 0 would flip the sign of every delta and leave the
likelihood identical, which is a label-switching mode rather than a wider prior.
"""
l64_sigma_kappa_prior() = truncated(Normal(0.0, 0.10), 0.0, Inf)

"The l60 two-arm observation, with the finishing factor made hierarchical."
l64_observation() = JointGammaPoissonObservation(
    feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
    kappa = HierarchicalKappa(σ_prior = l64_sigma_kappa_prior()),
)

"Build both hierarchical candidates in canonical order (control first)."
function l64_models()
    observation = l64_observation()

    m05 = CountModelBuilder(:m05_hierarchical_kappa) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(l60_production_wealth()) |>
        add(observation) |>
        build

    m12 = CountModelBuilder(:m12_hierarchical_kappa) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(l60_shots_pillar(BenchWeightedPlayerAggregation(w_bench = 0.10))) |>
        add(l60_production_wealth()) |>
        add(observation) |>
        build

    return Tuple{String,Any}[
        (L64_MODEL_NAMES[1], m05),
        (L64_MODEL_NAMES[2], m12),
    ]
end

"""
    l64_expected_params(name, n_teams) -> Int

The structural parameter contract, stated independently of the builder so the smoke
test checks the count rather than agreeing with itself.

    inter.μ, ha.γ_global, dyn.σ_a, dyn.σ_d           4 scalars
    dyn.raw_a, dyn.raw_d                             2·n_teams
    obs.ν, obs.log_κ, obs.σ_κ                        3 scalars
    obs.κ_team_raw                                   n_teams
    production_wealth.w                              1
    lineup.w_att, lineup.w_def                       2   (m12 only)
"""
function l64_expected_params(name::String, n_teams::Int)
    name == "m05_hierarchical_kappa" && return 3 * n_teams + 8
    name == "m12_hierarchical_kappa" && return 3 * n_teams + 10
    error("no structural parameter contract for $name")
end

"Production NUTS budget with a conservative target acceptance for the hierarchical scale."
l64_sampler() = QueuedNUTSConfig(
    n_samples = 800,
    n_warmup = 800,
    n_chains = 4,
    accept_rate = 0.90,
)

function l64_fit_configs(models, splitter, sampler; save_root = joinpath(@__DIR__, "results"))
    return Dict(name => FitConfig(
        name = name,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = QueuedExecution(),
        tags = copy(L64_TAGS),
        description = L64_MODEL_DESCRIPTIONS[name],
        save_dir = joinpath(save_root, name),
    ) for (name, model) in models)
end

"The persisted-run deduplication hash for a FitConfig, without constructing a dummy Fit."
function l64_run_config_hash(db, config::FitConfig)
    tags = filter(config.tags) do tag
        !any(prefix -> startswith(tag, prefix), ("time:", "folds_failed:", "latents:"))
    end
    canonical = join((db.experiment_name, config.name,
                      string(config.model), string(config.splitter),
                      string(config.sampler), string(config.execution),
                      join(tags, "\u001f"), config.description), "\u001e")
    return bytes2hex(SHA.sha256(canonical))
end

"Return the completed run UUID for this exact production recipe, or `nothing`."
function l64_completed_run_id(db, config::FitConfig)
    inference = BayesianFootball.Training.Inference
    conn = inference._db_connect(db)
    try
        rows = inference._db_rows(conn, """
            SELECT r.run_id
            FROM configs AS c
            JOIN runs AS r ON r.run_id = c.config_id
            WHERE c.config_hash = \$1 AND r.status = 'completed'
            LIMIT 1;
        """, (l64_run_config_hash(db, config),))
        return isempty(rows) ? nothing : string(rows.run_id[1])
    finally
        close(conn)
    end
end

"Register the hierarchical models and their assembled fit recipes in config_registry."
function l64_register!(db, models, splitter, sampler, configs)
    model_ids = Dict{String,Int}()
    fit_hashes = Dict{String,String}()
    for (name, model) in models
        model_ids[name] = save_model(
            db, name, model;
            description = L64_MODEL_DESCRIPTIONS[name],
            tags = L64_TAGS,
        )
        fit_hashes[name] = save_config(
            db, name * "_fit", configs[name];
            description = L64_MODEL_DESCRIPTIONS[name],
            tags = L64_TAGS,
        )
    end
    sampler_id = save_sampler(
        db, "queued_nuts_4x800_hierarchical_kappa", sampler;
        description = "Hierarchical-kappa production NUTS: 4 chains, 800 warmup and " *
                      "800 retained draws, target acceptance 0.90.",
        tags = L64_TAGS,
    )
    return (; model_ids, sampler_id, fit_hashes)
end

# ------------------------------------------------------------------------------
# Shared experiment state. The splitter, book and policy are l60's own objects —
# not copies with the same field values — so the hierarchical grid lands on the
# identical 40 boundaries and portfolio recipe as the shared-κ leaderboard. The
# sampler differs only in target acceptance, for the convergence reason above.
# ------------------------------------------------------------------------------
l64_candidate_models = l64_models()
l64_production_splitter = l60_production_splitter
l64_production_sampler = l64_sampler()
l64_configs = l64_fit_configs(
    l64_candidate_models, l64_production_splitter, l64_production_sampler)
l64_registry = l64_register!(
    db, l64_candidate_models, l64_production_splitter, l64_production_sampler, l64_configs)
