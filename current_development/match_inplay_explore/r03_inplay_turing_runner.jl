#=
r03_inplay_turing_runner.jl  —  Fit & compare Bayesian in-play intensity models.

Config-grid experiment over hierarchical team effects (attack / defense / trailing /
leading), each toggled independently, scored by held-out elpd (by-match split):

  l01 panel  ->  l02 long-format dataset  ->  l03 inputs (design matrix + team idx)
    ->  Turing NUTS per config  ->  held-out elpd ranking + posterior summaries.

Baseline (all team effects off) is the Bayesian twin of the l02 Poisson GLM and should
recover its coefficients (trailing≈+0.25, leading≈−0.24, log_pregame≈1.27 standardised 0.28).

Run with threads:  julia --project -t 32   (then pinthreads(:cores))
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics
using Random
using Turing
using ThreadPinning

pinthreads(:cores)

const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data
const Samplers    = BayesianFootball.Samplers

include("l01_inplay_inverse.jl")
include("l02_inplay_intensity.jl")
include("l03_inplay_turing.jl")

# ==========================================================================
# 1. DATA + PANEL + INPUTS  (reuse l01/l02)
# ==========================================================================
ds = Data.load_datastore_cached(Data.Ireland())
bf = ds.betfair_odds

saved_files      = Experiments.list_experiments("./data/dixon_coles_ab/", data_dir = "")
res_pre_game     = Experiments.load_experiment(saved_files, 1)
pre_game_latents = Experiments.extract_oos_predictions(ds, res_pre_game)
pg_tbl = DataFrame(match_id = Int.(pre_game_latents.df.match_id),
                   pg_λ_h   = [mean(Float64.(v)) for v in pre_game_latents.df.λ_h],
                   pg_λ_a   = [mean(Float64.(v)) for v in pre_game_latents.df.λ_a])

function build_panel(bf, ds, pg_tbl; config = Features.DoublePoissonMarketFeature(),
                     bin_minutes = 5.0, staleness = 10.0, min_sel = 6, mtk_max = 130.0)
    ids = unique(subset(bf, :minutes_to_kickoff => ByRow(x -> 0 < x <= mtk_max)).match_id)
    parts = Vector{DataFrame}(undef, length(ids))
    Threads.@threads for k in eachindex(ids)
        local tr
        try
            tr = inplay_lambda_trace(bf, ds, Int(ids[k]), config;
                                     bin_minutes = bin_minutes, staleness = staleness,
                                     min_sel = min_sel, mtk_max = mtk_max)
        catch
            tr = DataFrame()
        end
        parts[k] = tr
    end
    leftjoin(vcat([df for df in parts if nrow(df) > 0]...), pg_tbl, on = :match_id)
end

panel = build_panel(bf, ds, pg_tbl; bin_minutes = 5.0)
inp   = build_intensity_inputs(panel, ds)
println("[INFO] inputs: $(length(inp.y)) side-bins, $(inp.n_teams) teams, " *
        "$(length(unique(inp.match_id))) matches")

# By-match train/test split.
ms  = shuffle(MersenneTwister(1), unique(inp.match_id))
cut = round(Int, 0.75 * length(ms))
inp_tr = subset_inputs(inp, Set(ms[1:cut]))
inp_te = subset_inputs(inp, Set(ms[cut+1:end]))

# ==========================================================================
# 2. CONFIG GRID  (build up the team effects)
# ==========================================================================
configs = Dict(
    "baseline (global only)"   => InPlayIntensityConfig(),
    "+attack"                  => InPlayIntensityConfig(use_team_attack = true),
    "+attack+defense"          => InPlayIntensityConfig(use_team_attack = true, use_team_defense = true),
    "+trailing+leading"        => InPlayIntensityConfig(use_team_trailing = true, use_team_leading = true),
    "full"                     => InPlayIntensityConfig(use_team_attack = true, use_team_defense = true,
                                                        use_team_trailing = true, use_team_leading = true),
)

nuts = Samplers.NUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4, show_progress = :perchain)

results = DataFrame(config = String[], elpd_per_obs = Float64[], max_rhat = Float64[])
chains  = Dict{String,Any}()
for (name, cfg) in configs
    println("\n[SAMPLING] $name ...")
    ch = Samplers.run_sampler(make_model(inp_tr, cfg), nuts)
    chains[name] = (ch, cfg)
    elpd = held_out_elpd(ch, inp_te, cfg).per_obs
    rhat = maximum(skipmissing(summarystats(ch).nt.rhat))
    push!(results, (name, round(elpd, digits = 4), round(rhat, digits = 3)))
end
sort!(results, :elpd_per_obs, rev = true)
println("\n[HELD-OUT elpd per obs by config (higher = better)]"); show(results, allrows = true); println()

# ==========================================================================
# 3. WINNING-CONFIG POSTERIOR SUMMARIES
# ==========================================================================
best_name = results.config[1]
best_ch, best_cfg = chains[best_name]
println("\n[BEST CONFIG: $best_name] global coefficients:")
show(summarystats(best_ch)); println()

"Team delta posteriors for a hierarchical effect (e.g. :z_ld + :σ_ld) → per-team mean/CI."
function team_effect_table(chain, inp, zbase::Symbol, σname::Symbol)
    σ = _chainvec(chain, σname)
    rows = NamedTuple[]
    for i in 1:inp.n_teams
        d = _chainvec(chain, Symbol("$zbase[$i]")) .* σ
        push!(rows, (team = inp.team_names[i], mean = round(mean(d), digits = 3),
                     lo = round(quantile(d, 0.1), digits = 3), hi = round(quantile(d, 0.9), digits = 3)))
    end
    sort!(DataFrame(rows), :mean)
end

# Example: who parks the bus (δ_lead) / chases hardest (δ_trail), if those effects are in the best config.
if best_cfg.use_team_leading
    println("\n[δ_lead by team — most negative = shuts up shop hardest when winning]")
    show(team_effect_table(best_ch, inp, :z_ld, :σ_ld), allrows = true); println()
end
if best_cfg.use_team_trailing
    println("\n[δ_trail by team — most positive = chases hardest when losing]")
    show(team_effect_table(best_ch, inp, :z_tr, :σ_tr), allrows = true); println()
end
