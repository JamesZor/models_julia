#=
r01 — SINGLE-SPLIT SHAKEDOWN: the three κ modes, in parallel, Ireland (79).

Goals (in order):
  1. CONVERGENCE — R-hat ≤ ~1.01 / healthy ESS for ALL modes, checked on the RAW chains for
     the new κ params (κ0, τ_net/τ_att/τ_def, δ_net/z_att/z_def). NB check_convergence's
     curated conv.df DROPS params it doesn't know — do NOT rely on it for the new latents
     (the r17/r18 lesson).
  2. WHAT DID κ LEARN — per-team att/def multipliers (mean ± sd), team spread, and for
     :attdef the att-def correlation. τ pulled to ~0 with multipliers ≈ 1.00 = "learned
     nothing" (σ-hierarchy-null pattern).
  3. V0 sanity — :attack_only must reproduce the known HierarchicalTeamKappa behaviour
     (κ multipliers ~0.9–1.15, σ_κ small).

Execution: 3 variants × 4 chains = 12 concurrent chains via @sync/@spawn (r03 pattern;
each experiment's queue holds 4 items ⇒ 12 ≤ 16 pinned cores, no oversubscription).

Run on the server after git push → git pull → RESTART REPL:
    include("current_development/kappa_def/r01_single_split_shakedown.jl")
Flip SEGMENT to Data.IrelandFirstDivision() for the 718 follow-up (r02).
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

include("current_development/kappa_def/l01_kappa_def_models.jl")

# ==========================================
# 1. DATA (no market swap needed — market OFF)
# ==========================================
SEGMENT = Data.Ireland()
seg_tag = lowercase(string(nameof(typeof(SEGMENT))))
println("[INFO] Loading $(seg_tag) DataStore...")
ds = Data.load_datastore_cached(SEGMENT)

save_dir = "./data/kappa_def_dev_$(seg_tag)/"
mkpath(save_dir)

# ==========================================
# 2. SHARED CONFIG (r08/r03 conventions)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()          # carried, unused (Poisson)
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()       # V0 control only
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

_make(mode) = KappaDefDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    kappa_mode             = mode,
)

variants = [
    ("V0_attack_only", _make(:attack_only)),
    ("V2_net",         _make(:net)),
    ("V1_attdef",      _make(:attdef)),
]

function _build_task(model, name)
    Experiments.create_experiment_task(
        ds, model, name, save_dir;
        target_seasons  = ["2026"],
        history_seasons = 2,
        warmup_period   = 21,
        dynamics_col    = :match_week,
        samples         = 1000,
        warmup          = 500,
        chains          = 4,
        use_queue       = true,
        max_depth       = 10,
    )
end

# ==========================================
# 3. RUN ALL VARIANTS IN PARALLEL (12 chains ≤ 16 cores)
# ==========================================
println("\n>> Launching $(length(variants)) κ modes in parallel " *
        "($(length(variants))×4 = $(length(variants)*4) chains, $(Threads.nthreads()) threads)...")
raw_results = Dict{String, Any}()
rlock = ReentrantLock()
@sync for (name, model) in variants
    Threads.@spawn begin
        res = Experiments.run_experiment(_build_task(model, name))
        Experiments.save_experiment(res; quiet=true)
        lock(rlock) do
            raw_results[name] = res
        end
    end
end

# ==========================================
# 4. CONVERGENCE — curated banner + RAW κ-param diagnostics
# ==========================================
"raw-chain R-hat/ESS for parameter names matching any of `pats` (curated conv.df drops these)"
function raw_kappa_diag(ch, pats::Vector{String})
    s  = ess_rhat(ch)
    nm = string.(s.nt.parameters)
    keep = [any(occursin(p, n) for p in pats) for n in nm]
    DataFrame(parameter = nm[keep],
              rhat = round.(s.nt.rhat[keep], digits=4),
              ess  = round.(s.nt.ess[keep], digits=1))
end

KAPPA_PATS = ["κ0", "τ_net", "δ_net", "τ_att", "τ_def", "z_att", "z_def", "kap."]

"team index → name, recovered from the curated conv.df home_advantage rows (always present)"
function team_names_from_conv(conv_df, n_teams)
    ha_rows = conv_df[conv_df.parameter .== "home_advantage", [:raw_symbol, :entity]]
    names_ = fill("", n_teams)
    for r in eachrow(ha_rows)
        m = match(r"\[(\d+)\]", string(r.raw_symbol))
        m === nothing && continue
        i = parse(Int, m.captures[1])
        1 <= i <= n_teams && (names_[i] = string(r.entity))
    end
    any(isempty, names_) ? ["team_$i" for i in 1:n_teams] : names_
end

summaries = Dict{String, Any}()
for (name, model) in variants
    println("\n", "="^72, "\n>> MODE: $name\n", "="^72)
    res = raw_results[name]
    ch  = res.training_results.items[1][1]     # single split → one (chains-combined) Chains

    # curated banner (components) — informative but NOT sufficient for the new params
    chains_obj = Experiments.Diagnostics.extract_chains(ds, res)
    conv = Experiments.Diagnostics.check_convergence(chains_obj)
    println(conv)

    # n_teams from the FOLD's ha vector (chain-side truth; full-ds team count can mismatch)
    ha_idx = [match(r"\[(\d+)\]", string(s)) for s in conv.df.raw_symbol[conv.df.parameter .== "home_advantage"]]
    n_teams = maximum(parse(Int, m.captures[1]) for m in ha_idx if m !== nothing)

    # raw κ diagnostics — the real gate
    kd = raw_kappa_diag(ch, KAPPA_PATS)
    println("\n--- RAW κ-param diagnostics (the real convergence gate) ---")
    show(kd; allrows=true, allcols=true); println()
    max_rhat = isempty(kd) ? NaN : maximum(filter(!isnan, kd.rhat))
    println("max κ-param R-hat: $max_rhat  ", max_rhat <= 1.01 ? "✅" : (max_rhat <= 1.05 ? "⚠️ marginal" : "❌"))

    # what κ learned
    tnames = team_names_from_conv(conv.df, n_teams)
    tsum, glob = kappa_team_summary(model, ch, n_teams; team_names=tnames)
    println("\n--- per-team κ multipliers (goals-vs-xG conversion) ---")
    show(sort(tsum, :att_mult, rev=true); allrows=true, allcols=true); println()
    println("globals: κ0_conv=$(glob.κ0_conv)  att_spread=$(glob.att_spread)  " *
            "def_spread=$(glob.def_spread)  attdef_cor=$(glob.attdef_cor)")

    summaries[name] = (; max_rhat, glob)
end

# ==========================================
# 5. VERDICT TABLE
# ==========================================
println("\n", "█"^72, "\n  SHAKEDOWN SUMMARY ($seg_tag)\n", "█"^72)
for (name, _) in variants
    s = summaries[name]
    println(rpad(name, 16), " max_rhat=", rpad(s.max_rhat, 8),
            " κ0_conv=", rpad(s.glob.κ0_conv, 8),
            " att_spread=", rpad(s.glob.att_spread, 8),
            " def_spread=", rpad(s.glob.def_spread, 8),
            " attdef_cor=", s.glob.attdef_cor)
end
println("""

[READ]
 • Convergence gate: max κ-param R-hat ≤ 1.01 (from the RAW table, not the curated banner).
 • "Learned nothing" pattern: τ near 0, att/def_spread ≲ 0.02 (all multipliers ≈ 1.00) —
   the σ-hierarchy-null outcome; note it in EXPERIMENTS.md and don't bother evaluating.
 • V0 sanity: att_mult range should look like the familiar HierarchicalTeamKappa (~0.9–1.15).
 • :attdef vs :net — if att and def multipliers are strongly correlated across teams,
   the net (V2) parameterization captures it with half the params.
 • Cross-check the r00 persistence gate: def_spread here should only be trusted if the EDA
   said defensive residuals persist. Update EXPERIMENTS.md either way.
 • Next: flip SEGMENT to IrelandFirstDivision() (r02); then full-CV vs V0 judged vs the
   market (r03).
""")
