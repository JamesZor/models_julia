#=
r03 — SMOKE runner for the Stage-1 funnel cascade (l03) on ScottishLower (56/57).

    Shots ~ Poisson(λ_s) ; SoT|Shots ~ Bin(p₁) ; Goals|SoT ~ Bin(p₂)
    ⇒ goals marginal Poisson(λ_s·p₁·p₂) — pricing unchanged, observation model ~7× richer.

Mirrors scottish_lower_smile/r01_smoke.jl. Trains the funnel AND the none_pois comparator
(TeamDPGoalsModel) on IDENTICAL folds so the eval tables are directly comparable.

Checks:
  0a. bbc count sanity vs the 2026-07-17 EDA (coverage ≥95%, shots ≈10.2/9.1, SoT ≈4.46/4.04,
      sot≤shots 100%, goals>sot ≈1%). A miss ⇒ `shotsTotal` is not the column those numbers
      came from — STOP before training.
  0b. compile probe: one fold, 1 chain, compiled ReverseDiff — no -Inf flood / AD error,
      p₁≈0.44, p₂≈0.32.
  1.  both engines train (items > 0 — silent split drops are a known failure mode).
  2.  convergence: global max R-hat ≤ 1.05; p1_raw / p2_raw / δ_league_raw ≤ 1.01.
  3.  posterior sanity: λ_s ≈ 10, composite λ_goals ≈ 1.38.
  4.  SPREAD DIAGNOSTIC (the diagnosis, not a gate): across-team sd of implied log λ_goals,
      funnel vs none_pois. Materially narrower ⇒ global p₁/p₂ compressed team strength ⇒
      Stage 2 (hierarchical conversion, l04) is the indicated fix.
  5.  PPD + OOS eval vs the Bet365 fair close, both engines.

Stage-1 success bar is NOT "beats none_pois" — it is: converges, prices, and the spread
diagnostic + logloss delta together say which failure branch we are in.

Run on the server (kaimon REPL) after git pull + REPL restart (fresh structs):
    include(joinpath(pkgdir(BayesianFootball), "current_development/bbc_xg_proxy/r03_funnel_smoke.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using DynamicPPL, LogDensityProblems, ReverseDiff   # guide §8 gradient benchmark

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/bbc_xg_proxy/l03_funnel_cascade.jl"))

verdicts = Tuple{String, Bool}[]
_mark(name, ok) = (push!(verdicts, (name, ok)); println(ok ? "✅ $name" : "❌ $name"))
_r(x, d=3) = round(x, digits=d)

# ==========================================
# 1. DATA
# ==========================================
println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/funnel_smoke/")
mkpath(save_dir)

season_strings = sort(unique(String.(ds.matches.season)))
TARGET = season_strings[end]
println("[INFO] seasons: ", season_strings, " -> smoke target = ", TARGET)

# ==========================================
# CHECK 0a — bbc count sanity (EDA 2026-07-17)
# ==========================================
println("\n", "="^70, "\nCHECK 0a — bbc funnel counts vs the 2026-07-17 EDA\n", "="^70)

bbc = BBC_FUNNEL_DF[]
lut = BBC_FUNNEL[]

# coverage measured against the DataStore (what the model will actually ask for)
ds_ids   = Int.(ds.matches.match_id)
_usable(c, side) = (side == :h ? (c.shots_h >= 0 && c.sot_h >= 0 && c.sot_h <= c.shots_h) :
                                 (c.shots_a >= 0 && c.sot_a >= 0 && c.sot_a <= c.shots_a))
absent   = (shots_h = -1, shots_a = -1, sot_h = -1, sot_a = -1)
cov_h    = mean([_usable(get(lut, id, absent), :h) for id in ds_ids])
cov_a    = mean([_usable(get(lut, id, absent), :a) for id in ds_ids])
println("  DataStore matches: $(length(ds_ids));  usable-count coverage  home=$(_r(cov_h)) away=$(_r(cov_a))")
_mark("0a.1 coverage ≥ 0.95 both sides", cov_h >= 0.95 && cov_a >= 0.95)

ok_rows = bbc[.!ismissing.(bbc.shots_h) .& .!ismissing.(bbc.shots_a) .&
              .!ismissing.(bbc.sot_h)   .& .!ismissing.(bbc.sot_a), :]
lvl = combine(groupby(ok_rows, :tournament_id),
    nrow => :n,
    :shots_h => (x -> _r(mean(x), 2)) => :shots_h,
    :shots_a => (x -> _r(mean(x), 2)) => :shots_a,
    :sot_h   => (x -> _r(mean(x), 2)) => :sot_h,
    :sot_a   => (x -> _r(mean(x), 2)) => :sot_a,
    :home_score => (x -> _r(mean(x), 2)) => :goals_h,
    :away_score => (x -> _r(mean(x), 2)) => :goals_a)
show(lvl; allrows=true, allcols=true); println()

sh, sa = mean(ok_rows.shots_h), mean(ok_rows.shots_a)
th, ta = mean(ok_rows.sot_h),   mean(ok_rows.sot_a)
_mark("0a.2 shot levels ≈ 10.2/9.1 (got $(_r(sh,2))/$(_r(sa,2)))",
      8.5 <= sh <= 12.0 && 7.5 <= sa <= 11.0)
_mark("0a.3 SoT levels ≈ 4.46/4.04 (got $(_r(th,2))/$(_r(ta,2)))",
      3.5 <= th <= 5.5 && 3.2 <= ta <= 5.0)

p1_emp = (th + ta) / (sh + sa)
p2_emp = (mean(ok_rows.home_score) + mean(ok_rows.away_score)) / (th + ta)
println("  empirical p₁ = $(_r(p1_emp)) (prior centre 0.44),  p₂ = $(_r(p2_emp)) (prior centre 0.32)")
_mark("0a.4 empirical p₁/p₂ within 0.06 of the priors",
      abs(p1_emp - 0.44) < 0.06 && abs(p2_emp - 0.32) < 0.06)

viol_sot   = mean(vcat(ok_rows.sot_h .> ok_rows.shots_h, ok_rows.sot_a .> ok_rows.shots_a))
viol_goals = mean(vcat(ok_rows.home_score .> ok_rows.sot_h, ok_rows.away_score .> ok_rows.sot_a))
println("  cascade violations: sot>shots $(_r(viol_sot,4)) | goals>sot $(_r(viol_goals,4)) " *
        "(the latter ≈1% own goals — handled by the marginal-Poisson fallback)")
_mark("0a.5 sot > shots never happens", viol_sot == 0.0)
_mark("0a.6 goals > sot ≈ 1% (< 5%)", viol_goals < 0.05)

# ==========================================
# 2. MODELS (hl365 = Stage-A winner, identical for both engines)
# ==========================================
dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = 365.0)

m_funnel = TeamFunnelDPGoalsModel(dynamics_config = dyn_cfg)
m_dp     = TeamDPGoalsModel(dynamics_config = dyn_cfg)

specs = Tuple{String, Any}[
    ("funnel_pois_smoke", m_funnel),
    ("none_pois_smoke",   m_dp),
]

_mark("0b.0 funnel declares BBCFunnelFeature",
      any(f -> f isa BBCFunnelFeature, Features.required_features(m_funnel)))

# ==========================================
# CHECK 0b — COMPILE PROBE (one fold, 1 chain, compiled ReverseDiff)
# ==========================================
println("\n", "="^70, "\nCHECK 0b — compile probe\n", "="^70)
probe_ch = nothing
try
    cv = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons    = [TARGET],
        history_seasons   = 2,
        dynamics_col      = :match_biweek,
        warmup_period     = 16,
    )
    boundary = last(Data.create_id_boundaries(ds, cv))[1]
    fs = Features.create_features(boundary, ds, m_funnel, :match_biweek)

    # feature-side sanity before we burn MCMC time on it
    d = _unpack_funnel(fs.data, m_funnel)
    println("  fold rows=$(length(d.home_goals))  stats_mask h/a = " *
            "$(_r(mean(d.stats_mask_h)))/$(_r(mean(d.stats_mask_a)))  " *
            "casc_mask h/a = $(_r(mean(d.casc_mask_h)))/$(_r(mean(d.casc_mask_a)))")
    safe = all(d.sot_h_s .<= d.shots_h_s) && all(d.sot_a_s .<= d.shots_a_s) &&
           all(d.goals_h_c .<= d.sot_h_c) && all(d.goals_a_c .<= d.sot_a_c)
    _mark("0b.1 safe dummies hold (no k>n anywhere ⇒ no -Inf×0 NaN)", safe)

    # ---- 0b-bis: the sufficient-statistic likelihood is EXACT ----------------------------
    # The builder folds counts/masks/decay weights into constants and drops log(y!) and the
    # binomial coefficients (parameter-free, and the cascade/marginal routing is fixed by
    # data). So naive − sufficient must be the SAME constant at every parameter draw; the
    # variance of that difference across draws is the test.
    function _naive_ll(d, log_λ_h, log_λ_a, p1_raw, p2_raw, fw)
        p1, p2 = 1 / (1 + exp(-p1_raw)), 1 / (1 + exp(-p2_raw))
        tot = 0.0
        for (S, log_λ, shots, sot, sm, sot_c, goals_c, cm, goals) in (
            (:h, log_λ_h, d.shots_h_s, d.sot_h_s, d.stats_mask_h, d.sot_h_c, d.goals_h_c,
             d.casc_mask_h, d.home_goals),
            (:a, log_λ_a, d.shots_a_s, d.sot_a_s, d.stats_mask_a, d.sot_a_c, d.goals_a_c,
             d.casc_mask_a, d.away_goals))
            λ = exp.(log_λ)
            ll = fw .* (logpdf.(Poisson.(λ), shots) .* sm .+
                        logpdf.(Binomial.(shots, p1), sot) .* sm) .+
                 cm .* logpdf.(Binomial.(sot_c, p2), goals_c) .+
                 (1 .- cm) .* logpdf.(Poisson.(λ .* p1 .* p2), goals)
            tot += sum(ll .* d.match_weights)
        end
        return tot
    end
    function _suff_ll(d, log_λ_h, log_λ_a, p1_raw, p2_raw, fw)
        lp1, lq1 = -log1pexp(-p1_raw), -log1pexp(p1_raw)
        lp2, lq2 = -log1pexp(-p2_raw), -log1pexp(p2_raw)
        tot = 0.0
        for (S, log_λ) in ((d.suff_h, log_λ_h), (d.suff_a, log_λ_a))
            λ = exp.(log_λ)
            tot += fw * (sum(S.c_shots_lin .* log_λ) - sum(S.c_shots_rate .* λ) +
                         S.S_sot * lp1 + S.S_miss * lq1) +
                   S.S_goal * lp2 + S.S_save * lq2 +
                   sum(S.c_marg_lin .* log_λ) + S.S_marg_goals * (lp1 + lp2) -
                   exp(lp1 + lp2) * sum(S.c_marg_rate .* λ)
        end
        return tot
    end
    n_rows = length(d.home_goals)
    diffs = Float64[]
    for _ in 1:6
        lh = m_funnel.shot_scale .+ 0.3 .* randn(n_rows)
        la = m_funnel.shot_scale .+ 0.3 .* randn(n_rows)
        pr1, pr2 = randn(), randn()
        push!(diffs, _naive_ll(d, lh, la, pr1, pr2, m_funnel.funnel_weight) -
                     _suff_ll(d, lh, la, pr1, pr2, m_funnel.funnel_weight))
    end
    spread = maximum(diffs) - minimum(diffs)
    println("  naive − sufficient: const ≈ $(_r(mean(diffs), 2)), spread over 6 draws = " *
            "$(round(spread, sigdigits=3))  (must be ~0 — a pure data-only offset)")
    _mark("0b.1b sufficient-statistic likelihood is exact (spread < 1e-6)", spread < 1e-6)

    tm = PreGame.build_turing_model(m_funnel, fs)

    # ---- 0b-ter: gradient benchmark (docs/turing_ad_performance_guide.md §8) -------------
    try
        vi = DynamicPPL.VarInfo(tm); tm(vi); θ = vi[:]
        lf = DynamicPPL.LogDensityFunction(tm)
        tape = ReverseDiff.compile(ReverseDiff.GradientTape(x -> LogDensityProblems.logdensity(lf, x), θ))
        g = similar(θ)
        t_grad = minimum(@elapsed(ReverseDiff.gradient!(g, tape, θ)) for _ in 1:20)
        println("  gradient eval = $(round(t_grad * 1000, digits=3)) ms over $(length(θ)) params " *
                "(guide target < 1 ms)")
        _mark("0b.1c gradient eval < 1 ms", t_grad < 1e-3)
    catch e
        _mark("0b.1c gradient benchmark ran", false)
        @error "gradient benchmark failed" exception=(e, catch_backtrace())
    end

    global probe_ch = sample(tm, NUTS(200, 0.65; max_depth = 8,
                                      adtype = AutoReverseDiff(compile = true)), 200)
    lp = vec(Array(probe_ch[:lp]))
    _mark("0b.2 probe chain finite (no -Inf/NaN flood)", all(isfinite, lp))

    p1p = 1 ./ (1 .+ exp.(-vec(Array(probe_ch[:p1_raw]))))
    p2p = 1 ./ (1 .+ exp.(-vec(Array(probe_ch[:p2_raw]))))
    println("  probe p₁ = $(_r(mean(p1p)))  p₂ = $(_r(mean(p2p)))  " *
            "μ_base[end] = $(_r(mean(vec(Array(probe_ch[Symbol("inter.μ_base[$(Int(fs.data[:n_seasons]))]")]))), 3))")
    _mark("0b.3 probe p₁ ≈ 0.44 (±0.08)", abs(mean(p1p) - 0.44) < 0.08)
    _mark("0b.4 probe p₂ ≈ 0.32 (±0.08)", abs(mean(p2p) - 0.32) < 0.08)
catch e
    _mark("0b. compile probe ran", false)
    @error "compile probe FAILED" exception=(e, catch_backtrace())
end

# ==========================================
# 3. TRAIN — both engines, identical folds
# ==========================================
println("\n", "="^70, "\nTRAIN — funnel + none_pois, target=$TARGET\n", "="^70)
runs = Tuple{String, Any, Any}[]
for (name, model) in specs
    println("\n", "#"^68, "\n# RUN: $name\n", "#"^68)
    t0 = time()
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = [TARGET],
            history_seasons = 2,
            warmup_period   = 16,
            dynamics_col    = :match_biweek,
            samples         = 600,
            warmup          = 600,
            chains          = 4,
            use_queue       = true,
            # depth 8 caps a leapfrog at 255 steps. The first attempt at depth 10 with the
            # old parameterisation crushed ε to ~4e-4, every iteration maxed the tree, and
            # 0/20 chains finished in 4.5 h. The shot_scale offset fixes the init; the cap
            # bounds the worst case (smile-stream lesson: caps are safe when σ is sampled).
            max_depth       = 8,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        push!(runs, (name, model, res))
        n_items = length(res.training_results.items)
        println("  wall = $(_r((time() - t0) / 60, 1)) min, items = $n_items")
        _mark("1. $name trained (items=$n_items > 0, no silent drop)", n_items > 0)
    catch e
        _mark("1. $name trained", false)
        @error "FAILED: $name" exception=(e, catch_backtrace())
    end
end

# ==========================================
# CHECK 2 — convergence
# ==========================================
println("\n", "="^70, "\nCHECK 2 — convergence\n", "="^70)
for (name, model, res) in runs
    try
        chains_obj = Experiments.Diagnostics.extract_chains(ds, res)
        conv  = Experiments.Diagnostics.check_convergence(chains_obj)
        worst = isempty(conv.df) ? NaN : maximum(skipmissing(conv.df.rhat))
        _mark("2a. $name global max R-hat ≤ 1.05 (=$(_r(worst, 4)))", worst <= 1.05)

        raw = res.training_results.items[1][1]
        er  = DataFrame(MCMCChains.ess_rhat(raw))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        _rhat(p) = (rows = er[er.parameters .== Symbol(p), rcol]; isempty(rows) ? NaN : rows[1])
        new_params = ["δ_league_raw[1]", "δ_league_raw[2]"]
        model isa TeamFunnelDPGoalsModel && append!(new_params, ["p1_raw", "p2_raw"])
        rhats = [(p, _rhat(p)) for p in new_params]
        for (p, r) in rhats; println("    $(rpad(p, 18)) rhat=$(_r(r, 4))"); end
        _mark("2b. $name new-param R-hat ≤ 1.01", all(!isnan(r) && r <= 1.01 for (_, r) in rhats))
    catch e
        _mark("2. $name convergence check ran", false)
        @error "convergence check failed: $name" exception=(e, catch_backtrace())
    end
end

# ==========================================
# CHECK 3 — posterior sanity (funnel)
# ==========================================
println("\n", "="^70, "\nCHECK 3 — funnel posterior sanity\n", "="^70)
function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    return out
end
funnel_run = findfirst(r -> r[2] isa TeamFunnelDPGoalsModel, runs)
if funnel_run !== nothing
    _, _, fres = runs[funnel_run]
    p1 = 1 ./ (1 .+ exp.(-_pool(fres, "p1_raw")))
    p2 = 1 ./ (1 .+ exp.(-_pool(fres, "p2_raw")))
    println("  p₁ mean=$(_r(mean(p1))) 90%=[$(_r(quantile(p1, .05))), $(_r(quantile(p1, .95)))]")
    println("  p₂ mean=$(_r(mean(p2))) 90%=[$(_r(quantile(p2, .05))), $(_r(quantile(p2, .95)))]")
    _mark("3a. p₁ ≈ 0.44 (±0.06)", abs(mean(p1) - 0.44) < 0.06)
    _mark("3b. p₂ ≈ 0.32 (±0.06)", abs(mean(p2) - 0.32) < 0.06)

    d1 = _pool(fres, "δ_league_raw[1]"); d2 = _pool(fres, "δ_league_raw[2]")
    isempty(d1) || println("  δ₅₆−δ₅₇ (shot scale) mean=$(_r(mean(d1 .- d2), 4))")
end

# ==========================================
# CHECK 4 — OOS latents: level + SPREAD DIAGNOSTIC
# ==========================================
println("\n", "="^70, "\nCHECK 4 — λ levels + team-strength spread (funnel vs none_pois)\n", "="^70)
team_of = Dict(Int(r.match_id) => (String(r.home_team), String(r.away_team))
               for r in eachrow(ds.matches))
latents = Dict{String, Any}()
spreads = Dict{String, Float64}()

for (name, model, res) in runs
    try
        L = Experiments.extract_oos_predictions(ds, res)
        latents[name] = L
        df = L.df

        λh = [mean(r.λ_h) for r in eachrow(df)]
        λa = [mean(r.λ_a) for r in eachrow(df)]
        println("  $name: mean λ_goals h/a = $(_r(mean(λh)))/$(_r(mean(λa)))  (n=$(nrow(df)))")
        if model isa TeamFunnelDPGoalsModel
            λs = [mean(r.λ_s_h) for r in eachrow(df)]
            println("           mean λ_shots h = $(_r(mean(λs), 2))  (EDA 10.2)")
            _mark("4a. funnel λ_shots ≈ 10 (8–13)", 8.0 <= mean(λs) <= 13.0)
            _mark("4b. funnel λ_goals ≈ 1.38 (1.0–1.9)", 1.0 <= mean(λh) <= 1.9)
        end

        # spread = across-team sd of the posterior-mean home log λ_goals
        by_team = Dict{String, Vector{Float64}}()
        for (i, r) in enumerate(eachrow(df))
            t = get(team_of, Int(r.match_id), ("?", "?"))[1]
            push!(get!(by_team, t, Float64[]), log(λh[i]))
        end
        per_team = [mean(v) for (_, v) in by_team if length(v) >= 2]
        spreads[name] = length(per_team) >= 3 ? std(per_team) : NaN
        println("           across-team sd of log λ_goals(home) = $(_r(spreads[name], 4)) " *
                "over $(length(per_team)) teams")
    catch e
        _mark("4. $name OOS latents extracted", false)
        @error "extract_oos_predictions failed: $name" exception=(e, catch_backtrace())
    end
end

if haskey(spreads, "funnel_pois_smoke") && haskey(spreads, "none_pois_smoke")
    sf, sd0 = spreads["funnel_pois_smoke"], spreads["none_pois_smoke"]
    ratio = sf / sd0
    println("\n  >> SPREAD RATIO funnel/none_pois = $(_r(ratio)) " *
            "(≪1 ⇒ global p₁/p₂ compressed team strength ⇒ Stage 2 hierarchical conversion)")
    _mark("4c. spread diagnostic computed (informational, no gate)", isfinite(ratio))
end

# ==========================================
# CHECK 5 — PPD + OOS eval vs Bet365 fair close
# ==========================================
println("\n", "="^70, "\nCHECK 5 — PPD + eval vs close\n", "="^70)
for (name, model, res) in runs
    try
        ppd = Predictions.model_inference(ds, res)
        _mark("5a. $name PPD runs, non-empty (n=$(nrow(ppd)))",
              ppd isa Predictions.PPD && nrow(ppd) > 0)
    catch e
        _mark("5a. $name PPD runs", false)
        @error "PPD failed: $name" exception=(e, catch_backtrace())
    end
end

selections = [
    :home, :draw, :away,
    :btts_yes, :btts_no,
    :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
    :over_35, :under_35, :over_45, :under_45,
]
R03_EVAL = nothing
try
    metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
    append!(metric, [Evaluation.LogLoss(s) for s in selections])
    append!(metric, [Evaluation.GLMEdge(s) for s in selections])
    global R03_EVAL = Evaluation.evaluate_experiments(metric, [r[3] for r in runs], ds)

    present = sort(unique(R03_EVAL.model))
    function _col(df, model, colname)
        colname in names(df) || return NaN
        r = df[df.model .== model, colname]
        (isempty(r) || ismissing(r[1])) ? NaN : round(Float64(r[1]), digits=4)
    end

    println("\n", "="^70,
            "\n📉 LogLoss diff (model−market) per line — NEGATIVE = beats Bet365 close\n", "="^70)
    ll = DataFrame(model = present)
    for s in selections
        ll[!, s] = [_col(R03_EVAL, m, "logloss_$(s)_overall_diff_ll") for m in present]
    end
    show(ll; allrows=true, allcols=true, truncate=0); println()

    fam = Dict(
        :x12    => [:home, :draw, :away],
        :btts   => [:btts_yes, :btts_no],
        :totals => [:over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
                    :over_35, :under_35, :over_45, :under_45],
    )
    println("\n", "="^70, "\n📊 Family-pooled mean LogLoss diff (Stage-A ref: x12 0.0143, " *
            "BTTS 0.0014, totals 0.0002)\n", "="^70)
    fm = DataFrame(model = present)
    for (fname, sels) in fam
        fm[!, fname] = [round(mean(filter(!isnan,
            [_col(R03_EVAL, m, "logloss_$(s)_overall_diff_ll") for s in sels])), digits=4)
            for m in present]
    end
    show(fm; allrows=true, allcols=true, truncate=0); println()

    if nrow(fm) == 2
        fr = findfirst(occursin.("funnel", fm.model)); nr = findfirst(occursin.("none", fm.model))
        if fr !== nothing && nr !== nothing
            println("\n  >> funnel − none_pois delta (negative = funnel better):")
            for f in keys(fam)
                println("     $(rpad(String(f), 8)) $(_r(fm[fr, f] - fm[nr, f], 4))")
            end
        end
    end
    _mark("5b. eval table produced for both engines", length(present) == length(runs))
catch e
    _mark("5b. eval table produced for both engines", false)
    @error "evaluate_experiments failed" exception=(e, catch_backtrace())
end

# ==========================================
# SUMMARY
# ==========================================
println("\n", "="^70, "\nR03 FUNNEL SMOKE SUMMARY\n", "="^70)
for (name, ok) in verdicts; println(ok ? "✅ $name" : "❌ $name"); end
n_pass = count(last, verdicts)
println("\n$(n_pass)/$(length(verdicts)) checks passed.")
println(n_pass == length(verdicts) ?
    ">> SMOKE PASSED: read the spread ratio + logloss delta, then write Stage 2 (l04 hierarchical p)." :
    ">> SOME CHECKS FAILED — fix before Stage 2.")
