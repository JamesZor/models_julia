# current_development/scottish_lower/portfolio/r05_eval_layer2_inverse_shift.jl
#
# Compare the Scottish Lower pxG + wealth recombination champion before and after
# the generative Layer-2 inverse market shift. Run with multiple Julia threads:
#
#   julia --project -t auto
#   include("current_development/scottish_lower/portfolio/r05_eval_layer2_inverse_shift.jl")

using BayesianFootball
using DataFrames, Statistics, Optim, Serialization

const PF       = BayesianFootball.Portfolio
const DD       = BayesianFootball.Data
const EE       = BayesianFootball.Experiments
const Features = BayesianFootball.Features
const ROOT     = pkgdir(BayesianFootball)

# Loads TeamPxGRecombWealthIntegratedModel and its model-specific prediction methods.
include(joinpath(ROOT, "current_development/scottish_lower/open_play/l05_recomb_pxg_models.jl"))

const CHAMPION_NAME = "recomb_pxg_wealth_integrated_hl365_hs2"
const W_BASE = 0.25
const L2_SIGMA = 0.25

# -----------------------------------------------------------------------------
# Generative Layer-2 inverse shift (from rqs_001_multi_class_softmax_pooling.jl)
# -----------------------------------------------------------------------------
function inverse_dynamic_weight_log(rate_model::Float64, rate_mkt::Float64;
                                    w_base::Float64 = W_BASE,
                                    sigma::Float64 = L2_SIGMA)
    r_mod = max(rate_model, 1e-6)
    r_mkt = max(rate_mkt, 1e-6)
    delta_sq = (log(r_mod) - log(r_mkt))^2
    w_dynamic = 1.0 - exp(-delta_sq / (2.0 * sigma^2))
    return w_base + (1.0 - w_base) * w_dynamic
end

"""
    apply_layer2_shift!(latents_df, odds_df; w_base=0.25, sigma=0.25)

Fit market-implied home/away Poisson intensities from fair Betfair closing
probabilities, then geometrically pool every posterior intensity draw with the
market intensity. Adds `shifted_λ_h`, `shifted_λ_a`, `l2_w_h`, and `l2_w_a`.
Matches without market targets remain unshifted.
"""
function apply_layer2_shift!(latents_df::DataFrame, odds_df::DataFrame;
                             w_base::Float64 = W_BASE,
                             sigma::Float64 = L2_SIGMA)
    config = Features.DoublePoissonMarketFeature()
    init_guess = Features.get_initial_guess(config)
    N = nrow(latents_df)

    odds_by_match = Dict{eltype(latents_df.match_id), Dict{Symbol, Float64}}()
    for r in eachrow(odds_df)
        targets = get!(odds_by_match, r.match_id) do
            Dict{Symbol, Float64}()
        end
        targets[Symbol(r.selection)] = Float64(r.prob_fair_close)
    end

    match_ids = latents_df.match_id
    λ_h_vec = latents_df.λ_h
    λ_a_vec = latents_df.λ_a

    shifted_λ_h = Vector{Vector{Float64}}(undef, N)
    shifted_λ_a = Vector{Vector{Float64}}(undef, N)
    l2_w_h = zeros(Float64, N)
    l2_w_a = zeros(Float64, N)

    Threads.@threads for i in 1:N
        targets = get(odds_by_match, match_ids[i], nothing)
        raw_λ_h = λ_h_vec[i]
        raw_λ_a = λ_a_vec[i]

        if targets === nothing || isempty(targets)
            shifted_λ_h[i] = copy(raw_λ_h)
            shifted_λ_a[i] = copy(raw_λ_a)
            l2_w_h[i] = 1.0
            l2_w_a[i] = 1.0
            continue
        end

        loss = let cfg = config, tgts = targets
            θ -> begin
                P = Features.build_probability_matrix(cfg, θ, 10)
                sse = Features._calculate_error(Val(:result_1x2), P, tgts)
                sse += Features._calculate_error(Val(:uo), P, tgts; min_k = 1, max_k = 4)
                return sse
            end
        end

        result = optimize(loss, init_guess, NelderMead())
        mkt_params = Features.extract_parameters(config, Optim.minimizer(result))

        w_h = inverse_dynamic_weight_log(
            median(raw_λ_h), mkt_params.λ_home; w_base = w_base, sigma = sigma)
        w_a = inverse_dynamic_weight_log(
            median(raw_λ_a), mkt_params.λ_away; w_base = w_base, sigma = sigma)

        mult_h = mkt_params.λ_home ^ (1.0 - w_h)
        mult_a = mkt_params.λ_away ^ (1.0 - w_a)
        shifted_λ_h[i] = @. (raw_λ_h ^ w_h) * mult_h
        shifted_λ_a[i] = @. (raw_λ_a ^ w_a) * mult_a
        l2_w_h[i] = w_h
        l2_w_a[i] = w_a
    end

    latents_df.shifted_λ_h = shifted_λ_h
    latents_df.shifted_λ_a = shifted_λ_a
    latents_df.l2_w_h = l2_w_h
    latents_df.l2_w_a = l2_w_a
    return latents_df
end

# The recombination adapter prices from open-play + noise components rather than
# directly from λ_h/λ_a. Scale both components proportionally so their sum is the
# shifted total intensity while preserving the model's posterior decomposition.
function install_shifted_intensities!(df::DataFrame)
    raw_h = copy(df.λ_h)
    raw_a = copy(df.λ_a)

    if :lambda_open_h in propertynames(df) && :lambda_open_a in propertynames(df)
        noise_h = if :lambda_noise_h in propertynames(df)
            copy(df.lambda_noise_h)
        elseif :lambda_pen_h in propertynames(df)
            [(0.768 .* x) .+ 0.0276 for x in df.lambda_pen_h]
        else
            [max.(raw_h[i] .- df.lambda_open_h[i], 0.0) for i in eachindex(raw_h)]
        end
        noise_a = if :lambda_noise_a in propertynames(df)
            copy(df.lambda_noise_a)
        elseif :lambda_pen_a in propertynames(df)
            [(0.768 .* x) .+ 0.0276 for x in df.lambda_pen_a]
        else
            [max.(raw_a[i] .- df.lambda_open_a[i], 0.0) for i in eachindex(raw_a)]
        end

        scale_h = [df.shifted_λ_h[i] ./ max.(raw_h[i], 1e-9) for i in eachindex(raw_h)]
        scale_a = [df.shifted_λ_a[i] ./ max.(raw_a[i], 1e-9) for i in eachindex(raw_a)]
        df.lambda_open_h = [df.lambda_open_h[i] .* scale_h[i] for i in eachindex(raw_h)]
        df.lambda_open_a = [df.lambda_open_a[i] .* scale_a[i] for i in eachindex(raw_a)]
        df.lambda_noise_h = [noise_h[i] .* scale_h[i] for i in eachindex(raw_h)]
        df.lambda_noise_a = [noise_a[i] .* scale_a[i] for i in eachindex(raw_a)]
    end

    df.λ_h = df.shifted_λ_h
    df.λ_a = df.shifted_λ_a
    return df
end

println("\n", "="^100)
println("SCOTTISH LOWER CHAMPION: RAW vs LAYER-2 INVERSE MARKET SHIFT")
println("="^100)

@info "Loading Scottish Lower datastore"
ds = DD.load_datastore_cached(DD.ScottishLower(), max_age_hours = 720)

cache_dir = joinpath(@__DIR__, "cache")
mkpath(cache_dir)
odds_cache = joinpath(cache_dir, "betfair_summary_odds.jls")
odds = if isfile(odds_cache)
    deserialize(odds_cache)
else
    x = DD.summarize_betfair_market(
        ds; open_window = (-100000.0, -10.0), close_window = (-20.0, 0.0))
    serialize(odds_cache, x)
    x
end
@info "Loaded Betfair closing odds" matches = length(unique(odds.match_id)) quotes = nrow(odds)

folders = EE.list_experiments("scottish_open_play_grid"; data_dir = joinpath(ROOT, "data"))
loaded = EE.load_experiments(folders)
candidates = filter(exp -> startswith(exp.config.name, CHAMPION_NAME), loaded)
isempty(candidates) && error("Champion experiment '$CHAMPION_NAME' was not found")
expr_champ = first(candidates)
@info "Loaded champion experiment" name = expr_champ.config.name

raw_df = copy(EE.extract_oos_predictions(ds, expr_champ).df)
@info "Extracted all champion OOS latents" matches = nrow(raw_df)

shifted_df = copy(raw_df)
@info "Applying Layer-2 inverse shift" w_base = W_BASE sigma = L2_SIGMA threads = Threads.nthreads()
apply_layer2_shift!(shifted_df, odds; w_base = W_BASE, sigma = L2_SIGMA)
install_shifted_intensities!(shifted_df)
@info "Layer-2 shift complete" median_w_h = median(shifted_df.l2_w_h) median_w_a = median(shifted_df.l2_w_a)

MARKETS = DD.MarketConfig(reduce(vcat, (
    DD.AbstractMarket[DD.Market1X2(), DD.MarketBTTS()],
    [DD.MarketOverUnder(i + 0.5) for i in 0:4],
)))

book_spec = PF.BookSpec(
    markets = MARKETS,
    allocator = PF.KellyLogUtility(),
    shrink = PF.BakerMcHale(n_draws = 800),
)

@info "Building unshifted raw books"
raw_books = PF.build_books(book_spec, raw_df, expr_champ, odds, ds)
@info "Building Layer-2 shifted books"
shifted_books = PF.build_books(book_spec, shifted_df, expr_champ, odds, ds)

books = Dict("Raw" => raw_books, "Layer2 Shifted" => shifted_books)
policies = [
    ("Conservative (Cap 10%, λ=23)", PF.PolicySpec(
        trust = PF.FlatTrust(0.25), risk = PF.SlateDrawdown(23.0),
        cap = PF.FixedCap(0.10), filter = PF.KeepAll(), grouping = PF.DailySlate())),
    ("Balanced (Cap 15%, λ=15)", PF.PolicySpec(
        trust = PF.FlatTrust(0.25), risk = PF.SlateDrawdown(15.0),
        cap = PF.FixedCap(0.15), filter = PF.KeepAll(), grouping = PF.DailySlate())),
    ("Aggressive (Cap 20%, λ=10)", PF.PolicySpec(
        trust = PF.FlatTrust(0.50), risk = PF.SlateDrawdown(10.0),
        cap = PF.FixedCap(0.20), filter = PF.KeepAll(), grouping = PF.DailySlate())),
]

leaderboard = DataFrame(
    policy = String[], variant = String[], final_wealth = Float64[],
    net_roi_pct = Float64[], sharpe = Float64[], max_drawdown_pct = Float64[],
    total_bets = Int[],
)

for (policy_name, policy) in policies
    for variant in ("Raw", "Layer2 Shifted")
        slates = PF.group(PF.DailySlate(), books[variant])
        traj = PF.simulate(policy, slates; use_shrink = true)
        m = PF.path_metrics(traj)
        returns = traj.slate_pl
        sharpe = length(returns) > 1 && std(returns) > 1e-6 ?
            mean(returns) / std(returns) * sqrt(35) : 0.0
        push!(leaderboard, (
            policy_name, variant, round(m.final, digits = 3),
            round(m.roi, digits = 2), round(sharpe, digits = 2),
            round(m.mdd, digits = 2), m.n_bets,
        ))
    end
end

sort!(leaderboard, [:policy, order(:final_wealth, rev = true)])
println("\n", "="^100)
println("RAW vs LAYER-2 COMPARISON LEADERBOARD")
println("="^100)
show(stdout, MIME("text/plain"), leaderboard; allrows = true, allcols = true)
println("\n", "="^100)
