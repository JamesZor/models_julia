# current_development/scottish_lower_portfolio/r02_policy_sweep.jl
#
# Policy Grid Sweep on the Champion Scottish Model (`funnel_pxg_apm_hl365_hs2`).
#
# Sweeps across:
# - Trust (Model belief weight vs market): 0.10, 0.25, 0.50, 1.00
# - Risk Lambda (Drawdown budget): 0.0 (No Risk), 10.0, 15.0, 23.0, 30.0
# - Slate Exposure Cap: 0.10, 0.15, 0.25 (Max simultaneous stake per Saturday card)
# - Shrinkage: Raw Kelly vs. 800-draw Baker-McHale parameter uncertainty correction
#
# Runs in milliseconds because MatchBooks are pre-built.

include("_setup_scottish.jl")

champ_name = expr_champ.config.name
cache_file = joinpath(CACHE_DIR, "books_$(champ_name)_bm800.jls")
isfile(cache_file) || error("Run r01_build_books.jl first to generate books cache: $cache_file")

@info "Loading Champion MatchBooks from cache" champ_name
books = deserialize(cache_file)
slates = PF.group(PF.DailySlate(), books)

@printf("\nLoaded %d matches -> %d slates | Matches per slate: median %d, max %d\n",
        length(books), length(slates),
        median(length(s.books) for s in slates), maximum(length(s.books) for s in slates))

# -------------------------------------------------------------------
# Execute Policy Sweep
# -------------------------------------------------------------------
grid = DataFrame(
    trust = Float64[],
    lambda = Float64[],
    cap = Float64[],
    shrink = Bool[],
    mean_expo = Float64[],
    mean_k_risk = Float64[],
    roi = Float64[],
    final = Float64[],
    growth = Float64[],
    mdd = Float64[],
    sharpe = Float64[],
    n_capped = Int[]
)

@info "Sweeping 60 policy configurations on Champion model..."

for cap in (0.10, 0.15, 0.25), lam in (0.0, 10.0, 15.0, 23.0, 30.0), w in (0.10, 0.25, 0.50, 1.00), use_bm in (false, true)
    local pol = PF.PolicySpec(
        trust    = PF.FlatTrust(w),
        risk     = lam > 0.0 ? PF.SlateDrawdown(lam) : PF.NoRisk(),
        cap      = PF.FixedCap(cap),
        filter   = PF.KeepAll(),
        grouping = PF.DailySlate()
    )
    local traj = PF.simulate(pol, slates; use_shrink = use_bm)
    local m = PF.path_metrics(traj)
    
    # Calculate annualized-style slate Sharpe
    ret_series = traj.slate_pl
    sh = length(ret_series) > 1 && std(ret_series) > 1e-6 ? mean(ret_series) / std(ret_series) * sqrt(35) : 0.0
    
    push!(grid, (
        w, lam, cap, use_bm,
        round(m.mean_exposure, digits = 3),
        round(m.mean_k_risk, digits = 3),
        round(m.roi, digits = 3),
        round(m.final, digits = 3),
        round(m.growth_per_slate, digits = 5),
        round(m.mdd, digits = 3),
        round(sh, digits = 2),
        m.n_capped
    ))
end

println("\n", "="^100)
println("POLICY GRID SWEEP RESULTS — CHAMPION MODEL (funnel_pxg_apm)")
println("="^100)
show(grid; allrows = true, allcols = true, truncate = 0)
println()

# -------------------------------------------------------------------
# Top Policy Selections
# -------------------------------------------------------------------
println("\n", "="^100)
println("TOP POLICIES RANKED BY COMPOUND SLATE GROWTH (g)")
println("="^100)
sorted_growth = sort(grid, :growth, rev = true)
show(first(sorted_growth, 10); allrows = true, allcols = true, truncate = 0)
println()

println("\n", "="^100)
println("TOP CONSERVATIVE POLICIES (Max Drawdown < 20%) RANKED BY SHARPE")
println("="^100)
conservative = filter(r -> r.mdd <= 20.0, grid)
if !isempty(conservative)
    sorted_cons = sort(conservative, :sharpe, rev = true)
    show(first(sorted_cons, 10); allrows = true, allcols = true, truncate = 0)
else
    println("No policies with MDD <= 20% found.")
end
println()
