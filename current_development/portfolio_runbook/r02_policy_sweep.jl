# r02_policy_sweep.jl -- build once, sweep for free.
#
# This is the file that justifies the whole architecture. Books cost ~26 seconds to build;
# simulating a policy against them costs milliseconds. So a 36-cell policy grid is essentially
# free, and walk-forward evaluation becomes affordable rather than a weekend job.
#
# It also demonstrates the least intuitive property of the system: once the drawdown constraint
# binds, TRUST CANNOT CHANGE HOW MUCH YOU STAKE.

include("_setup.jl")

# ===================================================================
# 1. Build books once -- and persist them
# ===================================================================
#
# The cache is keyed on the BookSpec. Change `price`, `allocator`, `shrink` or `exec` and the
# key changes and you must rebuild. Change anything in a PolicySpec and you must not.

spec = PF.BookSpec(markets = MARKETS)
CACHE = joinpath(@__DIR__, "books_$(string(PF.book_cache_key(spec), base = 16)).jls")

books = if isfile(CACHE)
    @info "reusing cached books" CACHE
    deserialize(CACHE)
else
    @info "building books" CACHE
    b = @time PF.build_books(spec, latents_df, expr, odds, ds)
    serialize(CACHE, b)
    b
end
slates = PF.group(PF.DailySlate(), books)

# ===================================================================
# 2. The sweep
# ===================================================================

grid = DataFrame(trust = Float64[], lambda = Float64[], cap = Float64[], shrink = Bool[],
                 mean_expo = Float64[], mean_k = Float64[], roi = Float64[],
                 final = Float64[], growth = Float64[], mdd = Float64[], capped = Int[])

@info "sweeping policies (books are NOT rebuilt)"
@time for w in (0.10, 0.25, 0.50, 1.00), lam in (0.0, 10.0, 23.0), use_bm in (false, true)
    pol = PF.PolicySpec(trust = PF.FlatTrust(w),
                        risk  = lam > 0 ? PF.SlateDrawdown(lam) : PF.NoRisk(),
                        cap   = PF.FixedCap(0.25))
    t = PF.simulate(pol, slates; use_shrink = use_bm)
    m = PF.path_metrics(t)
    push!(grid, (w, lam, 0.25, use_bm, m.mean_exposure, m.mean_k_risk, m.roi,
                 m.final, m.growth_per_slate, m.mdd, m.n_capped))
end

for c in (:mean_expo, :mean_k, :roi, :final, :mdd)
    grid[!, c] = round.(grid[!, c], digits = 3)
end
grid.growth = round.(grid.growth, digits = 5)

println("\n", "="^96, "\n=== POLICY GRID ===\n", "="^96)
println(grid)

# ===================================================================
# 3. Read the grid: three things it is telling you
# ===================================================================
println("\n", "="^96, "\n=== HOW TO READ IT ===\n", "="^96)

# (a) Flat ROI is invariant to flat trust. Trust scales every stake by the same factor, and ROI
#     is a ratio of P/L to stake -- so it cancels. Never pick trust on ROI; pick it on growth.
let no_risk = grid[(grid.lambda .== 0) .& (.!grid.shrink), :]
    println("\n(a) ROI is blind to flat trust -- at lambda = 0 every trust level gives ROI ",
            unique(no_risk.roi), "\n    while final bankroll ranges ", extrema(no_risk.final),
            ".\n    => choose trust on GROWTH, never on ROI.")
end

# (b) Once lambda binds, trust stops mattering entirely. The risk model solves for the factor
#     that makes the stakes it is handed satisfy the constraint, so doubling the input halves
#     the factor. Trust reshapes the book; it cannot resize it.
let bound = grid[(grid.lambda .== 23.0) .& (.!grid.shrink), :]
    println("\n(b) At lambda = 23, trust 0.10 -> 1.00 all give final ", unique(bound.final),
            "\n    (mean_k falls ", first(bound.mean_k), " -> ", last(bound.mean_k),
            " to exactly compensate).",
            "\n    => lambda SUBSUMES trust. To move exposure, move lambda.")
end

# (c) Full Kelly without a risk budget is ruinous, and the cap is what stops it being fatal.
let f = grid[(grid.trust .== 1.0) .& (grid.lambda .== 0) .& (.!grid.shrink), :]
    println("\n(c) trust = 1.0, no lambda: final ", first(f.final), "x, drawdown ",
            first(f.mdd), "%, cap hit on ", first(f.capped), " slates.",
            "\n    Without FixedCap this configuration drives the bankroll NEGATIVE.")
end

# ===================================================================
# 4. Calibrate lambda to a target exposure -- the correct dial
# ===================================================================
#
# If you want to compare two systems fairly, put them on the same risk, then compare growth.
# Do it by moving lambda. Do NOT do it with a stake multiplier: an active constraint is
# scale-invariant, so the multiplier is a no-op (demonstrated below).

println("\n", "="^96, "\n=== CALIBRATING TO A TARGET EXPOSURE ===\n", "="^96)

base = PF.PolicySpec(trust = PF.FlatTrust(1.0), cap = PF.FixedCap(0.99))
for target in (0.05, 0.10, 0.15)
    lam = PF.calibrate_lambda(base, slates; target_exposure = target, use_shrink = false)
    pol = PF.PolicySpec(trust = PF.FlatTrust(1.0), risk = PF.SlateDrawdown(lam),
                        cap = PF.FixedCap(0.99))
    m = PF.path_metrics(PF.simulate(pol, slates; use_shrink = false))
    @printf("  target %.0f%% -> lambda %6.2f | realised %.1f%% | growth %.5f | final %.3fx | mdd %.1f%%\n",
            100target, lam, 100m.mean_exposure, m.growth_per_slate, m.final, m.mdd)
end

println("\n  and here is why the stake multiplier is NOT the dial:")
let pol = PF.PolicySpec(trust = PF.FlatTrust(1.0), risk = PF.SlateDrawdown(20.0),
                        cap = PF.FixedCap(0.99))
    for sc in (0.25, 1.0, 4.0)
        e = mean(PF.simulate(pol, slates; use_shrink = false, scale = sc).exposure)
        @printf("    stake multiplier %5.2f -> mean exposure %.4f\n", sc, e)
    end
    println("    identical. The constraint absorbs it. Use calibrate_lambda.")
end
