#=
RUNNER r02 — MPT / Kelly, deconstructed one step at a time on a single real match.

Not a race, not a backtest. Every step prints the object it just built so you can look at it
before moving on. Paste it block by block at the REPL, or `include` the whole thing for the
narrated version.

Prerequisites in scope (see r01 / the staking_layer preflight):
    include(".../staking_layer/src/loader.jl")     # StakingMatch, MMASK, solve_P, G_growth
    include(".../mpt_dev/l01_mpt_portfolio.jl")    # moments, solve_mpt, solve_msharpe, ...
    loaded = load_matches(RealSource(...))         # the real book

Reading order:
    1  the raw book            what the market is offering
    2  the score grid          the model's joint world distribution
    3  masks -> model probs    grid collapsed onto selections
    4  the edge                model vs market, per selection
    5  the return matrix R     the paper's rho = O - 1
    6  moments mu, Sigma       WHY a portfolio view differs from per-bet Kelly
    7  the efficient frontier  Markowitz, swept by hand
    8  where Kelly sits on it
    9  posterior uncertainty   the one thing the paper could not do
   10  parallel games          10 matches at once, MC-Kelly vs block-diagonal MPT
=#

using Statistics, LinearAlgebra, Printf, DataFrames

m = loaded.matches[1]
GRID = 12

banner(s) = println("\n" * "="^78 * "\n  " * s * "\n" * "="^78)

# ------------------------------------------------------------------
banner("1. THE RAW BOOK — what you are actually allowed to bet on")
# ------------------------------------------------------------------
# d = commission-adjusted decimal odds, q_mkt = de-vigged market prob, won = settlement.
# d == 1.0 means the line was MISSING from the exchange: return matrix column is pure loss,
# so the solver can never stake it. Check how many you actually have before trusting a result.

println(DataFrame(sel=SEL_NAMES, d=round.(m.d, digits=3),
                  q_mkt=round.(m.q_mkt, digits=4), won=m.won))
@printf("\nfinal score %d-%d   ·   live lines %d/11   ·   overround %.4f\n",
        m.score[1], m.score[2], count(m.d .> 1.0), sum(1 ./ m.d[m.d .> 1.0]))

# ------------------------------------------------------------------
banner("2. THE SCORE GRID — the model's joint distribution over worlds")
# ------------------------------------------------------------------
# m.P is 144 x S: one column per posterior draw, each a proper distribution over the 12x12
# scorelines. m.pbar is the posterior mean. THIS is the paper's p over k possible worlds --
# everything else in MPT/Kelly is a linear functional of it.

S = size(m.P, 2)
@printf("m.P is %d x %d   (144 states x %d posterior draws)\n", size(m.P)..., S)
@printf("each column sums to 1: %s   ·   pbar sums to %.10f\n",
        all(isapprox.(sum(m.P, dims=1), 1.0; atol=1e-9)), sum(m.pbar))

# eyeball it as a 12x12 (rows = home goals 0..11, cols = away goals 0..11)
Gbar = reshape(m.pbar, GRID, GRID)
println("\nposterior-mean score matrix, top-left 5x5 (rows=home, cols=away):")
show(stdout, "text/plain", round.(Gbar[1:5, 1:5], digits=4)); println()

# marginals -> expected goals. Note this is E over the POSTERIOR PREDICTIVE, not lambda-hat.
Eh = sum(vec(sum(Gbar, dims=2)) .* (0:GRID-1))
Ea = sum(vec(sum(Gbar, dims=1)) .* (0:GRID-1))
@printf("\nE[home goals] = %.3f   E[away goals] = %.3f   E[total] = %.3f\n", Eh, Ea, Eh + Ea)
@printf("most likely scoreline: %d-%d  (p = %.4f)\n",
        Tuple(argmax(Gbar))[1] - 1, Tuple(argmax(Gbar))[2] - 1, maximum(Gbar))

# ------------------------------------------------------------------
banner("3. MASKS — collapsing the grid onto the 11 selections")
# ------------------------------------------------------------------
# MMASK is 144 x 11, one binary column per selection. Model prob of selection j is just
# mask_j' * pbar. That single line is the whole bridge from "score model" to "betting market".

p_model = MMASK' * m.pbar
println(DataFrame(sel=SEL_NAMES, p_model=round.(p_model, digits=4),
                  n_states=Int.(vec(sum(MMASK, dims=1)))))

# sanity: 1X2 and each O/U pair must each sum to 1 (they partition the grid)
@printf("\n1X2 sums to %.6f   ·   O/U 2.5 pair sums to %.6f   ·   BTTS pair sums to %.6f\n",
        sum(p_model[1:3]), sum(p_model[6:7]), sum(p_model[10:11]))

# ------------------------------------------------------------------
banner("4. THE EDGE — model vs market, per selection")
# ------------------------------------------------------------------
# Two views. b = p*d - 1 is the per-bet expected return (what a per-bet Kelly signal screens on).
# It is NOT what the portfolio solvers maximise -- they maximise growth, which depends on how
# these co-move. Keep both in your head; step 6 is where they part company.

edge = DataFrame(sel=SEL_NAMES, p_model=round.(p_model, digits=4),
                 q_mkt=round.(m.q_mkt, digits=4),
                 diff=round.(p_model .- m.q_mkt, digits=4),
                 b=round.(p_model .* m.d .- 1.0, digits=4), live=m.d .> 1.0)
println(sort(edge[edge.live, :], :b, rev=true))

# ------------------------------------------------------------------
banner("5. THE RETURN MATRIX R — the paper's rho = O - 1")
# ------------------------------------------------------------------
# R is 144 x 11: R[w, j] = net return per unit staked on selection j if world w happens.
# Section 2.4 of the paper builds exactly this and calls it the odds matrix.

@printf("R is %d x %d\n", size(m.R)...)
j = findfirst(SEL_NAMES .== "home")
println("\ncolumn '$(SEL_NAMES[j])': unique values = ", unique(round.(m.R[:, j], digits=4)))
println("  -> d-1 = $(round(m.d[j]-1, digits=4)) in winning states, -1 in losing states")
println("\nreturn vector for the ACTUAL result $(m.score[1])-$(m.score[2]):")
# HGRID = vec([h for h in 0:11, a in 0:11]) is column-major with HOME varying fastest,
# so the linear index of scoreline (h,a) is  h + 12a + 1  -- not  12h + a + 1.
w_actual = min(m.score[1], GRID - 1) + GRID * min(m.score[2], GRID - 1) + 1
@assert HGRID[w_actual] == min(m.score[1], GRID-1) && AGRID[w_actual] == min(m.score[2], GRID-1)
println(DataFrame(sel=SEL_NAMES, R_actual=round.(m.R[w_actual, :], digits=3), won=m.won))

# ------------------------------------------------------------------
banner("6. MOMENTS — why a portfolio is not 11 independent bets")
# ------------------------------------------------------------------
# mu = E[rho] is the per-selection edge again. Sigma = Cov[rho] is what per-bet Kelly THROWS AWAY.
# Look at the correlation block: home and over_25 co-move, under_25 and btts_no co-move hard.
# Staking them as if independent double-counts the same underlying bet.

μ, Σ, S2 = moments(m.pbar, m.R)
live = findall(m.d .> 1.0)
D = sqrt.(diag(Σ)); C = Σ[live, live] ./ (D[live] * D[live]')
println("correlation of net returns (live selections only):")
show(stdout, "text/plain", DataFrame([SEL_NAMES[live] round.(C, digits=2)],
                                     ["sel"; SEL_NAMES[live]])); println()
@printf("\nlargest |correlation| off-diagonal: %.3f\n",
        maximum(abs.(C - I(length(live)))))

# ------------------------------------------------------------------
banner("7. THE EFFICIENT FRONTIER — Markowitz, swept by hand")
# ------------------------------------------------------------------
# Sweep gamma (risk aversion) and trace expected return against risk. This IS the paper's
# section 4.2 partial ordering, computed for one football match. High gamma -> all cash,
# low gamma -> pile onto the single best-EV selection.

cap = 0.2
front = DataFrame(γ=Float64[], stake=Float64[], E=Float64[], σ=Float64[], sharpe=Float64[], G=Float64[])
for γ in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0]
    f = solve_mpt(m.pbar, m.R; γ=γ, cap=cap)
    push!(front, (γ, sum(f), dot(μ, f), sqrt(max(dot(f, Σ * f), 0.0)),
                  sharpe(f, m.pbar, m.R), G_growth(f, m.pbar, m.R)))
end
println(round.(front, digits=5))

# ------------------------------------------------------------------
banner("8. WHERE KELLY SITS ON THAT FRONTIER")
# ------------------------------------------------------------------
# The paper's remark (eq 4.11): quadratic Kelly == MPT at gamma = 1/2, because the geometric mean
# is approximately the arithmetic mean minus half the variance. Check it on real numbers.
# Also note MaxSharpe picks the same DIRECTION but is forced to spend the whole cap -- which is
# how a strategy with the best Sharpe ratio ends up with worse log-growth.

strats = Dict("Kelly"       => solve_P(m.pbar, m.R; cap=cap),
              "QuadKelly"   => solve_quad_kelly(m.pbar, m.R; cap=cap),
              "MPT(γ=0.5)"  => solve_mpt(m.pbar, m.R; γ=0.5, cap=cap),
              "MSharpe"     => solve_msharpe(m.pbar, m.R; cap=cap),
              "MSharpeFrac" => frac(solve_msharpe(m.pbar, m.R; cap=cap), 0.5),
              "Drawdown"    => solve_kelly_drawdown(m.pbar, m.R; cap=cap, α=0.3, β=0.1),
              "DRO(η=0.1)"  => solve_kelly_dro(m.pbar, m.R; η=0.1, cap=cap))

cmp = DataFrame(strategy=String[], stake=Float64[], E=Float64[], σ=Float64[],
                sharpe=Float64[], G=Float64[], realised=Float64[])
payoff = [m.won[k] ? m.d[k] - 1.0 : -1.0 for k in 1:11]
for k in ["Kelly", "QuadKelly", "MPT(γ=0.5)", "MSharpe", "MSharpeFrac", "Drawdown", "DRO(η=0.1)"]
    f = strats[k]
    push!(cmp, (k, sum(f), dot(μ, f), sqrt(max(dot(f, Σ * f), 0.0)),
                sharpe(f, m.pbar, m.R), G_growth(f, m.pbar, m.R), dot(f, payoff)))
end
println(round.(cmp, digits=5))
@printf("\nmax |Kelly - QuadKelly| stake difference: %.6f\n",
        maximum(abs.(strats["Kelly"] .- strats["QuadKelly"])))

# ------------------------------------------------------------------
banner("9. POSTERIOR UNCERTAINTY — the bit the paper could not do")
# ------------------------------------------------------------------
# The paper fakes parameter uncertainty with a box of radius eta around a point estimate (5.4).
# You have S posterior draws. Two things worth seeing:
#
#  (a) The Kelly objective is LINEAR in p. So solving once at pbar already maximises the
#      Bayes-expected growth -- there is nothing to gain by solving per draw. But the ARGMAX is
#      not linear, so averaging the per-draw SOLUTIONS gives a different (and wrong) portfolio.
#      That is a trap worth seeing once with your own numbers.
#  (b) Sharpe is NOT linear in p, so its per-draw spread is real information about fragility.

F = hcat([solve_quad_kelly(m.P[:, s], m.R; cap=cap) for s in 1:S]...)   # 11 x S
f_pbar = solve_quad_kelly(m.pbar, m.R; cap=cap)
f_avg  = vec(mean(F, dims=2))

println(DataFrame(sel=SEL_NAMES,
                  f_at_pbar=round.(f_pbar, digits=4),
                  f_avg_of_draws=round.(f_avg, digits=4),
                  f_sd=round.(vec(std(F, dims=2)), digits=4),
                  frac_draws_staked=round.(vec(mean(F .> 1e-6, dims=2)), digits=3)))
@printf("\nG at the pbar solution      : %.6f\n", G_growth(f_pbar, m.pbar, m.R))
@printf("G at the averaged solution  : %.6f   <- lower: averaging argmaxes is NOT Bayesian\n",
        G_growth(f_avg, m.pbar, m.R))

# how often would a selection be staked at all? a selection staked in 40% of draws is a
# coin-flip dressed as an edge -- this is the posterior version of the paper's eta.
unstable = SEL_NAMES[(vec(mean(F .> 1e-6, dims=2)) .> 0.05) .& (vec(mean(F .> 1e-6, dims=2)) .< 0.95)]
println("\nselections staked in only some draws (fragile): ", isempty(unstable) ? "none" : unstable)

# ------------------------------------------------------------------
banner("10. PARALLEL GAMES — 10 matches at once (paper section 2.4.1)")
# ------------------------------------------------------------------
# The open problem. Exact Kelly over 10 simultaneous matches needs 144^10 joint worlds. Two
# tractable routes, both computable from what you already have:
#
#   (A) MC-Kelly: sample joint worlds from the per-match grids and run the SAME solve_P on the
#       sampled scenario matrix. Exact in the limit of scenarios, no independence assumption
#       beyond matches being independent (which they are, pregame).
#   (B) Block-diagonal MPT: independent matches -> Sigma is block diagonal, so the quadratic
#       objective factorises and you can solve the whole round in one shot.
#
# Compare both against staking each match separately. The question is whether joint allocation
# actually buys anything once a portfolio cap is already binding.

R_ = 10                                    # a "round" of parallel matches
round_ms = loaded.matches[1:R_]
n_scen = 4000

# (A) Monte-Carlo joint worlds
Random_state = 20260722
using Random; Random.seed!(Random_state)
cols = Matrix{Float64}[]
for mm in round_ms
    cdf = cumsum(mm.pbar)                                                 # hoist: O(144) once
    st  = [clamp(searchsortedfirst(cdf, rand()), 1, 144) for _ in 1:n_scen]
    push!(cols, mm.R[st, :])                                              # n_scen x 11
end
R_joint = hcat(cols...)                                                   # n_scen x 110
p_joint = fill(1.0 / n_scen, n_scen)

f_joint = solve_P(p_joint, R_joint; cap=cap)
f_sep   = vcat([solve_P(mm.pbar, mm.R; cap=cap) for mm in round_ms]...)

@printf("joint  MC-Kelly : total stake %.4f over %d live legs\n",
        sum(f_joint), count(f_joint .> 1e-6))
@printf("separate Kelly  : total stake %.4f over %d live legs  (cap %.2f applied PER MATCH)\n",
        sum(f_sep), count(f_sep .> 1e-6), cap)
@printf("\nGrowth on the SAME scenario set:\n  joint    G = %.6f\n  separate G = %.6f\n",
        G_growth(f_joint, p_joint, R_joint), G_growth(f_sep, p_joint, R_joint))
println("\n^ separate staking spends R x cap of bankroll across the round. That is the")
println("  bankruptcy mechanism from your earlier portfolio-Kelly work, visible in one number.")

# (B) block-diagonal MPT over the same round
μ_blk = vcat([moments(mm.pbar, mm.R)[1] for mm in round_ms]...)
Σ_blk = cat([Matrix(moments(mm.pbar, mm.R)[2]) for mm in round_ms]...; dims=(1, 2))
f_blk = let M = length(μ_blk)
    projected_ascent(fill(1e-3, M),
                     f -> dot(μ_blk, f) - 0.5 * dot(f, Σ_blk * f),
                     (g, f) -> (g .= μ_blk .- (Σ_blk * f)),
                     f -> proj_cap!(f, cap))
end
@printf("\nblock-diag MPT  : total stake %.4f   G on MC scenarios = %.6f\n",
        sum(f_blk), G_growth(f_blk, p_joint, R_joint))
@printf("stake correlation joint-MC vs block-MPT: %.4f\n", cor(f_joint, f_blk))
