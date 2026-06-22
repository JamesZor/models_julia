using Pkg
Pkg.activate(".")
using Optim
using Distributions
using Random
using LinearAlgebra

println("Starting Toy Example...")

# 1. Setup Toy Probabilities (N x N grid, N=4 max goals)
Random.seed!(42)
N = 4 # 0 to 4 goals
raw_probs = rand(N+1, N+1)
raw_probs[1:2, 1:2] .*= 3.0 # Boost low scores (0-0, 1-0, 0-1, 1-1) to be more realistic
true_probs = raw_probs ./ sum(raw_probs)

# 2. Define Market Masks
# We want to bet on: Under 1.5, BTTS (Both Teams To Score), Home Win
function get_market_masks(N)
    U15 = zeros(N+1, N+1)
    BTTS = zeros(N+1, N+1)
    H_WIN = zeros(N+1, N+1)
    
    for h in 0:N
        for a in 0:N
            if h + a < 1.5
                U15[h+1, a+1] = 1.0
            end
            if h > 0 && a > 0
                BTTS[h+1, a+1] = 1.0
            end
            if h > a
                H_WIN[h+1, a+1] = 1.0
            end
        end
    end
    return U15, BTTS, H_WIN
end

U15_mask, BTTS_mask, H_WIN_mask = get_market_masks(N)

# Calculate fair odds, and add a small bookie margin (vig)
vig = 1.03 # 3% overround
odds_U15   = 1.0 / (sum(true_probs .* U15_mask) * vig)
odds_BTTS  = 1.0 / (sum(true_probs .* BTTS_mask) * vig)
odds_HWIN  = 1.0 / (sum(true_probs .* H_WIN_mask) * vig)

println("Market Odds Offered:")
println("  Under 1.5: ", round(odds_U15, digits=2))
println("  BTTS:      ", round(odds_BTTS, digits=2))
println("  Home Win:  ", round(odds_HWIN, digits=2))

# Create our "Model's" probabilities (giving us a massive edge to ensure betting)
model_probs = copy(true_probs)
model_probs .*= 1.0 .+ (U15_mask .* 0.4)   # 40% higher probability than bookie expects
model_probs .*= 1.0 .+ (BTTS_mask .* 0.2)  # 20% higher probability
model_probs ./= sum(model_probs)

println("\nChecking Expected Values (EV):")
println("  U1.5 EV:  ", round(sum(model_probs .* U15_mask) * odds_U15, digits=2))
println("  BTTS EV:  ", round(sum(model_probs .* BTTS_mask) * odds_BTTS, digits=2))
println("  HWIN EV:  ", round(sum(model_probs .* H_WIN_mask) * odds_HWIN, digits=2))

# 3. Naive Multinomial Kelly Optimization (Long / Whelan)
# We have 3 markets. s = [stake_u15, stake_btts, stake_hwin]
function expected_log_wealth(s, probs; odds=[odds_U15, odds_BTTS, odds_HWIN], masks=[U15_mask, BTTS_mask, H_WIN_mask])
    if sum(s) >= 0.99 || any(s .< 0.0)
        return -Inf # Invalid state
    end
    
    expected_log_w = 0.0
    for h in 1:(N+1)
        for a in 1:(N+1)
            # Terminal wealth for this specific exact score
            W = (1.0 - sum(s)) # Cash that survives
            for m in 1:length(s)
                W += s[m] * masks[m][h,a] * odds[m]
            end
            
            if W <= 0.0
                return -Inf
            end
            expected_log_w += probs[h,a] * log(W)
        end
    end
    return expected_log_w
end

# Optim.jl requires minimization
objective(s) = -expected_log_wealth(s, model_probs)

initial_s = [0.01, 0.01, 0.01]
lower = [0.0, 0.0, 0.0]
upper = [0.99, 0.99, 0.99]

res = optimize(objective, lower, upper, initial_s, Fminbox(LBFGS()))
naive_stakes = Optim.minimizer(res)

println("\n--- Naive Multinomial Kelly Stakes ---")
println("U1.5 Stake:  ", round(naive_stakes[1], digits=4))
println("BTTS Stake:  ", round(naive_stakes[2], digits=4))
println("H_WIN Stake: ", round(naive_stakes[3], digits=4))
println("Total Stake: ", round(sum(naive_stakes), digits=4))


# 4. Bayesian Shrinkage over "MCMC" draws (Baker & McHale)
println("\nGenerating 500 fake MCMC posterior draws with variance...")
S = 500
mcmc_draws = []
for _ in 1:S
    # Simulate parameter uncertainty
    draw = model_probs .+ randn(N+1, N+1) * 0.02
    draw .= max.(draw, 0.0001)
    draw ./= sum(draw)
    push!(mcmc_draws, draw)
end

# Objective for shrinkage factor k
function shrinkage_objective(k_vec)
    k = k_vec[1]
    if k <= 0.0 || k > 1.0
        return Inf
    end
    
    scaled_stakes = k .* naive_stakes
    
    mean_log_w = 0.0
    for draw in mcmc_draws
        elw = expected_log_wealth(scaled_stakes, draw)
        if elw == -Inf
            return Inf
        end
        mean_log_w += elw
    end
    return -(mean_log_w / S)
end

res_shrink = optimize(shrinkage_objective, [0.01], [1.0], [0.5], Fminbox(LBFGS()))
optimal_k = Optim.minimizer(res_shrink)[1]

println("\n--- Bayesian Shrinkage (McHale) ---")
println("Optimal Shrinkage Factor k*: ", round(optimal_k, digits=4))
println("Shrunk U1.5 Stake:  ", round(optimal_k * naive_stakes[1], digits=4))
println("Shrunk BTTS Stake:  ", round(optimal_k * naive_stakes[2], digits=4))
println("Shrunk H_WIN Stake: ", round(optimal_k * naive_stakes[3], digits=4))
println("Total Shrunk Stake: ", round(optimal_k * sum(naive_stakes), digits=4))

# 5. Out-of-Sample Performance Comparison
println("\n--- Out-of-Sample Performance Comparison ---")
shrunk_stakes = optimal_k .* naive_stakes

naive_mean_log_w = 0.0
shrunk_mean_log_w = 0.0

for draw in mcmc_draws
    naive_mean_log_w += expected_log_wealth(naive_stakes, draw)
    shrunk_mean_log_w += expected_log_wealth(shrunk_stakes, draw)
end

naive_mean_log_w /= S
shrunk_mean_log_w /= S

println("Naive Expected Log-Wealth:   ", round(naive_mean_log_w, digits=6))
println("Shrunk Expected Log-Wealth:  ", round(shrunk_mean_log_w, digits=6))

# Translate log-wealth to expected geometric growth rate per match
naive_growth = exp(naive_mean_log_w) - 1.0
shrunk_growth = exp(shrunk_mean_log_w) - 1.0

println("\nNaive Expected Growth Rate:   ", round(naive_growth * 100, digits=4), "% per match")
println("Shrunk Expected Growth Rate: ", round(shrunk_growth * 100, digits=4), "% per match")

if shrunk_mean_log_w > naive_mean_log_w
    println("\n✅ WIN: Shrinkage increased out-of-sample expected utility by protecting against parameter variance!")
else
    println("\n❌ Naive was better (this shouldn't happen if k* was optimized properly).")
end
