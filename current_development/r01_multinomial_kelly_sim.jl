using Pkg
Pkg.activate(".")
using Optim
using Distributions
using Random
using LinearAlgebra
using Base.Threads
using ThreadPinning

pinthreads(:cores) # Lock OS threads as per BayesianFootball conventions

println("Starting Massive Multithreaded Multinomial Simulation...")
println("Running on ", Threads.nthreads(), " threads.")

N = 4 # Max goals per team (0 to 4)
M = 20_000 # Number of simulated matches
S = 50 # Number of fake MCMC draws per match (lowered slightly for speed)

# Market Masks
function get_market_masks(N)
    U15 = zeros(N+1, N+1)
    BTTS = zeros(N+1, N+1)
    H_WIN = zeros(N+1, N+1)
    for h in 0:N
        for a in 0:N
            if h + a < 1.5 U15[h+1, a+1] = 1.0 end
            if h > 0 && a > 0 BTTS[h+1, a+1] = 1.0 end
            if h > a H_WIN[h+1, a+1] = 1.0 end
        end
    end
    return U15, BTTS, H_WIN
end

U15_mask, BTTS_mask, H_WIN_mask = get_market_masks(N)

# Utility function
function expected_log_wealth(s, probs; odds, masks)
    if sum(s) >= 0.99 || any(s .< 0.0) return -Inf end
    elw = 0.0
    for h in 1:(N+1), a in 1:(N+1)
        W = (1.0 - sum(s))
        for m in 1:length(s) W += s[m] * masks[m][h,a] * odds[m] end
        if W <= 0.0 return -Inf end
        elw += probs[h,a] * log(W)
    end
    return elw
end

naive_logs = zeros(M)
shrunk_logs = zeros(M)
bankrupt_naive = Threads.Atomic{Int}(0)
bankrupt_shrunk = Threads.Atomic{Int}(0)

println("\nSimulating ", M, " matches across ", Threads.nthreads(), " cores...")

Threads.@threads for m in 1:M
    # Thread-local RNG
    rng = Xoshiro(m)
    
    # 1. Generate True Probs for the match
    raw = rand(rng, N+1, N+1)
    raw[1:2, 1:2] .*= 3.0 # Bump up 0-0, 1-0 etc
    true_probs = raw ./ sum(raw)
    
    # Calculate Market Odds (Vig = 3%)
    vig = 1.03
    o_U15 = 1.0 / (sum(true_probs .* U15_mask) * vig)
    o_BTTS = 1.0 / (sum(true_probs .* BTTS_mask) * vig)
    o_HWIN = 1.0 / (sum(true_probs .* H_WIN_mask) * vig)
    odds = [o_U15, o_BTTS, o_HWIN]
    masks = [U15_mask, BTTS_mask, H_WIN_mask]
    
    # 2. Model Probs (We have some edge, but also some noise/error)
    model_probs = copy(true_probs)
    # Randomly boost the edge between 10% and 40% on U15 and BTTS
    model_probs .*= 1.0 .+ (U15_mask .* (0.1 + rand(rng)*0.3)) 
    model_probs .*= 1.0 .+ (BTTS_mask .* (0.1 + rand(rng)*0.3))
    model_probs ./= sum(model_probs)
    
    # 3. Naive Optimization (Long/Whelan)
    obj(s) = -expected_log_wealth(s, model_probs; odds=odds, masks=masks)
    res = optimize(obj, [0.0,0.0,0.0], [0.99,0.99,0.99], [0.01,0.01,0.01], Fminbox(LBFGS()))
    naive_s = Optim.minimizer(res)
    
    # If no bets placed, just record log(1) = 0 and continue
    if sum(naive_s) < 1e-4
        naive_logs[m] = 0.0
        shrunk_logs[m] = 0.0
        continue
    end
    
    # 4. Bayesian Shrinkage over fake MCMC draws (McHale)
    mcmc_draws = []
    for _ in 1:S
        # The MCMC draws show the parameter uncertainty (variance) around our model_probs
        d = model_probs .+ randn(rng, N+1, N+1) * 0.02
        d .= max.(d, 0.0001)
        d ./= sum(d)
        push!(mcmc_draws, d)
    end
    
    function shrink_obj(k_vec)
        k = k_vec[1]
        if k <= 0.0 || k > 1.0 return Inf end
        scaled = k .* naive_s
        mean_lw = 0.0
        for draw in mcmc_draws
            elw = expected_log_wealth(scaled, draw; odds=odds, masks=masks)
            if elw == -Inf return Inf end
            mean_lw += elw
        end
        return -(mean_lw / S)
    end
    
    res_shrink = optimize(shrink_obj, [0.01], [1.0], [0.5], Fminbox(LBFGS()))
    opt_k = Optim.minimizer(res_shrink)[1]
    shrunk_s = opt_k .* naive_s
    
    # 5. Play the Match! Draw the actual outcome from true_probs
    cdf = cumsum(vec(true_probs))
    val = rand(rng)
    idx = findfirst(x -> x >= val, cdf)
    (h, a) = Tuple(CartesianIndices(true_probs)[idx])
    
    # Calculate Terminal Wealth Multiplier
    W_naive = (1.0 - sum(naive_s))
    W_shrunk = (1.0 - sum(shrunk_s))
    
    for i in 1:length(naive_s)
        if masks[i][h,a] > 0.5
            W_naive += naive_s[i] * odds[i]
            W_shrunk += shrunk_s[i] * odds[i]
        end
    end
    
    # Severe ruin penalty (down to 1% of bankroll)
    if W_naive <= 0.01 
        Threads.atomic_add!(bankrupt_naive, 1)
        W_naive = 0.01 
    end
    if W_shrunk <= 0.01 
        Threads.atomic_add!(bankrupt_shrunk, 1)
        W_shrunk = 0.01 
    end
    
    naive_logs[m] = log(W_naive)
    shrunk_logs[m] = log(W_shrunk)
end

println("\n--- Simulation Complete ---")
println("Matches Simulated: ", M)
println("Total Naive Bankruptcies (<1% left):  ", bankrupt_naive[])
println("Total Shrunk Bankruptcies (<1% left): ", bankrupt_shrunk[])

# Total geometric return (sum of logs)
total_naive_log = sum(naive_logs)
total_shrunk_log = sum(shrunk_logs)

println("\nTotal Naive Log-Wealth:   ", round(total_naive_log, digits=4))
println("Total Shrunk Log-Wealth:  ", round(total_shrunk_log, digits=4))

# Final Bankroll per starting unit
println("\nTheoretical Final Bankroll (Starting at \$1.00):")
# To prevent printing Inf or massive numbers, check bounds
if total_naive_log > 100
    println("Naive:  \$ MASSIVE (Log: ", round(total_naive_log, digits=2), ")")
else
    println("Naive:  \$ ", round(exp(total_naive_log), digits=2))
end

if total_shrunk_log > 100
    println("Shrunk: \$ MASSIVE (Log: ", round(total_shrunk_log, digits=2), ")")
else
    println("Shrunk: \$ ", round(exp(total_shrunk_log), digits=2))
end

if total_shrunk_log > total_naive_log
    println("\n✅ TOTAL WIN: Shrunken Portfolio generated significantly higher long-term wealth!")
else
    println("\n❌ Naive was better.")
end
