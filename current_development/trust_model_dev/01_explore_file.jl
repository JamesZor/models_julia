using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

pinthreads(:cores)


const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Data        = BayesianFootball.Data
const Signals     = BayesianFootball.Signals


ds = Data.load_datastore_cached(Data.Ireland())
src_dir   = "./data/double_poisson_smile_src_grid/"   # NEW: this runner's src cells



# load_experiments
list_of_experiments = Experiments.list_experiments(src_dir, data_dir="")
expr = Experiments.load_experiment(list_of_experiments, 5)

split_id = 25

split = BayesianFootball.Data.create_data_splits(ds, expr.config.splitter)[split_id]
#=
13 columns and 400 rows omitted, GroupedSplit(Tourns: [79], Season: 2026, Week: 8, Hist: 2))
=#

expr_split = expr.training_results[split_id]
#=
julia> expr.training_results[split_id]
(MCMC chain (800×68×4 Array{Float64, 3}), GroupedSplit(Tourns: [79], Season: 2026, Week: 6, Hist: 2))
=#

next_matches = BayesianFootball.Data.get_next_matches(ds, expr_split, expr.config.splitter)

all_expr_inference = BayesianFootball.Predictions.model_inference(ds, expr) 

next_matches_inference = subset(all_expr_inference.df, :match_id => in( next_matches.match_id))


valid_match_ids = Set(next_matches.match_id)
# in. uses broadcasting to check multiple elements at once, Ref() protects the Set from being broadcasted over
result = subset(all_expr_inference.df, :match_id => x -> in.(x, Ref(valid_match_ids)))



past_match_ids_covers = collect( split[1].match_id)


# Use a Set for fast lookup
past_match_ids_set = Set(past_match_ids_covers)

# Filter odds to just the training split
past_odds = subset(ds.odds, :match_id => x -> in.(x, Ref(past_match_ids_set)))

# Select and rename the columns we need
odds_features = select(past_odds, 
    :match_id, 
    :selection, 
    :is_winner, 
    :prob_fair_close => :market_devig
)



# 1. Rebuild the team_map manually
# The model maps teams by taking all unique teams in the training split, 
# sorting them alphabetically, and assigning 1 to N.
all_teams_in_split = sort(unique(vcat(split[1].home_team, split[1].away_team)))
team_map = Dict(name => i for (i, name) in enumerate(all_teams_in_split))

# 2. Extract model_prob from `distribution`
model_preds = select(all_expr_inference.df, 
    :match_id, 
    :selection, 
    :distribution => ByRow(mean) => :model_prob
)

# 3. Get the home and away team strings from ds.matches
match_teams = select(ds.matches, :match_id, :home_team, :away_team)

# 4. Filter odds to our validation set
past_odds = subset(ds.odds, :match_id => x -> in.(x, Ref(Set(past_match_ids_covers))))
odds_features = select(past_odds, 
    :match_id, 
    :selection, 
    :is_winner, 
    :prob_fair_close => :market_devig
)

# 5. Join them all together!
feature_matrix = innerjoin(odds_features, model_preds, on=[:match_id, :selection])
feature_matrix = innerjoin(feature_matrix, match_teams, on=:match_id)

# 6. Apply the `team_map` to get the indices
feature_matrix.home_team_index = [get(team_map, t, 0) for t in feature_matrix.home_team]
feature_matrix.away_team_index = [get(team_map, t, 0) for t in feature_matrix.away_team]

# Keep only the columns you requested
feature_matrix = select(feature_matrix, 
    :match_id, 
    :selection, 
    :is_winner, 
    :market_devig, 
    :model_prob, 
    :home_team_index, 
    :away_team_index
)



using Turing
using Distributions
using StatsFuns: logistic

# 1. The Builder Function (Data prep outside the model)
function build_trust_model(df::AbstractDataFrame)
# Convert dataframe columns to pure, static arrays for the AD tape
  is_winner = Vector{Int}(df.is_winner)
  p_model   = Vector{Float64}(df.model_prob)
  q_market  = Vector{Float64}(df.market_devig)
  
  return single_line_trust(is_winner, p_model, q_market)
end

# 2. The Engine (Pure vector math, no loops)
@model function single_line_trust(is_winner::Vector{Int}, p_model::Vector{Float64}, q_market::Vector{Float64})
# Prior on trust (mean < 0 means "distrust the model by default")
    w_raw ~ Normal(-1.0, 1.0) 
    
    w_l = logistic(w_raw)
    
# Pure broadcast arithmetic (creates 1 tracked array node)
    p_tilde = clamp.(w_l .* p_model .+ (1.0 - w_l) .* q_market, 1e-6, 1.0 - 1e-6)
    
# Vectorized likelihood (AD-safe way to add to the target log-density)
    ll = logpdf.(Bernoulli.(p_tilde), is_winner)
    Turing.@addlogprob! sum(ll)
end



using StatsFuns: logistic

# 1. Filter the feature matrix for just the Over 2.5 market
# Make sure the string exactly matches how it appears in your `selection` column
df_over25 = subset(feature_matrix, :selection => ByRow(==(:under_25)))

println("Found $(nrow(df_over25)) historical matches for Over 2.5")

# 2. Instantiate the Turing model with our AD-safe builder
trust_model_over25 = build_trust_model(df_over25)

# 3. Setup MCMC Sampling (Matching your NUTSConfig)
n_samples   = 1000
n_chains    = 4
n_warmup    = 500
accept_rate = 0.65
max_depth   = 10

println("Starting MCMC sampling (4 chains, multithreaded)...")

# 4. Execute the sampler
chain_over25 = sample(
    trust_model_over25, 
    NUTS(n_warmup, accept_rate; max_depth=max_depth), 
    MCMCThreads(), 
    n_samples, 
    n_chains
)

# 5. View the results!
describe(chain_over25)

#=
Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 3.88 seconds
Compute duration  = 15.42 seconds
parameters        = w_raw
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics

  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64 

       w_raw   -0.8343    1.0050    0.0242   1742.9086   2655.6592    1.0034      113.0071


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

       w_raw   -2.7724   -1.5123   -0.8501   -0.1295    1.1084
=#


# 6. Transform the posterior back from logit-space to see the actual trust weight `w_l`
w_posterior = logistic.(vec(Array(chain_over25[:w_raw])));


function p_d()
  println("=========================================")
  println("OVER 2.5 TRUST WEIGHT ANALYSIS")
  println("Mean Trust: ", round(mean(w_posterior), digits=3))
  println("Median Trust: ", round(median(w_posterior), digits=3))
  println("95% HDI: ", round.(quantile(w_posterior, [0.025, 0.975]), digits=3))
end


p_d()

#=
julia> p_d()
=========================================
OVER 2.5 TRUST WEIGHT ANALYSIS
Mean Trust: 0.334
Median Trust: 0.299
95% HDI: [0.059, 0.752]
=#









   # 1. Fix the subset error using `ByRow` and a Set for speed
valid_match_ids = Set(next_matches.match_id)
next_matches_inference = subset(all_expr_inference.df, :match_id => ByRow(x -> x in valid_match_ids))

# 2. Filter down to just the Over 2.5 predictions
# Note: we use the Symbol `:over_25` based on your dataframe output
next_model_over25 = subset(next_matches_inference, :selection => ByRow(==(:over_25)))

# Extract the mean model probability from the distribution
next_model_preds = select(next_model_over25, 
    :match_id, 
    :selection, 
    :distribution => ByRow(mean) => :model_prob
)

# 3. Get the market odds for the next matches
next_odds = subset(ds.odds, :match_id => ByRow(x -> x in valid_match_ids))
next_odds_features = select(next_odds, 
    :match_id, 
    :selection, 
    :is_winner,
    :prob_fair_close => :market_devig,
    :odds_close # Let's keep the actual decimal odds so we can see the price!
)

# 4. Join the model predictions and the market odds
next_matrix = innerjoin(next_model_preds, next_odds_features, on=[:match_id, :selection])

# 5. Apply your learned Trust Blend!
w_l = 0.334 # The mean trust we learned from the training split

# Calculate the final probability we will bet on
next_matrix.blended_prob = @. (w_l * next_matrix.model_prob) + ((1.0 - w_l) * next_matrix.market_devig)

# Let's also calculate the EV (Expected Value) using our blended probability vs the bookmaker's decimal odds
next_matrix.expected_value = @. (next_matrix.blended_prob * next_matrix.odds_close) - 1.0

# View the results sorted by highest edge
sort!(next_matrix, :expected_value, rev=true)
display(next_matrix)



# Here is the code to run your hierarchical model:
# 1. The Builder Function
function build_hierarchical_model(df::AbstractDataFrame, total_teams::Int)
  is_winner = Vector{Int}(df.is_winner)
  p_model   = Vector{Float64}(df.model_prob)
  q_market  = Vector{Float64}(df.market_devig)
  home_idx  = Vector{Int}(df.home_team_index)
  away_idx  = Vector{Int}(df.away_team_index)
  
  return hierarchical_trust(is_winner, p_model, q_market, home_idx, away_idx, total_teams)
end

# 2. The Hierarchical Engine
@model function hierarchical_trust(
  is_winner::Vector{Int}, 
  p_model::Vector{Float64}, 
  q_market::Vector{Float64},
  home_idx::Vector{Int},
  away_idx::Vector{Int},
  n_teams::Int
)
  # Global trust intercept (w0)
  w0 ~ Normal(-1.0, 1.0)
  
  # The spread between teams (must be strictly positive, hence Exponential)
  σ_team ~ Exponential(0.5)
  
  # The raw, non-centered team deltas
  z_team ~ filldist(Normal(0, 1), n_teams)
  
  # AD-SAFE: Extract the team effects using zero-copy `view()`
  home_effects = view(z_team, home_idx)
  away_effects = view(z_team, away_idx)
  
  # Calculate the final logit-trust for every single match
  η = w0 .+ σ_team .* (home_effects .+ away_effects)
  
  # Transform to probability space (creates a vector of w_l for every match)
  w_l_vector = logistic.(η)
  
  # Broadcast the blend
  p_tilde = clamp.(w_l_vector .* p_model .+ (1.0 .- w_l_vector) .* q_market, 1e-6, 1.0 - 1e-6)
  
  # Vectorized likelihood
  ll = logpdf.(Bernoulli.(p_tilde), is_winner)
  Turing.@addlogprob! sum(ll)
end

# 3. Instantiate and Sample!
# We need to know the maximum number of teams to size our `z_team` array
total_teams = length(team_map)

hierarchical_model_over25 = build_hierarchical_model(df_over25, total_teams)

println("Starting Hierarchical MCMC sampling (4 chains, multithreaded)...")

chain_hierarchical = sample(
  hierarchical_model_over25, 
  NUTS(n_warmup, accept_rate; max_depth=max_depth), 
  MCMCThreads(), 
  n_samples, 
  n_chains
)

describe(chain_hierarchical)

#=
julia> describe(chain_hierarchical)
Chains MCMC chain (1000×27×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 5.63 seconds
Compute duration  = 22.32 seconds
parameters        = w0, σ_team, z_team[1], z_team[2], z_team[3], z_team[4], z_team[5], z_team[6], z_team[7], z_team[8], z_team[9], z_team[10], z_team[11]
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics

  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64 

          w0   -0.8495    1.0043    0.0124   6560.0625   2887.0933    1.0011      293.9360
      σ_team    0.5200    0.5209    0.0072   3657.3259   2086.7897    1.0009      163.8734
   z_team[1]   -0.0384    1.0138    0.0122   6858.8605   3176.7331    1.0003      307.3242
   z_team[2]   -0.0136    0.9991    0.0127   6197.2412   2839.4496    1.0012      277.6791
   z_team[3]   -0.0474    1.0145    0.0128   6238.5186   3035.4820    1.0006      279.5286
   z_team[4]    0.0541    1.0054    0.0129   6082.0551   2741.4159    1.0017      272.5179
   z_team[5]   -0.0058    0.9870    0.0114   7528.1796   3152.7275    1.0010      337.3143
   z_team[6]    0.0788    1.0002    0.0117   7255.1485   2914.3775    1.0013      325.0806
   z_team[7]    0.0469    1.0006    0.0112   7986.6588   3148.5158    1.0006      357.8573
   z_team[8]    0.0254    1.0165    0.0129   6201.9859   2811.3218    0.9994      277.8917
   z_team[9]    0.0700    1.0155    0.0131   5947.6023   2738.0289    0.9999      266.4935
  z_team[10]   -0.1032    0.9944    0.0121   6806.5016   2858.5586    0.9999      304.9781
  z_team[11]   -0.0270    1.0163    0.0127   6487.3593   2218.0393    1.0021      290.6783


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

          w0   -2.7926   -1.5285   -0.8407   -0.1531    1.0384
      σ_team    0.0119    0.1472    0.3609    0.7167    1.9321
   z_team[1]   -2.0431   -0.7098   -0.0351    0.6377    1.9121
   z_team[2]   -1.9659   -0.6924    0.0077    0.6341    1.9418
   z_team[3]   -2.0966   -0.7128   -0.0206    0.6164    1.9452
   z_team[4]   -1.9043   -0.6404    0.0665    0.7481    1.9923
   z_team[5]   -1.9608   -0.6651   -0.0085    0.6511    1.9625
   z_team[6]   -1.8720   -0.5988    0.0759    0.7734    2.0176
   z_team[7]   -1.8528   -0.6328    0.0507    0.7028    2.0259
   z_team[8]   -1.9059   -0.6682    0.0151    0.7244    1.9890
   z_team[9]   -1.9178   -0.6053    0.0761    0.7078    2.1019
  z_team[10]   -2.0558   -0.7736   -0.1066    0.5756    1.8537
  z_team[11]   -2.0569   -0.7264   -0.0191    0.6743    1.9156
=#
    next_match_teams = select(ds.matches, :match_id, :home_team, :away_team)
    next_matrix = innerjoin(next_matrix, next_match_teams, on=:match_id)

# Extract the mean parameters from the hierarchical chain
w0_mean = mean(chain_hierarchical[:w0])
σ_team_mean = mean(chain_hierarchical[:σ_team])
z_team_means = [mean(chain_hierarchical[Symbol("z_team[$i]")]) for i in 1:total_teams]

# Calculate the hierarchical blended prob for the next matches
hierarchical_blended_probs = Float64[]

for row in eachrow(next_matrix)
    # Get the team indices (default to 0 if not found)
    h_idx = get(team_map, row.home_team, 0)
    a_idx = get(team_map, row.away_team, 0)
    
    # If the team was in the training set, apply their effect, otherwise 0
    h_effect = h_idx > 0 ? z_team_means[h_idx] : 0.0
    a_effect = a_idx > 0 ? z_team_means[a_idx] : 0.0
    
    # Calculate this specific match's trust
    η = w0_mean + σ_team_mean * (h_effect + a_effect)
    w_hierarchical = logistic(η)
    
    # Blend it!
    p_tilde_hier = w_hierarchical * row.model_prob + (1.0 - w_hierarchical) * row.market_devig
    push!(hierarchical_blended_probs, p_tilde_hier)
end

# Add it to our comparison matrix
next_matrix.hierarchical_blended_prob = hierarchical_blended_probs

# Let's see how much the probability actually changed!
next_matrix.prob_difference = next_matrix.hierarchical_blended_prob .- next_matrix.blended_prob

display(select(next_matrix, :match_id, :blended_prob, :hierarchical_blended_prob, :prob_difference))

display(select(next_matrix, 
    :match_id, 
    :home_team, 
    :away_team, 
    :is_winner, 
    :blended_prob, 
    :hierarchical_blended_prob, 
    :prob_difference
))




    # 1. Neutral Prior (Assume model and market are equally good by default)
    @model function neutral_trust(is_winner, p_model, q_market)
        w_raw ~ Normal(0.0, 1.0) # Changed from -1.0
        w_l = logistic(w_raw)
        p_tilde = clamp.(w_l .* p_model .+ (1.0 - w_l) .* q_market, 1e-6, 1.0 - 1e-6)
        Turing.@addlogprob! sum(logpdf.(Bernoulli.(p_tilde), is_winner))
    end

    # 2. Optimistic Prior (Assume model is usually right)
    @model function optimistic_trust(is_winner, p_model, q_market)
        w_raw ~ Normal(-0.646, 1.0) # Changed from -1.0
        w_l = logistic(w_raw)
        p_tilde = clamp.(w_l .* p_model .+ (1.0 - w_l) .* q_market, 1e-6, 1.0 - 1e-6)
        Turing.@addlogprob! sum(logpdf.(Bernoulli.(p_tilde), is_winner))
    end

    # Extract pure vectors for the models
    is_win_vec = Vector{Int}(df_over25.is_winner)
    p_mod_vec  = Vector{Float64}(df_over25.model_prob)
    q_mkt_vec  = Vector{Float64}(df_over25.market_devig)

    println("Sampling Neutral Prior...")
    chain_neutral = sample(neutral_trust(is_win_vec, p_mod_vec, q_mkt_vec), NUTS(500, 0.65), MCMCThreads(), 1000, 4)

    println("Sampling Optimistic Prior...")
    chain_optimistic = sample(optimistic_trust(is_win_vec, p_mod_vec, q_mkt_vec), NUTS(500, 0.65), MCMCThreads(), 1000, 4)

    # Compare the results!
    w_neutral = logistic.(vec(Array(chain_neutral[:w_raw])))
    w_optimistic = logistic.(vec(Array(chain_optimistic[:w_raw])))

    println("=========================================")
    println("PESSIMISTIC PRIOR (-1.0): Mean Trust = 0.334")
    println("NEUTRAL PRIOR (0.0):      Mean Trust = ", round(mean(w_neutral), digits=3))
    println("OPTIMISTIC PRIOR (1.0):   Mean Trust = ", round(mean(w_optimistic), digits=3))
    println("=========================================")


  # ### Why CLV is the Ultimate Prior
  # As we saw, a settled match is just one single "noisy bit" of information (Win = 1, Loss = 0). It takes thousands of matches for the likelihood to overcome the prior.
  # But Closing Line Value (CLV) is a continuous signal. If your model differs from the Opening odds, and by kickoff the Closing odds have moved toward your model, that is massive, continuous proof that your
  # model contains genuine information.
  #
  # The textbook lays out a beautiful two-stage design:
  #
  # 1. CLV → Prior: We run a simple regression. How much does the model's deviation from the Open predict the market's movement to the Close? We use that regression coefficient to dynamically set the prior mean
  # ( w0_mean ) instead of hardcoding  -1.0  or  0.0 .
  # 2. Outcomes → Likelihood: The MCMC model still uses the actual match outcomes (Wins/Losses) to fit the final posterior.
  #
  # Priors move fast (CLV), Likelihoods stay honest (Outcomes).
  #
  # Because we have  ds.odds , we actually have both  prob_fair_open  and  prob_fair_close . We can calculate this right now in your REPL:

    # 1. Grab the opening probabilities too
    odds_clv_features = select(past_odds, 
        :match_id, 
        :selection, 
        :is_winner, 
        :prob_fair_open,
        :prob_fair_close
    )

    # Join with model predictions
    clv_matrix = innerjoin(odds_clv_features, model_preds, on=[:match_id, :selection])

    # 2. Calculate the Deviations
    # How much did the model disagree with the OPENING market?
    model_deviation = clv_matrix.model_prob .- clv_matrix.prob_fair_open

    # How much did the market move from OPEN to CLOSE?
    market_movement = clv_matrix.prob_fair_close .- clv_matrix.prob_fair_open

    # 3. Simple Linear Regression (Slope = Covariance / Variance)
    # If slope > 0, the market is moving TOWARD our model!
    covariance = cov(model_deviation, market_movement)
    variance   = var(model_deviation)
    beta_clv   = covariance / variance

    println("CLV Beta: ", round(beta_clv, digits=3))

    # 4. Map the Beta to a Prior Mean
    # If Beta is 0 (no CLV), prior is pessimistic (-1.0).
    # If Beta is large (strong CLV), prior becomes optimistic.
    dynamic_prior_mean = -1.0 + (beta_clv * 5.0) # Scaling factor depends on beta size
    println("Dynamic Prior Mean: ", round(dynamic_prior_mean, digits=3))

#=
julia> println("CLV Beta: ", round(beta_clv, digits=3))
CLV Beta: 0.071

julia> dynamic_prior_mean = -1.0 + (beta_clv * 5.0) # Scaling factor depends on beta size
-0.645682391003497

julia> println("Dynamic Prior Mean: ", round(dynamic_prior_mean, digits=3))
Dynamic Prior Mean: -0.646
=#

    # 1. The Updated Engine with the CLV Prior Argument
    @model function hierarchical_trust_with_clv(
        is_winner::Vector{Int}, 
        p_model::Vector{Float64}, 
        q_market::Vector{Float64},
        home_idx::Vector{Int},
        away_idx::Vector{Int},
        n_teams::Int,
        clv_prior_mean::Float64 # <--- Our new dynamic input!
    )
        # 1. Global trust is now anchored by the market's respect for our model (CLV)
        w0 ~ Normal(clv_prior_mean, 1.0)
        
        # 2. Hierarchical spread
        σ_team ~ Exponential(0.5)
        z_team ~ filldist(Normal(0, 1), n_teams)
        
        # 3. Extract team effects safely
        home_effects = view(z_team, home_idx)
        away_effects = view(z_team, away_idx)
        
        # 4. Calculate logit-trust
        η = w0 .+ σ_team .* (home_effects .+ away_effects)
        w_l_vector = logistic.(η)
        
        # 5. Blend and evaluate likelihood against the HONEST outcomes
        p_tilde = clamp.(w_l_vector .* p_model .+ (1.0 .- w_l_vector) .* q_market, 1e-6, 1.0 - 1e-6)
        ll = logpdf.(Bernoulli.(p_tilde), is_winner)
        Turing.@addlogprob! sum(ll)
    end

    # 2. The Updated Builder
    function build_ultimate_model(df::AbstractDataFrame, total_teams::Int, clv_mean::Float64)
        is_winner = Vector{Int}(df.is_winner)
        p_model   = Vector{Float64}(df.model_prob)
        q_market  = Vector{Float64}(df.market_devig)
        home_idx  = Vector{Int}(df.home_team_index)
        away_idx  = Vector{Int}(df.away_team_index)
        
        return hierarchical_trust_with_clv(is_winner, p_model, q_market, home_idx, away_idx, total_teams, clv_mean)
    end



    ultimate_model = build_ultimate_model(df_over25, total_teams, dynamic_prior_mean)

    println("Sampling the Ultimate Model...")
    chain_ultimate = sample(
        ultimate_model, 
        NUTS(n_warmup, accept_rate; max_depth=max_depth), 
        MCMCThreads(), 
        n_samples, 
        n_chains
    )

    describe(chain_ultimate)

#=
julia> describe(chain_ultimate)
Chains MCMC chain (1000×27×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 3.63 seconds
Compute duration  = 14.15 seconds
parameters        = w0, σ_team, z_team[1], z_team[2], z_team[3], z_team[4], z_team[5], z_team[6], z_team[7], z_team[8], z_team[9], z_team[10], z_team[11]
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics

  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64 

          w0   -0.4707    0.9807    0.0119   6805.7394   3041.2138    1.0005      481.0390
      σ_team    0.5139    0.4960    0.0076   4296.8834   2686.2880    1.0003      303.7096
   z_team[1]   -0.0397    1.0023    0.0117   7271.0744   3019.5189    1.0014      513.9295
   z_team[2]   -0.0054    0.9921    0.0121   6701.7053   3080.3351    1.0018      473.6857
   z_team[3]   -0.0319    0.9766    0.0114   7326.0221   2705.0310    1.0006      517.8133
   z_team[4]    0.0750    1.0154    0.0129   6207.8793   2619.7614    1.0008      438.7814
   z_team[5]   -0.0053    1.0316    0.0132   6110.6250   2970.6031    1.0013      431.9073
   z_team[6]    0.0851    0.9750    0.0127   5959.5445   2875.3846    1.0012      421.2288
   z_team[7]    0.0616    0.9948    0.0116   7273.5242   2572.0943    1.0050      514.1026
   z_team[8]    0.0084    0.9689    0.0110   7771.5808   3147.0120    1.0013      549.3060
   z_team[9]    0.0765    1.0180    0.0134   5789.6909   2691.9052    1.0007      409.2233
  z_team[10]   -0.1110    1.0018    0.0134   5616.5440   2644.2177    1.0024      396.9850
  z_team[11]   -0.0241    1.0146    0.0125   6626.7879   3161.3560    1.0039      468.3904


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

          w0   -2.4451   -1.1437   -0.4476    0.1915    1.4567
      σ_team    0.0172    0.1547    0.3637    0.7205    1.8603
   z_team[1]   -2.0155   -0.7273   -0.0374    0.6443    1.8936
   z_team[2]   -1.9652   -0.6517   -0.0066    0.6490    2.0001
   z_team[3]   -1.9140   -0.7013   -0.0402    0.6414    1.8081
   z_team[4]   -1.9354   -0.5908    0.0702    0.7504    2.0138
   z_team[5]   -2.0298   -0.7145   -0.0213    0.7035    2.0280
   z_team[6]   -1.8190   -0.5761    0.0903    0.7249    1.9949
   z_team[7]   -1.8833   -0.6091    0.0619    0.7606    1.9490
   z_team[8]   -1.9412   -0.6410    0.0099    0.6533    1.9180
   z_team[9]   -1.9851   -0.6008    0.0870    0.7437    2.0756
  z_team[10]   -2.0581   -0.7994   -0.1125    0.5465    1.8925
  z_team[11]   -2.0069   -0.7352   -0.0136    0.6629    1.9115
=#
    # 1. Reverse the team_map so we can look up names by their index
    inverse_team_map = Dict(v => k for (k, v) in team_map)
    
    # 2. Extract the global parameters from the ultimate chain
    w0_mean = mean(chain_ultimate[:w0])
    σ_mean  = mean(chain_ultimate[:σ_team])

    # 3. Create a DataFrame to hold the results
    team_results = DataFrame(
        team_index = Int[],
        team_name  = String[],
        z_score    = Float64[],
        trust_pct  = Float64[]
    )

    for i in 1:total_teams
        # Extract this team's z-score
        z_mean = mean(chain_ultimate[Symbol("z_team[$i]")])
        
        # Calculate the team-specific logit and apply inverse-logit (logistic)
        team_logit = w0_mean + (σ_mean * z_mean)
        team_trust = logistic(team_logit)
        
        # Add to our results
        push!(team_results, (
            i, 
            inverse_team_map[i], 
            z_mean, 
            team_trust * 100.0 # Convert to percentage for readability
        ))
    end

    # 4. Sort by highest trust to see the spread
    sort!(team_results, :trust_pct, rev=true)

    println("=========================================")
    println("ULTIMATE MODEL TEAM SPREAD")
    println("Global Average Trust: ", round(logistic(w0_mean) * 100.0, digits=2), "%")
    println("=========================================")
    display(team_results)




#=
julia> display(team_results)
11×4 DataFrame
 Row │ team_index  team_name             z_score      trust_pct 
     │ Int64       String                Float64      Float64   
─────┼──────────────────────────────────────────────────────────
   1 │          6  galway-united          0.0850973     39.4853
   2 │          9  sligo-rovers           0.0764579     39.3793
   3 │          4  drogheda-united        0.0750039     39.3614
   4 │          7  shamrock-rovers        0.0616135     39.1973
   5 │          8  shelbourne             0.00843557    38.5479
   6 │          5  dundalk-fc            -0.0053027     38.3808
   7 │          2  cork-city             -0.0054427     38.3791
   8 │         11  waterford-fc          -0.024131      38.1522
   9 │          3  derry-city            -0.0318538     38.0586
  10 │          1  bohemian              -0.0396609     37.964
  11 │         10  st-patricks-athletic  -0.111037      37.104
=#


    # 1. Ensure clv_matrix has team names
    if !hasproperty(clv_matrix, :home_team)
        clv_matrix = innerjoin(clv_matrix, match_teams, on=:match_id)
    end
    
    team_clv_priors = zeros(Float64, total_teams)
    
    # 2. Run the regression for every team
    for i in 1:total_teams
        team_name = inverse_team_map[i]
        
        # Filter to matches where this team played (home or away)
        team_matches = subset(clv_matrix, [:home_team, :away_team] => ByRow((h, a) -> h == team_name || a == team_name))
        
        if nrow(team_matches) > 5 # Need at least a few matches to run a regression
            t_model_dev = team_matches.model_prob .- team_matches.prob_fair_open
            t_mkt_move  = team_matches.prob_fair_close .- team_matches.prob_fair_open
            
            v = var(t_model_dev)
            beta = v > 1e-8 ? cov(t_model_dev, t_mkt_move) / v : 0.0
            
            # Scale the beta (same * 5.0 factor as before)
            team_clv_priors[i] = beta * 5.0
        end
    end

    # 3. Center the priors around 0 so w0 acts as the global mean
    team_clv_priors .-= mean(team_clv_priors)


#=
julia> team_clv_priors .-= mean(team_clv_priors)
11-element Vector{Float64}:
  0.0024415698680460007
 -0.2201758770871004
  0.10991004326887122
  0.10789777750334167
 -0.07395407760580064
  0.15681008077467945
 -0.00982471910263516
  0.24872823929711135
 -0.19320287773018638
 -0.07087920288405353
 -0.05775095630227334
=#

    println("Calculated Per-Team CLV Priors!")



    @model function hierarchical_trust_team_clv(
        is_winner::Vector{Int}, 
        p_model::Vector{Float64}, 
        q_market::Vector{Float64},
        home_idx::Vector{Int},
        away_idx::Vector{Int},
        n_teams::Int,
        global_clv_mean::Float64,
        team_clv_priors::Vector{Float64} # <--- NEW: Vector of priors
    )
        # Global trust
        w0 ~ Normal(global_clv_mean, 1.0)
        
        # Hierarchical spread
        σ_team ~ Exponential(0.5)
        
        # Raw team deviations
        z_team ~ filldist(Normal(0, 1), n_teams)
        
        # AD-SAFE Extraction
        home_z = view(z_team, home_idx)
        away_z = view(z_team, away_idx)
        
        # Extract the hard-coded CLV priors for the matches
        home_prior = view(team_clv_priors, home_idx)
        away_prior = view(team_clv_priors, away_idx)
        
        # Calculate logit-trust: 
        # w0 + (Prior + z_score) * σ
        η = w0 .+ σ_team .* (home_prior .+ home_z .+ away_prior .+ away_z)
        
        w_l_vector = logistic.(η)
        
        p_tilde = clamp.(w_l_vector .* p_model .+ (1.0 .- w_l_vector) .* q_market, 1e-6, 1.0 - 1e-6)
        ll = logpdf.(Bernoulli.(p_tilde), is_winner)
        Turing.@addlogprob! sum(ll)
    end

    # Sample it!
    is_win_vec = Vector{Int}(df_over25.is_winner)
    p_mod_vec  = Vector{Float64}(df_over25.model_prob)
    q_mkt_vec  = Vector{Float64}(df_over25.market_devig)
    h_idx_vec  = Vector{Int}(df_over25.home_team_index)
    a_idx_vec  = Vector{Int}(df_over25.away_team_index)

    per_team_model = hierarchical_trust_team_clv(
        is_win_vec, p_mod_vec, q_mkt_vec, h_idx_vec, a_idx_vec, 
        total_teams, dynamic_prior_mean, team_clv_priors
    )

    println("Sampling the Per-Team CLV Model...")
    chain_per_team = sample(per_team_model, NUTS(n_warmup, accept_rate; max_depth=max_depth), MCMCThreads(), n_samples, n_chains)
    describe(chain_per_team)

#=
julia> describe(chain_per_team)
Chains MCMC chain (1000×27×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 4.2 seconds
Compute duration  = 15.42 seconds
parameters        = w0, σ_team, z_team[1], z_team[2], z_team[3], z_team[4], z_team[5], z_team[6], z_team[7], z_team[8], z_team[9], z_team[10], z_team[11]
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics

  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64 

          w0   -0.4672    0.9843    0.0146   4472.7775   3110.5646    1.0014      290.0446
      σ_team    0.5093    0.4886    0.0077   3606.0491   2360.9812    1.0002      233.8402
   z_team[1]   -0.0184    1.0063    0.0138   5283.9512   2837.5012    1.0004      342.6465
   z_team[2]    0.0083    1.0222    0.0148   4770.5799   2820.9466    1.0006      309.3561
   z_team[3]   -0.0359    1.0122    0.0138   5339.8230   2921.2002    1.0010      346.2696
   z_team[4]    0.0818    0.9858    0.0146   4520.6960   2417.5305    1.0011      293.1519
   z_team[5]   -0.0174    1.0199    0.0132   5999.7261   2817.4557    0.9998      389.0621
   z_team[6]    0.0870    0.9704    0.0125   6002.0702   2964.1224    1.0012      389.2141
   z_team[7]    0.0665    1.0033    0.0136   5448.0181   2685.8793    1.0021      353.2857
   z_team[8]    0.0264    1.0072    0.0131   5884.9227   2879.5251    1.0011      381.6175
   z_team[9]    0.0879    1.0159    0.0137   5524.9565   2545.2281    1.0010      358.2749
  z_team[10]   -0.1025    0.9889    0.0132   5612.9041   2394.2216    1.0013      363.9780
  z_team[11]   -0.0145    0.9807    0.0138   5073.0978   2741.2498    1.0003      328.9733


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

          w0   -2.4200   -1.1033   -0.4679    0.1986    1.4519
      σ_team    0.0161    0.1561    0.3651    0.6970    1.8383
   z_team[1]   -1.9911   -0.6753   -0.0176    0.6572    1.9035
   z_team[2]   -2.0098   -0.6813    0.0006    0.7000    2.0070
   z_team[3]   -1.9928   -0.7092   -0.0482    0.6298    1.9220
   z_team[4]   -1.8258   -0.5883    0.0867    0.7598    2.0478
   z_team[5]   -1.9688   -0.7038   -0.0231    0.6729    1.9795
   z_team[6]   -1.8324   -0.5736    0.1097    0.7508    1.9705
   z_team[7]   -1.9056   -0.6080    0.0613    0.7367    2.0394
   z_team[8]   -1.9771   -0.6426    0.0077    0.7200    1.9973
   z_team[9]   -1.8979   -0.5995    0.0819    0.7963    2.0396
  z_team[10]   -2.0800   -0.7544   -0.0816    0.5520    1.8828
  z_team[11]   -1.9364   -0.6782   -0.0275    0.6381    1.9259
=#
    # Create a DataFrame for the new per-team results
    per_team_results = DataFrame(
        team_index = Int[],
        team_name  = String[],
        per_team_trust_pct = Float64[]
    )

    w0_mean_pt = mean(chain_per_team[:w0])
    σ_mean_pt  = mean(chain_per_team[:σ_team])

    for i in 1:total_teams
        z_mean_pt = mean(chain_per_team[Symbol("z_team[$i]")])
        
	# Calculate the total effect: (Prior + Z_score)
        team_total_effect = team_clv_priors[i] + z_mean_pt
        
	# Calculate logit and trust
        team_logit = w0_mean_pt + (σ_mean_pt * team_total_effect)
        team_trust = logistic(team_logit)
        
        push!(per_team_results, (i, inverse_team_map[i], team_trust * 100.0))
    end

    # Join with the old team_results we generated earlier
    comparison_df = innerjoin(team_results, per_team_results, on=[:team_index, :team_name])

    # Calculate the difference
    comparison_df.diff_pct = comparison_df.per_team_trust_pct .- comparison_df.trust_pct

    sort!(comparison_df, :per_team_trust_pct, rev=true)

    println("=========================================")
    println("TRUST PERCENTAGE COMPARISON")
    println("=========================================")
    display(comparison_df)

#=
11×6 DataFrame
 Row │ team_index  team_name             z_score      trust_pct  per_team_trust_pct  diff_pct   
     │ Int64       String                Float64      Float64    Float64             Float64    
─────┼──────────────────────────────────────────────────────────────────────────────────────────
   1 │          8  shelbourne             0.00843557    38.5479             41.8951   3.34717
   2 │          6  galway-united          0.0850973     39.4853             41.5081   2.02277
   3 │          4  drogheda-united        0.0750039     39.3614             40.8402   1.47879
   4 │          3  derry-city            -0.0318538     38.0586             39.4248   1.36622
   5 │          7  shamrock-rovers        0.0616135     39.1973             39.2139   0.0165784
   6 │          1  bohemian              -0.0396609     37.964              38.3365   0.372465
   7 │         11  waterford-fc          -0.024131      38.1522             37.6601  -0.49211
   8 │          5  dundalk-fc            -0.0053027     38.3808             37.4327  -0.948089
   9 │          9  sligo-rovers           0.0764579     39.3793             37.2657  -2.11359
  10 │         10  st-patricks-athletic  -0.111037      37.104              36.4588  -0.645226
  11 │          2  cork-city             -0.0054427     38.3791             36.0062  -2.37287
=#

    w0_pt = mean(chain_per_team[:w0])
    σ_pt  = mean(chain_per_team[:σ_team])

    pt_blended_probs = Float64[]

    for row in eachrow(next_matrix)
  	# Get indices (0 if the team is completely new and unseen)
	h_idx = get(team_map, row.home_team, 0)
	a_idx = get(team_map, row.away_team, 0)
	
	# Calculate Home Effect: (Prior + z_score)
	h_effect = 0.0
	if h_idx > 0
	   h_effect = team_clv_priors[h_idx] + mean(chain_per_team[Symbol("z_team[$h_idx]")])
	end
	
	# Calculate Away Effect: (Prior + z_score)
	a_effect = 0.0
	if a_idx > 0
	   a_effect = team_clv_priors[a_idx] + mean(chain_per_team[Symbol("z_team[$a_idx]")])
	end
	
	# Calculate final logit and trust
	η = w0_pt + σ_pt * (h_effect + a_effect)
	w_pt = logistic(η)
	
	# Blend!
        p_tilde_pt = w_pt * row.model_prob + (1.0 - w_pt) * row.market_devig
        push!(pt_blended_probs, p_tilde_pt)
    end

    # Add the new probabilities to the dataframe
    next_matrix.per_team_clv_blended_prob = pt_blended_probs

    # Calculate the difference from the global hierarchical model
    next_matrix.model_diff = next_matrix.per_team_clv_blended_prob .- next_matrix.hierarchical_blended_prob

    println("======================================================")
    println("OUT-OF-SAMPLE COMPARISON: Global CLV vs Per-Team CLV")
    println("======================================================")

    display(select(next_matrix, 
        :match_id, 
        :home_team, 
        :away_team, 
        :is_winner,
        :hierarchical_blended_prob, 
        :per_team_clv_blended_prob, 
        :model_diff
    ))


1/ 0.454792
1/0.459725




    # 1. Get the raw L1 distribution arrays for the out-of-sample matches
    next_model_raw = subset(next_matches_inference, :selection => ByRow(==(:over_25)))
    match_to_l1_dist = Dict(r.match_id => r.distribution for r in eachrow(next_model_raw))
    
    # 2. Extract the full chains for the global parameters
    w0_draws = vec(Array(chain_per_team[:w0]))
    σ_draws  = vec(Array(chain_per_team[:σ_team]))

    dist_staking_results = Float64[]

    for row in eachrow(next_matrix)
	# The L1 draws for this match
        l1_draws = match_to_l1_dist[row.match_id]
        
        q_mkt = row.market_devig
        odds  = row.odds_close
        
        h_idx = get(team_map, row.home_team, 0)
        a_idx = get(team_map, row.away_team, 0)
        
	# We pair up the draws (using the minimum length just in case they differ)
        n_samples = min(length(w0_draws), length(l1_draws))
        
	# We will compute the Kelly stake for EVERY draw and average them
        draw_stakes = Float64[]
        
        for s in 1:n_samples
            # 1. Get the global parameters for THIS draw
            w0_s = w0_draws[s]
            σ_s  = σ_draws[s]
            
            # 2. Get the specific team effects for THIS draw
            # (We use get() on the DataFrame column to safely pull the s-th element)
            h_z_s = h_idx > 0 ? chain_per_team[Symbol("z_team[$h_idx]")][s] : 0.0
            a_z_s = a_idx > 0 ? chain_per_team[Symbol("z_team[$a_idx]")][s] : 0.0
            
            h_prior = h_idx > 0 ? team_clv_priors[h_idx] : 0.0
            a_prior = a_idx > 0 ? team_clv_priors[a_idx] : 0.0
            
            # 3. Calculate trust for this specific draw
            η_s = w0_s + σ_s * (h_prior + h_z_s + a_prior + a_z_s)
            w_s = logistic(η_s)
            
            # 4. Blend using the specific L1 draw and the specific L2 trust draw!
            p_tilde_s = w_s * l1_draws[s] + (1.0 - w_s) * q_mkt
            
            # 5. Calculate Kelly stake for this draw: f* = (p * b - (1-p)) / b
            b = odds - 1.0
            stake_s = (p_tilde_s * (b + 1.0) - 1.0) / b
            
            # Floor at 0 (we only place back bets)
            push!(draw_stakes, max(0.0, stake_s))
        end
        
	# The final action is the average of the stakes!
        final_stake = mean(draw_stakes)
        push!(dist_staking_results, final_stake)
    end

    # Add the final safe stakes to the matrix
    next_matrix.distributional_stake = dist_staking_results

    println("======================================================")
    println("DISTRIBUTIONAL STAKING (Baker-McHale & Gelman Method)")
    println("======================================================")

    display(select(next_matrix, 
        :match_id, 
        :home_team, 
        :away_team, 
        :is_winner,
        :distributional_stake
    ))



  composed_distributions = Vector{Float64}[]

    for row in eachrow(next_matrix)
        l1_draws = match_to_l1_dist[row.match_id]
        q_mkt = row.market_devig
        
        h_idx = get(team_map, row.home_team, 0)
        a_idx = get(team_map, row.away_team, 0)
        
        n_samples = min(length(w0_draws), length(l1_draws))
        
  	# Store the fully composed distribution here!
        match_composed_dist = Float64[]
        
        for s in 1:n_samples
            w0_s = w0_draws[s]
            σ_s  = σ_draws[s]
            
            h_z_s = h_idx > 0 ? chain_per_team[Symbol("z_team[$h_idx]")][s] : 0.0
            a_z_s = a_idx > 0 ? chain_per_team[Symbol("z_team[$a_idx]")][s] : 0.0
            
            h_prior = h_idx > 0 ? team_clv_priors[h_idx] : 0.0
            a_prior = a_idx > 0 ? team_clv_priors[a_idx] : 0.0
            
            η_s = w0_s + σ_s * (h_prior + h_z_s + a_prior + a_z_s)
            w_s = logistic(η_s)
            
            # This is the composed L1 + L2 probability!
            p_tilde_s = w_s * l1_draws[s] + (1.0 - w_s) * q_mkt
            push!(match_composed_dist, p_tilde_s)
        end
        
        push!(composed_distributions, match_composed_dist)
    end

    # Save the full distribution to the DataFrame
    next_matrix.composed_distribution = composed_distributions

    println("======================================================")
    println("COMPOSED POSTERIOR UNCERTAINTY (L1 + L2)")
    println("======================================================")

    # Let's print the mean and standard deviation for the first few matches
    for i in 1:5
        row = next_matrix[i, :]
        comp_dist = row.composed_distribution
        l1_dist   = match_to_l1_dist[row.match_id]
        
        println("Match: ", row.home_team, " vs ", row.away_team)
        println("  Market Prob:     ", round(row.market_devig, digits=4))
        println("  L1 Mean Prob:    ", round(mean(l1_dist), digits=4))
        println("  L1 Std Dev:      ", round(std(l1_dist), digits=4))
        println("  Composed Mean:   ", round(mean(comp_dist), digits=4))
        println("  Composed Std:    ", round(std(comp_dist), digits=4))
        println("-------------------------------------------------")
    end
