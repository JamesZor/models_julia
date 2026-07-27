# current_development/trust_model_dev/r01_data_pipe.jl

using Revise
using BayesianFootball
using DataFrames
using Statistics

# Load the loader file containing the monkey-patched extractors
include("l01_basic_main.jl")

const Data = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments
const Features = BayesianFootball.Features
const Predictions = BayesianFootball.Predictions
const Markets = BayesianFootball.Data.Markets

println("1. Loading Datastore...")
ds = Data.load_datastore_cached(Data.Ireland())

println("2. Loading L1 Experiment & Generating Predictions (PPD)...")
src_dir = "./data/double_poisson_smile_src_grid/"
list_of_experiments = Experiments.list_experiments(src_dir, data_dir="")
# Loading experiment index 5 as a realistic L1 baseline
l1_experiment = Experiments.load_experiment(list_of_experiments, 5)

# Generate the PPD DataFrame using the standard inference pipeline
l1_ppd = Predictions.model_inference(ds, l1_experiment)
l1_ppd_df = l1_ppd.df

println("3. Creating Trust Config...")
using BayesianFootball.Data.Markets

const TRUST_MARKETS = AbstractMarket[
    Market1X2(),
    MarketBTTS(),
    MarketOverUnder(0.5),
    MarketOverUnder(1.5),
    MarketOverUnder(2.5),
    MarketOverUnder(3.5),
    MarketOverUnder(4.5)
]

# Provide the exact markets to the model configuration
model_config = TrustModelConfig(
    market_config = BayesianFootball.Data.Markets.MarketConfig(TRUST_MARKETS)
)

println("4. Generating a Split Boundary...")
# We use the EXACT SAME splitter that the L1 experiment used to guarantee identical folds!
splitter = l1_experiment.config.splitter
splits = Data.create_id_boundaries(ds, splitter)
boundary = splits[20][1] 
ordered_ids = vcat(boundary.history_match_ids, boundary.target_match_ids)

println("5. Building Native Team Map...")
matches_df = subset(ds.matches, :match_id => ByRow(in(ordered_ids)))
all_teams = unique(vcat(matches_df.home_team, matches_df.away_team))
team_map = Dict(name => i for (i, name) in enumerate(sort(all_teams)))

println("6. Testing Custom Extractors...")

trust_features_list = [
    TrustDatasetFeature(l1_ppd_df, TRUST_MARKETS) 
]

F_data = Dict{Symbol, Any}()
F_data[:n_teams] = length(team_map)

# Extract EVERYTHING safely joined!
Features.add_feature!(F_data, trust_features_list[1], ordered_ids, team_map, ds)


println("\n=============================================")
println("✅ FEATURE EXTRACTION SUCCESSFUL")
println("=============================================")
println("Keys in F_data: ", keys(F_data))
println("p_model size:           ", length(F_data[:p_model]))
println("q_market size:          ", length(F_data[:q_market]))
println("is_winner size:         ", length(F_data[:is_winner]))
println("market_index size:      ", length(F_data[:market_index]))
println("home_team_index size:   ", length(F_data[:home_team_index]))
println("away_team_index size:   ", length(F_data[:away_team_index]))

println("\n=============================================")
println("📊 Market Extraction Summary")
println("=============================================")
total_rows = length(F_data[:p_model])
println("Total rows extracted: $total_rows")

# Group by market name to prove we got everything!
unique_markets = unique(F_data[:market_name_str])
for m_name in unique_markets
    # Find all rows for this market
    mask = F_data[:market_name_str] .== m_name
    count = sum(mask)
    
    # Grab the assigned parent index
    first_idx = findfirst(mask)
    m_idx = F_data[:market_index][first_idx]
    
    println("Market: $(rpad(m_name, 10)) | Parent Idx: $(rpad(m_idx, 2)) | Total Matches: $count")
end

println("\nAlignment Check (Randomly sampling 5 diverse rows):")
using Random
Random.seed!(42)
sample_indices = rand(1:total_rows, 5)

for i in sample_indices
    m_idx = F_data[:market_index][i]
    m_name = F_data[:market_name_str][i]
    h_idx = F_data[:home_team_index][i]
    a_idx = F_data[:away_team_index][i]
    p_val = round(F_data[:p_model][i], digits=3)
    q_val = round(F_data[:q_market][i], digits=3)
    won = F_data[:is_winner][i]
    
    println("Row $(rpad(i,4)) | Market: $(rpad(m_name, 10)) (Idx: $m_idx) | HomeIdx: $(rpad(h_idx,2)) | AwayIdx: $(rpad(a_idx,2)) | L1 p: $(rpad(p_val,5)) | L0 q: $(rpad(q_val,5)) | won: $won")
end
println("=============================================")
