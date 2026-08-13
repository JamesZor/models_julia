using BayesianFootball
using DataFrames, Dates, Statistics, Printf

const PF = BayesianFootball.Portfolio       # the module under demonstration
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments
const BT = BayesianFootball.BackTesting


# 1.Get the data for one match.
# 2. Do the math ( run the score matrix and kelly solver 
# 3. pack the Box ( put the results ino the matchbook 
#


# load datastore and experiment results
ds = DD.load_datastore_cached(DD.ScottishLower())

expr = EE.load_experiment(
       EE.list_experiments("./data/experiments/plus_minus_biweek", data_dir = ""),
       3)


# get the latents for the models for each match
latents = EE.extract_oos_predictions(ds, expr)

#++++++++++++++++++++++++++++++++++++++++ 
## Step 1 --- Grab the data for excatly one match
#++++++++++++++++++++++++++++++++++++++++ 
  # 1 .pick a row index as the arbitrary match
row_idx = 1 
m_id = latents.df[row_idx, :].match_id

  # 2. Look up match details from the datastore ( no fixtures need) 
match_row = first(subset(ds.matches, :match_id => ByRow(isequal(m_id))))
match_date = Date(match_row.match_date)

  # 3. Grab the odds for this match - betfair historic
odds_df = DD.summarize_betfair_market(
    ds,
    open_window=(-100000.0, -10.0),
    close_window=(-20.0, 0.0)
)
match_odds = subset(odds_df, :match_id => ByRow(isequal(m_id)))


# 4. print 
function step_one_display_print(match_row, match_date, match_odds)
    println("\n=== STEP 1 COMPLETE ===")
    println("Match: ", match_row.home_team, " vs ", match_row.away_team)
    println("Date:  ", match_date)
    println("Odds:  Found $(nrow(match_odds)) selections for this match.")
    println("=======================\n")
end

step_one_display_print(match_row, match_date, match_odds)


#++++++++++++++++++++++++++++++++++++++++ 
## Step 2 --- The Maths ( Score matrix and market probabilities) 
#++++++++++++++++++++++++++++++++++++++++ 

  # 1. Extract the latent parameters for this match 
param = Predictions.extract_params(latents.model, latents.df[row_idx, :])

  # 2. Compute the score matrix 
score_matrix = Predictions.compute_score_matrix(latents.model, param)
    
    # peek at the probability of a 1-1 draw 
    p_1_1 = mean(score_matrix.data[2,2,:]) 
    p_1_1_median = median(score_matrix.data[2,2,:]) 
#=
julia> p_1_1 = mean(score_matrix.data[2,2,:])
0.11069518157126589

julia> p_1_1_median = median(score_matrix.data[2,2,:])
0.11121948186556027
=#

  # 3. Define the markets we care about 
markets_config = DD.MarketConfig([DD.Market1X2(), DD.MarketBTTS(), DD.MarketOverUnder(0.5), DD.MarketOverUnder(1.5)]) 

  # 4. we create a minimal book spec
spec = PF.BookSpec(markets = markets_config)
#=
BookSpec  → determines a MatchBook. THIS IS THE CACHE KEY.
  ├── markets        2: 1X2, BTTS
  ├── price          DeArb
  ├── allocator      KellyLogUtility
  ├── shrink         BakerMcHale(n_draws=128, grid=[0.0 … 1.0] (51 pts), seed=20260805)
  ├── commission     PerBetCommission(rate=0.02)
  ├── budget         0.99
  └── cache key      cc34ed68eab4b6f5
  change any of these and 600+ books must be rebuilt (~26s).
=#

  # 4. Compute the model's true probability for every market in our config
model_probs = Dict(
    string(m) => Predictions.compute_market_probs(score_matrix, m)
    for m in spec.markets.markets
)

#=
Dict{String, Dict{Symbol, Vector{Float64}}} with 4 entries:
  "Market[1X2]"     => Dict(:away=>[0.261695, 0.252517, 0.246793, 0.239813, 0.30078, 0.279295, 0.296617, 0.316142, 0.217865, 0.206274  …  0.278132, 0.244689, 0.225575, 0.303952, 0.202921, 0.287268, 0.264621, 0.…
  "Market[BTTS]"    => Dict(:btts_yes=>[0.625296, 0.578373, 0.574737, 0.651691, 0.517109, 0.547321, 0.550288, 0.624026, 0.548438, 0.550169  …  0.523642, 0.586295, 0.58148, 0.585053, 0.564811, 0.549645, 0.621201…
  "Market[O/U 1.5]" => Dict(:over_15=>[0.834231, 0.797024, 0.795678, 0.861032, 0.726467, 0.760737, 0.759746, 0.822399, 0.783451, 0.790692  …  0.737966, 0.806582, 0.809543, 0.79054, 0.805413, 0.76108, 0.83012, 0…
  "Market[O/U 0.5]" => Dict(:under_05=>[0.0390755, 0.0510682, 0.0515214, 0.0310804, 0.0766529, 0.0637581, 0.0641188, 0.0427773, 0.0556999, 0.0532117  …  0.0722266, 0.0478891, 0.0469177, 0.0532639, 0.048274, 0.0…
=#
  # 5. Marry the Model probabilities with the Betfair odds we found in Step 1
    sels = PF.extract_selections(match_odds, m_id, spec, model_probs)
#=
5-element Vector{BayesianFootball.Portfolio.Selection}:
 Selection(1X2_away @ 3.072, p=0.272 vs mkt 0.325)
 Selection(1X2_home @ 2.314, p=0.49 vs mkt 0.432)
 Selection(1X2_draw @ 4.127, p=0.238 vs mkt 0.242)
 Selection(O/U 1.5_over_15 @ 1.185, p=0.787 vs mkt 0.844)
 Selection(O/U 1.5_under_15 @ 6.4, p=0.213 vs mkt 0.156)
=#

#++++++++++++++++++++++++++++++++++++++++ 
## Step 3 ---  The allocator adn shrinkage ( The kelly solver)
#++++++++++++++++++++++++++++++++++++++++ 

  # 1. Flatten the score matrix (12x12) into 144 prob grid
sm_data = Predictions.score_matrix_data(score_matrix)

max_h, max_a, n_samples = size(sm_data)
p_grid = vec(mean(sm_data, dims=3)[:, :, 1])
p_grid ./= sum(p_grid)

  # 2. Build the payoff matrix R ( Rows = Selections, Columns = 144 grid states ) 
R = PF.payoff_matrix(sels, max_h, max_a, spec.exec.commission)


#=
julia> R = PF.payoff_matrix(sels, max_h, max_a, spec.exec.commission)
144×5 Matrix{Float64}:
 -1.0      -1.0       3.06416  -1.0      5.292
 -1.0       1.28758  -1.0      -1.0      5.292
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0      -1.0      5.292
 -1.0      -1.0       3.06416   0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
  ⋮
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
 -1.0      -1.0       3.06416   0.1813  -1.0
 -1.0       1.28758  -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
  2.03078  -1.0      -1.0       0.1813  -1.0
 -1.0      -1.0       3.06416   0.1813  -1.0
=#
res = PF.allocate(spec.allocator, p_grid, R, spec.exec)


  # 4. Calculate parameter-uncertainty shrinkage (if configured)
k = PF.shrink_factor(spec.shrink, score_matrix, R, p_grid, spec.allocator, spec.exec; seed_offset=122221)

function step_3_print_display(sels, res)
    println("\n=== STEP 3 COMPLETE ===")
    println("Newton Solver Converged: ", res.converged)
    for i in 1:length(sels)
        stake_pct = round(res.a[i] * 100, digits=2)
        println(rpad(sels[i].family, 15), " | Raw Kelly Stake: ", stake_pct, "%")
    end
    println("Shrinkage multiplier:    ", round(k, digits=3))
    println("=======================\n")
end 

step_3_print_display(sels, res)




#++++++++++++++++++++++++++++++++++++++++ 
##  Phase 2 ---- Portfolio Risk and Staking ( The slate) 
#++++++++++++++++++++++++++++++++++++++++ 

# Pack the maths that we have done into a MatchBook and put it into a Slate. 
# A slate is jsut a list of matches that settle simultaneously on the same day or time. 

# struct MatchBook
#     m_id::Int
#     date::Date
#     sels::Vector{Selection}
#     p_grid::Vector{Float64}
#     R::Matrix{Float64}
#     settle::Union{Nothing,Vector{Float64}}
#     a_kelly::Vector{Float64}
#     k_shrink::Float64
#     kkt::Float64
#     converged::Bool
# end
#


match_book = PF.MatchBook(
  m_id,
  match_date,
  sels, 
  p_grid,
  R,
  nothing,
  res.a,
  k,
  res.kkt,
  res.converged 
)


#=
"A set of matches that settle together and therefore share one bankroll."
struct Slate
    window::Date
    books::Vector{MatchBook}
end
=#
slate = PF.Slate(
  match_date,
  [match_book] 
)


  # Phase 2 - Step 2 -- Define the bankroll context. 
  #
#=
"Context handed to trust / risk / filter so bankroll- or time-dependent policies are possible."
struct SlateContext
    idx::Int
    date::Date
    bankroll::Float64
end
=#
ctx = PF.SlateContext(
  1,
  match_date,
  1000.00
)

  # Phase 2 - Step 3 -- Define the risk Policy 
  #

#=
"""
    PolicySpec

Everything that is a pure post-multiplier on an already-built book. Free to sweep against a
cached `Vector{MatchBook}`.

Note: `risk` is homogeneous of degree 0 in the stakes it is handed, so once the drawdown
constraint binds, `trust` can only reshape the book -- it cannot rescale it. See `stake_slate`.
"""
Base.@kwdef struct PolicySpec{T<:AbstractTrustModel,
                              R<:AbstractRiskModel,
                              C<:AbstractExposureCap,
                              F<:AbstractSelectionFilter,
                              G<:AbstractSlateGrouping}
    trust::T    = FlatTrust(0.25)
    risk::R     = SlateDrawdown(23.0)
    cap::C      = FixedCap(0.25)
    filter::F   = KeepAll()
    grouping::G = DailySlate()
end
=#
policy = PF.PolicySpec(
  trust = PF.FlatTrust(0.5),
  risk = PF.SlateDrawdown(20.0),
  cap = PF.FixedCap(0.25)
  )


  # Phase 3 - Deconstructing staking slate. 

  # Phase 3 - step 1 - Apply trust and Shrinkage 
  b = slate.books[1]
  raw_stakes = copy(b.a_kelly)
  raw_stakes .*= b.k_shrink 

  # Phase 3 - step 2 - Apply the drawdown budget 
# This looks at the worst-case scenarios across all the matches in the slate 
# If the risking these stakes could cause a 30% drop drawdown, but our policy say max 20%
# it Calculates a risk_multipler to scale down the stakes. 

rets = [ b.R * raw_stakes ]
probs = [b.p_grid]
risk_multiplier = PF.risk_factor(policy.risk, probs, rets)

  # Phase 3 - step 3 Apply the cap 
final_stakes = raw_stakes .* risk_multiplier 
final_stakes, capped = PF.apply_cap(policy.cap, [final_stakes])
final_stakes = final_stakes[1]

function phase3_display(b, raw_stakes, risk_multiplier, final_stakes, capped)
    println("\n=== PHASE 2 COMPLETE ===")                                                                          
    println("1. Raw Kelly Stakes:       ", round.(b.a_kelly .* 100, digits=2), "%")                                
    println("2. After Shrinkage (k):    ", round.(raw_stakes .* 100, digits=2), "%")                               
    println("3. Risk Multiplier applied:", round(risk_multiplier, digits=3))                                       
    println("4. Final Stakes to Place:  ", round.(final_stakes .* 100, digits=2), "%")                             
    println("   Did we hit max cap?     ", capped)                                                                 
    println("=======================\n")    
end 

phase3_display(b, raw_stakes, risk_multiplier, final_stakes, capped)

