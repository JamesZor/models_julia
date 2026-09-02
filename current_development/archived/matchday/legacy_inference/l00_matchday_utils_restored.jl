# current_development/match_day_inference/l00_matchday_utils.jl

#=
A loader file to keep all the functions for running a match day inference. 
Looking to be integrated into a module of the main package, but later.

=#

using Revise
using BayesianFootball

using LibPQ

using DataFrames
using ThreadPinning
pinthreads(:cores)



# ========================================
#  Stage 1 - Training the model
# ========================================

function get_target_seasons_string(segment::Data.DataTournemantSegment) 
    # none - place holder 
    println("Placeholder for the type: $(segment)")
    return 
end  

get_target_seasons_string(::Data.Ireland)       = ["2026"]

### --------
# for the runner


# ==========================================
# i. Imports
# ==========================================
using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)
using Dates
using Distributions
using Turing
using Statistics

# ==========================================
# ii. Short Hands
# ==========================================
const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Evaluation = BayesianFootball.Evaluation
const BackTe
            popfirst!(vols)
        end
        
        # Clear the terminal screen (ANSI escape codes)
        print("\e[2J\e[H")
        
        println("="^70)
        println(" ⚽ $event_name  |  $market_type")
        println("    Market ID: $market_id")
        println("    Runner/Selection ID: $(runner.selection_id)")
        println("    Total Matched (Selection): £$(round(total_vol, digits=2))")
        println("="^70)
        
        # Diagnostics
        println(" ⏱️ TICK DIAGNOSTICS")
        println("    Time since last Betfair tick : $(round(time_since_tick, digits=1))s")
        
        # Format deltas with + sign if positive
        fmt_d(val) = isnan(val) ? "0.0" : (val > 0 ? "+$(round(val, digits=2))" : "$(round(val, digits=2))")
        
        println("    Tick Delta (Best Back)       : $(fmt_d(delta_back))")
        println("    Tick Delta (Best Lay)        : $(fmt_d(delta_lay))")
        println("    Tick Delta (Volume)          : $(fmt_d(delta_vol))")
        println("="^70)
        println()
        
        # Print Text-based Order Book
        println(" 📈 CURRENT ORDER BOOK (Top 3)")
        println(format_order_book(runner))
        println("="^70)
        
        if length(times) > 1
            # Plot 1: Prices
            p1 = lineplot(times, backs, title="Best Back & Lay Odds Trend", name="Best Back", color=:cyan, xlabel="Time (s)", ylabel="Odds", width=65, height=10)
            lineplot!(p1, times, lays, name="Best Lay", color=:magenta)
            println(p1)
            
            println()
        else
            println("Gathering trend data points...")
        end
        
        sleep(0.5) # Poll Redis every 500ms
    end
end
=#


