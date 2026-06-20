# current_development/MetaModels/src/utils.jl

using Printf
using DataFrames
import MCMCChains

export check_fold_rhats

"""
    check_fold_rhats(meta_results)

Iterates through all folds in a MetaExperimentResults object, extracts the MCMC chain,
and prints the maximum R-hat (Gelman-Rubin diagnostic) across all parameters.
Flags any folds where max R-hat exceeds 1.05.
"""
function check_fold_rhats(meta_results)
    println("\n" * "="^65)
    println("  MCMC CONVERGENCE CHECK (R-hat)")
    println("="^65)
    
    n_folds = length(meta_results.fold_results)
    println("Checking maximum R-hat across $n_folds folds...")
    
    for (i, fold_res) in enumerate(meta_results.fold_results)
        chain = fold_res.chain
        
        # MCMCChains.summarize returns a structure that can be converted to DataFrame
        summ = DataFrame(MCMCChains.summarize(chain))
        
        # Locate the rhat column robustly
        rhat_col_idx = findfirst(c -> lowercase(string(c)) == "rhat", names(summ))
        
        if rhat_col_idx !== nothing
            rhats = summ[!, rhat_col_idx]
            max_rhat = maximum(skipmissing(rhats))
            
            flag = max_rhat > 1.05 ? "⚠️ WARNING" : "✓ OK"
            @printf("  Fold %2d: Max R-hat = %.4f  %s\n", i, max_rhat, flag)
        else
            println("  Fold $i: Could not locate rhat column in chain summary.")
        end
    end
    println("="^65)
end
