"Print the shared, kickoff-filtered fold inventory."
function sl_fold_table(ds, folds::AbstractVector{SLFold})
    println("  fold  season step fitted dropped t+1 last fitted first OOS")
    for f in folds
        println("  $(f.idx) $(f.season) $(f.step) $(length(f.fitted_ids)) $(length(f.dropped_ids)) $(nrow(f.oos_df)) $(sl_last_kickoff(ds, f.fitted_ids)) $(sl_first_kickoff(f.oos_df))")
    end
    nothing
end
