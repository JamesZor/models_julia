# src/dataV2/display.jl


# ==============================================================================
# PRETTY PRINTING (Data Containers)
# ==============================================================================

function Base.show(io::IO, ::MIME"text/plain", ds::DataStore)
    # Title & Segment
    printstyled(io, "DataStore", color=:cyan, bold=true)
    printstyled(io, " [$(typeof(ds.segment))]", color=:yellow, bold=true)
    printstyled(io, "\n (Container for football data)\n", color=:light_black)
    println(io, "=========")

    # Define a helper to print each dataframe section consistently
    function print_df_summary(label, field_name, df)
        # Section Header with Accessor
        printstyled(io, "[$label] ", color=:magenta, bold=true)
        printstyled(io, "access via: ", color=:light_black)
        printstyled(io, "ds.$field_name\n", color=:cyan)
        
        if isempty(df)
            printstyled(io, "  (Empty)\n", color=:light_black)
        else
            dims = "$(nrow(df)) rows × $(ncol(df)) cols"
            printstyled(io, "  Dimensions: ", color=:light_black)
            printstyled(io, "$dims\n", color=:white)
            
            # Column Preview (First 5-6 columns to avoid clutter)
            cols = names(df)
            preview_len = min(6, length(cols))
            preview_cols = join(string.(Symbol.(cols[1:preview_len])), ", ")
            remaining = length(cols) - preview_len
            
            print(io, "  Columns: ")
            printstyled(io, preview_cols, color=:light_black)
            if remaining > 0
                printstyled(io, " + $remaining more", color=:light_black, italic=true)
            end
            println(io)
        end
        println(io)
    end

    # Print the main sections
    print_df_summary("Matches", :matches, ds.matches)
    print_df_summary("Statistics", :statistics, ds.statistics)
    print_df_summary("Odds (Sofascore)", :odds, ds.odds)
    print_df_summary("Odds (Betfair)", :betfair_odds, ds.betfair_odds)
    print_df_summary("Lineups", :lineups, ds.lineups)
    print_df_summary("Incidents", :incidents, ds.incidents)
    print_df_summary("BBC (match totals)", :bbc, ds.bbc)
    print_df_summary("BBC (shot events)", :bbc_events, ds.bbc_events)
end

# Compact inline show (for arrays/logging)
function Base.show(io::IO, ds::DataStore)
    seg = typeof(ds.segment)
    m = nrow(ds.matches)
    s = nrow(ds.statistics)
    o = nrow(ds.odds)
    bo = nrow(ds.betfair_odds)
    l = nrow(ds.lineups)
    i = nrow(ds.incidents)
    
    print(io, "DataStore[$seg](matches=$m, stats=$s, odds=$o, betfair=$bo, lineups=$l, incidents=$i)")
end



