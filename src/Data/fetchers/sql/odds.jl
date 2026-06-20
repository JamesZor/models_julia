# src/data/fetchers/sql/odds.jl

# Make sure the Markets module is available in this scope
using .Markets 

"""
Maps raw API market ids to internal market groups.
"""
function map_odds_market_group(market_id::Integer)::String 
    if market_id in [1,3] return "1X2"
    elseif market_id == 2 return "Double chance" 
    elseif market_id == 4 return "Draw no bet" 
    elseif market_id == 5 return "Both teams to score" 
    elseif market_id == 6 return "First team to score" 
    elseif market_id == 9 return "Match goals" 
    elseif market_id == 17 return "Asian handicap" 
    elseif market_id == 21 return "Corners 2-way" 
    else return "Other" 
    end 
end

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::OddsData)
    # Same SQL as before, ensuring we pull `m.start_timestamp AS match_date`
    query = """
        SELECT 
            m.tournament_id, m.season_id, m.start_timestamp AS match_date,
            o.match_id, o.market_id, o.market_name, o.choice_name, o.choice_group,
            o.initial_fraction_num, o.initial_fraction_den,
            o.fraction_num, o.fraction_den, o.winning
        FROM sofascore.match_odds o
        JOIN sofascore.matches m ON o.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
    """
    try
        return DataFrame(LibPQ.execute(conn, query, [t_ids]))
    catch e
        @warn "Failed to fetch OddsData: $(e)"
        return DataFrame()
    end
end

const ODDS_SCHEMA = Dict{Symbol, Type}(
    :match_id => Int32,
    :market_name => String,
    :selection => Symbol,
    :odds_open => Float64,
    :odds_close => Float64
)

function process_data(df::DataFrame, ::OddsData; config::MarketConfig = Markets.DEFAULT_MARKET_CONFIG)

    df.market_group = map_odds_market_group.(df.market_id)
    # 1. Reconstruct fractions so the Markets module parsers don't break
    df.initial_fractional_value = [
        ismissing(n) ? missing : "$n/$d" 
        for (n, d) in zip(df.initial_fraction_num, df.initial_fraction_den)
    ]
    df.final_fractional_value = [
        ismissing(n) ? missing : "$n/$d" 
        for (n, d) in zip(df.fraction_num, df.fraction_den)
    ]

    # 2. Replicate the old prepare_market_data logic
    dfs = DataFrame[]
    for market in config.markets
        # Dispatch to your specific implementations (1x2, BTTS, etc.)
        raw_df = Markets._process_market_type(df, market)
        
        if !isempty(raw_df)
            push!(dfs, raw_df)
        end
    end
    
    if isempty(dfs)
        @warn "No market data found for the provided configuration."
        return DataFrame()
    end
    
    long_df = vcat(dfs...)

    # 3. Attach the Date
    # Because _build_long_rows drops the date, we leftjoin it back on 
    # from our raw SQL DataFrame. No need to wait for MatchesData anymore!
    date_df = unique(select(df, [:match_id, :match_date]))
    long_df = leftjoin(long_df, date_df, on=:match_id)
    rename!(long_df, :match_date => :date)

    # 4. Math Enrichment (Probabilities, Vig, Fair Odds, CLM)
    Markets._enrich_market_data!(long_df)

    # Apply strict schema
    apply_schema!(long_df, ODDS_SCHEMA)

    return long_df
end

function validate_data(df::DataFrame, ::OddsData)
    # Now we check against your enriched columns
    if !("fair_odds_close" in names(df)) || !("clm_prob" in names(df))
        @error "OddsData QA Failed: Missing enriched market columns."
        return false
    end
    return true
end
