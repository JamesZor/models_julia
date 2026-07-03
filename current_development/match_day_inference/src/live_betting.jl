# current_development/match_day_inference/src/live_betting.jl

using Redis
using JSON3
using DataFrames
using Statistics
using Dates
using Printf
using PrettyTables
using BayesianFootball
import BayesianFootball.Signals: KellyCriterion, BayesianKelly, compute_stake

# ═══════════════════════════════════════════════════════════════════════════════
#  MARKET TYPE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ppd_to_betfair_type(market_name::String, market_line::Float64) -> Union{String, Nothing}

Maps a PPD market (name + line) to the corresponding Betfair market type string.
Examples: ("1X2", 0.0) -> "MATCH_ODDS", ("OverUnder", 2.5) -> "OVER_UNDER_25"
"""
function ppd_to_betfair_type(market_name::String, market_line::Float64)
    if market_name == "1X2"
        return "MATCH_ODDS"
    elseif market_name == "OverUnder"
        line_str = replace(@sprintf("%.1f", market_line), "." => "")
        return "OVER_UNDER_$line_str"
    elseif market_name == "BTTS"
        return "BOTH_TEAMS_TO_SCORE"
    else
        return nothing  # DoubleChance etc. not on Betfair
    end
end

"""
    betfair_to_ppd_type(bf_type::String) -> Union{Tuple{String, Float64}, Nothing}

Reverse maps a Betfair market type string to PPD (market_name, market_line).
Examples: "MATCH_ODDS" -> ("1X2", 0.0), "OVER_UNDER_25" -> ("OverUnder", 2.5)
"""
function betfair_to_ppd_type(bf_type::String)
    if bf_type == "MATCH_ODDS"
        return ("1X2", 0.0)
    elseif startswith(bf_type, "OVER_UNDER_")
        line_str = bf_type[12:end]  # e.g. "25" from "OVER_UNDER_25"
        line = parse(Float64, line_str[1:end-1] * "." * line_str[end:end])
        return ("OverUnder", line)
    elseif bf_type == "BOTH_TEAMS_TO_SCORE"
        return ("BTTS", 0.0)
    else
        return nothing
    end
end

"""
    selection_display_name(selection::Symbol, market_name::String) -> String

Returns a human-readable display name for a PPD selection symbol.
"""
function selection_display_name(selection::Symbol, market_name::String)
    s = string(selection)
    if market_name == "1X2"
        return s == "home" ? "Home" : s == "away" ? "Away" : s == "draw" ? "Draw" : titlecase(s)
    elseif market_name == "OverUnder"
        # :over_25 -> "Over 2.5", :under_05 -> "Under 0.5"
        if startswith(s, "over_")
            digits = s[6:end]
            return "Over $(digits[1:end-1]).$(digits[end])"
        elseif startswith(s, "under_")
            digits = s[7:end]
            return "Under $(digits[1:end-1]).$(digits[end])"
        end
        return s
    elseif market_name == "BTTS"
        return s == "btts_yes" ? "Yes" : s == "btts_no" ? "No" : s
    else
        return s
    end
end

"""
    market_display_name(market_name::String, market_line::Float64) -> String

Returns a compact display label for the market group header.
"""
function market_display_name(market_name::String, market_line::Float64)
    if market_name == "1X2"
        return "1X2"
    elseif market_name == "OverUnder"
        return "O/U $(@sprintf("%.1f", market_line))"
    elseif market_name == "BTTS"
        return "BTTS"
    else
        return market_name
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  RUNNER PARSING (Generalized for all market types)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _resolve_runner_role(runner_name, market_name, market_line, home_norm, away_norm)

Internal: determines which PPD selection symbol a Betfair runner corresponds to.
"""
function _resolve_runner_role(runner_name::String, market_name::String, market_line::Float64,
                               home_norm::String, away_norm::String)
    normalize_name(name) = lowercase(replace(string(name), "-" => "", "_" => "", " " => ""))
    name_norm = normalize_name(runner_name)
    
    if market_name == "1X2"
        if occursin(home_norm, name_norm) || occursin(name_norm, home_norm) || name_norm == "home"
            return :home
        elseif occursin(away_norm, name_norm) || occursin(name_norm, away_norm) || name_norm == "away"
            return :away
        elseif occursin("draw", name_norm) || name_norm == "thedraw"
            return :draw
        end
    elseif market_name == "OverUnder"
        line_str = replace(@sprintf("%.1f", market_line), "." => "")
        if occursin("over", name_norm)
            return Symbol("over_$line_str")
        elseif occursin("under", name_norm)
            return Symbol("under_$line_str")
        end
    elseif market_name == "BTTS"
        if occursin("yes", name_norm)
            return :btts_yes
        elseif occursin("no", name_norm)
            return :btts_no
        end
    end
    
    return nothing
end

"""
    _expected_selections(market_name::String, market_line::Float64)

Returns the expected PPD selection symbols for a given market type.
"""
function _expected_selections(market_name::String, market_line::Float64)
    if market_name == "1X2"
        return [:home, :draw, :away]
    elseif market_name == "OverUnder"
        line_str = replace(@sprintf("%.1f", market_line), "." => "")
        return [Symbol("over_$line_str"), Symbol("under_$line_str")]
    elseif market_name == "BTTS"
        return [:btts_yes, :btts_no]
    else
        return Symbol[]
    end
end

"""
    parse_runners_for_market(runners, market_name, market_line, home_team, away_team; selections)

Generalized runner parser. Maps Betfair runner payloads to PPD selection symbols
for any market type (1X2, OverUnder, BTTS).

Runner identity is resolved via:
  1. `runner_name` field injected by the Python streamer (Option 1)
  2. `selections` kwarg from `live_market_meta` (Option 2)
  3. Fallback to raw runner key
"""
function parse_runners_for_market(runners, market_name::String, market_line::Float64,
                                   home_team::String, away_team::String;
                                   selections::Dict{String,String} = Dict{String,String}())
    odds_dict = Dict{Symbol, NamedTuple}()
    
    expected = _expected_selections(market_name, market_line)
    if isempty(expected)
        return odds_dict
    end
    
    # Initialize all expected selections with NaN
    for sel in expected
        odds_dict[sel] = (back = NaN, lay = NaN, back_size = 0.0, lay_size = 0.0)
    end
    
    if isempty(runners)
        return odds_dict
    end
    
    # Normalization helper
    normalize_name(name) = lowercase(replace(string(name), "-" => "", "_" => "", " " => ""))
    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)
    
    for (k, runner) in runners
        runner_key = string(k)
        
        # Resolve runner name: Option 1 (injected) -> Option 2 (metadata) -> fallback
        runner_name = if haskey(runner, :runner_name) && runner.runner_name !== nothing
            string(runner.runner_name)
        elseif haskey(selections, runner_key)
            selections[runner_key]
        else
            runner_key
        end
        
        # Map runner to selection based on market type
        role = _resolve_runner_role(runner_name, market_name, market_line, home_norm, away_norm)
        
        if role !== nothing && role in expected
            # Extract back & lay prices
            back_p = NaN; back_s = 0.0
            if haskey(runner, :best_back) && !isempty(runner.best_back)
                back_p = Float64(runner.best_back[1].price)
                back_s = haskey(runner.best_back[1], :size) ? Float64(runner.best_back[1].size) : 0.0
            end
            
            lay_p = NaN; lay_s = 0.0
            if haskey(runner, :best_lay) && !isempty(runner.best_lay)
                lay_p = Float64(runner.best_lay[1].price)
                lay_s = haskey(runner.best_lay[1], :size) ? Float64(runner.best_lay[1].size) : 0.0
            end
            
            odds_dict[role] = (back = back_p, lay = lay_p, back_size = back_s, lay_size = lay_s)
        end
    end
    
    return odds_dict
end

# Backward-compatible wrapper for 1X2 markets
function parse_redis_runners(runners, home_team::String, away_team::String;
                              selections::Dict{String,String} = Dict{String,String}())
    return parse_runners_for_market(runners, "1X2", 0.0, home_team, away_team; selections=selections)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  REDIS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    get_live_market_mappings(redis_conn)

Scans the `live_market_meta` hash in Redis and returns a NamedTuple:
  - `markets`: Dict{Tuple{String, String, String}, String} mapping (home_slug, away_slug, bf_type) -> market_id
  - `selections`: Dict{String, Dict{String, String}} mapping market_id -> {selection_id -> runner_name}
"""
function get_live_market_mappings(redis_conn)
    mapping_lookup = Dict{Tuple{String, String, String}, String}()
    selections_lookup = Dict{String, Dict{String, String}}()
    try
        meta_dict = Redis.hgetall(redis_conn, "live_market_meta")

        for (market_id, raw_json) in meta_dict
            try
                meta = JSON3.read(raw_json)
                home_slug = haskey(meta, :home_slug) && meta.home_slug !== nothing ? string(meta.home_slug) : ""
                away_slug = haskey(meta, :away_slug) && meta.away_slug !== nothing ? string(meta.away_slug) : ""
                market_type = haskey(meta, :market_type) ? string(meta.market_type) : ""

                if !isempty(home_slug) && !isempty(away_slug) && !isempty(market_type)
                    mapping_lookup[(home_slug, away_slug, market_type)] = string(market_id)
                end

                # Option 2: Extract selections mapping {selection_id -> runner_name}
                if haskey(meta, :selections) && meta.selections !== nothing
                    sel_dict = Dict{String, String}()
                    for (sid, sname) in pairs(meta.selections)
                        sel_dict[string(sid)] = string(sname)
                    end
                    selections_lookup[string(market_id)] = sel_dict
                end
            catch e
                @warn "Failed to parse metadata JSON for market $market_id: $e"
            end
        end
    catch e
        @error "Failed to retrieve live_market_meta from Redis: $e"
    end
    return (markets=mapping_lookup, selections=selections_lookup)
end

"""
    get_available_markets_for_match(market_id_lookup, home_slug, away_slug)

Finds all Betfair markets available for a specific match.
Returns a sorted vector of NamedTuples: (ppd_market, ppd_line, bf_type, market_id)
"""
function get_available_markets_for_match(market_id_lookup, home_slug::String, away_slug::String)
    available = NamedTuple{(:ppd_market, :ppd_line, :bf_type, :market_id), Tuple{String, Float64, String, String}}[]
    for ((h, a, bf_type), mid) in market_id_lookup.markets
        if (h == home_slug && a == away_slug) || (h == away_slug && a == home_slug)
            ppd_info = betfair_to_ppd_type(bf_type)
            if ppd_info !== nothing
                push!(available, (ppd_market=ppd_info[1], ppd_line=ppd_info[2], bf_type=bf_type, market_id=string(mid)))
            end
        end
    end
    # Sort: 1X2 first, then O/U by line, then BTTS
    sort!(available, by=x -> (x.ppd_market == "1X2" ? 0 : x.ppd_market == "OverUnder" ? 1 : 2, x.ppd_line))
    return available
end

"""
    fetch_live_odds_for_market(redis_conn, market_id_lookup, home, away, mkt)

Fetches and parses live runner odds from Redis for a specific market.
`mkt` is a NamedTuple from `get_available_markets_for_match`.
"""
function fetch_live_odds_for_market(redis_conn, market_id_lookup, home::String, away::String, mkt)
    sel_dict = get(market_id_lookup.selections, mkt.market_id, Dict{String, String}())
    raw_data = Redis.hget(redis_conn, "live_markets", mkt.market_id)
    
    if raw_data === nothing
        return parse_runners_for_market(Dict(), mkt.ppd_market, mkt.ppd_line, home, away)
    end
    
    try
        data = JSON3.read(raw_data)
        if haskey(data, :runners)
            return parse_runners_for_market(data.runners, mkt.ppd_market, mkt.ppd_line, home, away; selections=sel_dict)
        end
    catch e
        @warn "Failed to parse Redis JSON for market $(mkt.market_id): $e"
    end
    
    return parse_runners_for_market(Dict(), mkt.ppd_market, mkt.ppd_line, home, away)
end

# Legacy convenience methods
function poll_redis_live_odds(redis_conn, market_id_lookup, home_slug::String, away_slug::String, market_type::String = "1X2")
    bf_market_type = market_type == "1X2" ? "MATCH_ODDS" : market_type
    market_id = get(market_id_lookup.markets, (home_slug, away_slug, bf_market_type), nothing)
    if market_id === nothing
        market_id = get(market_id_lookup.markets, (away_slug, home_slug, bf_market_type), nothing)
    end
    if market_id === nothing
        return parse_redis_runners(Dict(), home_slug, away_slug)
    end
    sel_dict = get(market_id_lookup.selections, market_id, Dict{String, String}())
    raw_data = Redis.hget(redis_conn, "live_markets", market_id)
    if raw_data === nothing
        return parse_redis_runners(Dict(), home_slug, away_slug)
    end
    try
        data = JSON3.read(raw_data)
        if haskey(data, :runners)
            return parse_redis_runners(data.runners, home_slug, away_slug; selections=sel_dict)
        end
    catch e
        @warn "Failed to parse Redis JSON for market $market_id: $e"
    end
    return parse_redis_runners(Dict(), home_slug, away_slug)
end

function poll_redis_live_odds(redis_conn, match_id::Int, home_team::String, away_team::String)
    market_id_lookup = get_live_market_mappings(redis_conn)
    return poll_redis_live_odds(redis_conn, market_id_lookup, home_team, away_team)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  BETTING SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    calculate_betting_signals(ppd, redis_conn, todays_matches, market_id_lookup; kelly_fraction=0.5, min_edge=0.0)

Generates EV and Kelly stakes for today's matches across ALL available Betfair markets.
Returns a DataFrame with one row per (match, market, selection).
"""
function calculate_betting_signals(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame,
                                    market_id_lookup; kelly_fraction=0.5, min_edge=0.0)
    kelly_std = KellyCriterion(kelly_fraction)
    kelly_bayes = BayesianKelly(min_edge)
    
    df_results = DataFrame(
        match_id = Int[], home_team = String[], away_team = String[],
        market = String[], market_line = Float64[], selection = Symbol[],
        prob_model = Float64[], odds_back = Float64[], odds_lay = Float64[],
        ev = Float64[], stake_std_kelly = Float64[], stake_bayes_kelly = Float64[]
    )
    
    for row in eachrow(todays_matches)
        mid = Int(row.match_id)
        home = String(row.home_team)
        away = String(row.away_team)
        
        available = get_available_markets_for_match(market_id_lookup, home, away)
        
        for mkt in available
            # Get PPD rows for this specific market
            ppd_rows = subset(ppd.df,
                :match_id => ByRow(==(mid)),
                :market_name => ByRow(==(mkt.ppd_market)),
                :market_line => ByRow(==(mkt.ppd_line))
            )
            isempty(ppd_rows) && continue
            
            live_odds = fetch_live_odds_for_market(redis_conn, market_id_lookup, home, away, mkt)
            
            for sel_row in eachrow(ppd_rows)
                sel = sel_row.selection
                dist = sel_row.distribution
                p_model = mean(dist)
                
                odds_info = get(live_odds, sel, (back=NaN, lay=NaN, back_size=0.0, lay_size=0.0))
                back_odds = odds_info.back
                lay_odds = odds_info.lay
                
                ev = NaN; stake_std = 0.0; stake_bayes = 0.0
                if !isnan(back_odds) && back_odds > 1.0
                    ev = (p_model * back_odds) - 1.0
                    stake_std = compute_stake(kelly_std, dist, back_odds)
                    stake_bayes = compute_stake(kelly_bayes, dist, back_odds)
                end
                
                push!(df_results, (
                    match_id=mid, home_team=home, away_team=away,
                    market=mkt.ppd_market, market_line=mkt.ppd_line, selection=sel,
                    prob_model=p_model, odds_back=back_odds, odds_lay=lay_odds,
                    ev=ev, stake_std_kelly=stake_std, stake_bayes_kelly=stake_bayes
                ))
            end
        end
    end
    
    return df_results
end

function calculate_betting_signals(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame;
                                    kelly_fraction=0.5, min_edge=0.0)
    market_id_lookup = get_live_market_mappings(redis_conn)
    return calculate_betting_signals(ppd, redis_conn, todays_matches, market_id_lookup;
                                     kelly_fraction=kelly_fraction, min_edge=min_edge)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  PRETTY-PRINTED DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _build_match_table(ppd, redis_conn, market_id_lookup, home, away, mid, kelly_std, kelly_bayes)

Builds the PrettyTables matrix and metadata for a single match across all available markets.
Returns (data::Matrix{Any}, hlines::Vector{Int}, has_value_bets::Bool) or nothing if no data.
"""
function _build_match_table(ppd::Predictions.PPD, redis_conn, market_id_lookup,
                             home::String, away::String, mid::Int,
                             kelly_std, kelly_bayes)
    available = get_available_markets_for_match(market_id_lookup, home, away)
    
    # Collect all rows across markets
    table_rows = NamedTuple[]
    hlines = Int[]  # row indices where horizontal lines should appear between market groups
    
    for mkt in available
        ppd_rows = subset(ppd.df,
            :match_id => ByRow(==(mid)),
            :market_name => ByRow(==(mkt.ppd_market)),
            :market_line => ByRow(==(mkt.ppd_line))
        )
        isempty(ppd_rows) && continue
        
        live_odds = fetch_live_odds_for_market(redis_conn, market_id_lookup, home, away, mkt)
        
        market_label = market_display_name(mkt.ppd_market, mkt.ppd_line)
        first_in_group = true
        
        for sel_row in eachrow(ppd_rows)
            sel = sel_row.selection
            dist = sel_row.distribution
            p_model = mean(dist)
            
            odds_info = get(live_odds, sel, (back=NaN, lay=NaN, back_size=0.0, lay_size=0.0))
            back_odds = odds_info.back
            lay_odds = odds_info.lay
            
            ev = NaN; stake_std = 0.0; stake_bayes = 0.0
            if !isnan(back_odds) && back_odds > 1.0
                ev = (p_model * back_odds) - 1.0
                stake_std = compute_stake(kelly_std, dist, back_odds)
                stake_bayes = compute_stake(kelly_bayes, dist, back_odds)
            end
            
            sel_label = selection_display_name(sel, mkt.ppd_market)
            is_value = !isnan(ev) && ev > 0.0
            
            model_price = p_model > 0.0 ? 1.0 / p_model : NaN
            mid_odds = (!isnan(back_odds) && !isnan(lay_odds)) ? (back_odds + lay_odds) / 2.0 : NaN
            mid_pct = (!isnan(mid_odds) && mid_odds > 0.0) ? 1.0 / mid_odds : NaN
            
            push!(table_rows, (
                market = first_in_group ? market_label : "",
                selection = is_value ? "* $sel_label" : "  $sel_label",
                prob = p_model,
                mid_pct = mid_pct,
                model_price = model_price,
                mid = mid_odds,
                back = back_odds,
                lay = lay_odds,
                ev = ev,
                kelly_std = stake_std,
                kelly_bayes = stake_bayes,
            ))
            first_in_group = false
        end
        
        # Mark boundary between market groups
        push!(hlines, length(table_rows))
    end
    
    if isempty(table_rows)
        return nothing
    end
    
    # Remove the last hline (bottom of table, PrettyTables draws this automatically)
    if !isempty(hlines) && hlines[end] == length(table_rows)
        pop!(hlines)
    end
    
    # Build the matrix
    n = length(table_rows)
    data = Matrix{Any}(undef, n, 11)
    has_value = false
    
    for (i, r) in enumerate(table_rows)
        data[i, 1] = r.market
        data[i, 2] = r.selection
        data[i, 3] = @sprintf("%.1f%%", r.prob * 100)
        data[i, 4] = isnan(r.mid_pct) ? "----" : @sprintf("%.1f%%", r.mid_pct * 100)
        data[i, 5] = isnan(r.model_price) ? "----" : @sprintf("%.2f", r.model_price)
        data[i, 6] = isnan(r.mid) ? "----" : @sprintf("%.2f", r.mid)
        data[i, 7] = isnan(r.back)  ? "----" : @sprintf("%.2f", r.back)
        data[i, 8] = isnan(r.lay)   ? "----" : @sprintf("%.2f", r.lay)
        data[i, 9] = isnan(r.ev)    ? "----" : (r.ev > 0 ? @sprintf("+%.1f%%", r.ev * 100) : @sprintf("%.1f%%", r.ev * 100))
        data[i, 10] = r.kelly_std > 0   ? @sprintf("%.2f%%", r.kelly_std * 100)   : "----"
        data[i, 11] = r.kelly_bayes > 0 ? @sprintf("%.2f%%", r.kelly_bayes * 100) : "----"
        
        if !isnan(r.ev) && r.ev > 0
            has_value = true
        end
    end
    
    return (data=data, hlines=hlines, has_value=has_value)
end

"""
    print_live_betting_dashboard(ppd, redis_conn, todays_matches, market_id_lookup; kelly_fraction=0.5, min_edge=0.0)

PrettyTables-formatted live betting dashboard showing model probabilities vs Betfair odds
across 1X2, Over/Under, and BTTS markets with Kelly staking recommendations.
"""
function print_live_betting_dashboard(ppd::Predictions.PPD, redis_conn,
                                      todays_matches::AbstractDataFrame, market_id_lookup;
                                      kelly_fraction=0.5, min_edge=0.0)
    kelly_std = KellyCriterion(kelly_fraction)
    kelly_bayes = BayesianKelly(min_edge)
    
    table_format = PrettyTables.TextTableFormat(borders = PrettyTables.text_table_borders__unicode_rounded)
    
    println("\n" * "="^125)
    println(" LIVE MATCHDAY BETTING DASHBOARD | Kelly: $kelly_fraction | Min Edge: $min_edge | $(Dates.format(now(), "HH:MM:SS"))")
    println("="^125)
    
    for row in eachrow(todays_matches)
        mid = Int(row.match_id)
        home = String(row.home_team)
        away = String(row.away_team)
        
        result = _build_match_table(ppd, redis_conn, market_id_lookup, home, away, mid, kelly_std, kelly_bayes)
        
        if result === nothing
            println("\n> $home vs $away (ID: $mid)")
            println("   [!] No model predictions or live markets found.")
            continue
        end
        
        value_tag = result.has_value ? " [\$]" : ""
        println("\n> $home vs $away (ID: $mid)$value_tag")
        
        pretty_table(
            result.data;
            column_labels = ["Market", "Selection", "Model %", "Mid %", "Model Price", "Mid", "Back", "Lay", "EV", "Kelly", "Bayes K"],
            table_format = table_format,
            alignment = [:l, :l, :r, :r, :r, :r, :r, :r, :r, :r, :r]
        )
    end
    
    println("\n" * "="^125)
    println(" * = Value Bet (EV > 0) | [\$] = Match has value bets | Kelly = Std Kelly ($kelly_fraction frac)")
    println("="^125 * "\n")
end

function print_live_betting_dashboard(ppd::Predictions.PPD, redis_conn,
                                      todays_matches::AbstractDataFrame;
                                      kelly_fraction=0.5, min_edge=0.0)
    market_id_lookup = get_live_market_mappings(redis_conn)
    return print_live_betting_dashboard(ppd, redis_conn, todays_matches, market_id_lookup;
                                        kelly_fraction=kelly_fraction, min_edge=min_edge)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  PRETTY-PRINTED DASHBOARD — TWO-MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _lookup_ppd_distribution(ppd, mid, market_name, market_line, sel)

Finds the posterior predictive distribution for a specific (match, market, line,
selection) tuple in a PPD, or `nothing` if that model has no row for it (e.g. the
two compared models expose different O/U lines).
"""
function _lookup_ppd_distribution(ppd::Predictions.PPD, mid::Int, market_name::String,
                                   market_line::Float64, sel::Symbol)
    rows = subset(ppd.df,
        :match_id => ByRow(==(mid)),
        :market_name => ByRow(==(market_name)),
        :market_line => ByRow(==(market_line)),
        :selection => ByRow(==(sel))
    )
    return isempty(rows) ? nothing : rows[1, :distribution]
end

"""
    _build_match_table_compare(ppd_1, ppd_2, redis_conn, market_id_lookup, home, away, mid, kelly_std)

Builds the PrettyTables matrix for a single match, showing both models' probabilities,
EV, and Kelly stakes side by side against the live Betfair book. Markets/selections are
driven by `ppd_1`; `ppd_2` values fall back to "----" where that model has no matching row.
"""
function _build_match_table_compare(ppd_1::Predictions.PPD, ppd_2::Predictions.PPD,
                                     redis_conn, market_id_lookup,
                                     home::String, away::String, mid::Int,
                                     kelly_std)
    available = get_available_markets_for_match(market_id_lookup, home, away)

    table_rows = NamedTuple[]
    hlines = Int[]

    for mkt in available
        ppd_rows_1 = subset(ppd_1.df,
            :match_id => ByRow(==(mid)),
            :market_name => ByRow(==(mkt.ppd_market)),
            :market_line => ByRow(==(mkt.ppd_line))
        )
        isempty(ppd_rows_1) && continue

        live_odds = fetch_live_odds_for_market(redis_conn, market_id_lookup, home, away, mkt)
        market_label = market_display_name(mkt.ppd_market, mkt.ppd_line)
        first_in_group = true

        for sel_row in eachrow(ppd_rows_1)
            sel = sel_row.selection
            dist_1 = sel_row.distribution
            p_1 = mean(dist_1)

            dist_2 = _lookup_ppd_distribution(ppd_2, mid, mkt.ppd_market, mkt.ppd_line, sel)
            p_2 = dist_2 === nothing ? NaN : mean(dist_2)

            odds_info = get(live_odds, sel, (back=NaN, lay=NaN, back_size=0.0, lay_size=0.0))
            back_odds = odds_info.back
            lay_odds = odds_info.lay

            ev_1 = NaN; stake_1 = 0.0
            ev_2 = NaN; stake_2 = 0.0
            if !isnan(back_odds) && back_odds > 1.0
                ev_1 = (p_1 * back_odds) - 1.0
                stake_1 = compute_stake(kelly_std, dist_1, back_odds)
                if dist_2 !== nothing
                    ev_2 = (p_2 * back_odds) - 1.0
                    stake_2 = compute_stake(kelly_std, dist_2, back_odds)
                end
            end

            sel_label = selection_display_name(sel, mkt.ppd_market)
            is_value = (!isnan(ev_1) && ev_1 > 0.0) || (!isnan(ev_2) && ev_2 > 0.0)

            mid_odds = (!isnan(back_odds) && !isnan(lay_odds)) ? (back_odds + lay_odds) / 2.0 : NaN
            mid_pct = (!isnan(mid_odds) && mid_odds > 0.0) ? 1.0 / mid_odds : NaN

            push!(table_rows, (
                market = first_in_group ? market_label : "",
                selection = is_value ? "* $sel_label" : "  $sel_label",
                p1 = p_1,
                p2 = p_2,
                mid_pct = mid_pct,
                back = back_odds,
                lay = lay_odds,
                ev1 = ev_1,
                ev2 = ev_2,
                kelly1 = stake_1,
                kelly2 = stake_2,
            ))
            first_in_group = false
        end

        push!(hlines, length(table_rows))
    end

    isempty(table_rows) && return nothing

    if !isempty(hlines) && hlines[end] == length(table_rows)
        pop!(hlines)
    end

    n = length(table_rows)
    data = Matrix{Any}(undef, n, 11)
    has_value = false

    for (i, r) in enumerate(table_rows)
        data[i, 1]  = r.market
        data[i, 2]  = r.selection
        data[i, 3]  = @sprintf("%.1f%%", r.p1 * 100)
        data[i, 4]  = isnan(r.p2) ? "----" : @sprintf("%.1f%%", r.p2 * 100)
        data[i, 5]  = isnan(r.mid_pct) ? "----" : @sprintf("%.1f%%", r.mid_pct * 100)
        data[i, 6]  = isnan(r.back) ? "----" : @sprintf("%.2f", r.back)
        data[i, 7]  = isnan(r.lay)  ? "----" : @sprintf("%.2f", r.lay)
        data[i, 8]  = isnan(r.ev1)  ? "----" : (r.ev1 > 0 ? @sprintf("+%.1f%%", r.ev1 * 100) : @sprintf("%.1f%%", r.ev1 * 100))
        data[i, 9]  = isnan(r.ev2)  ? "----" : (r.ev2 > 0 ? @sprintf("+%.1f%%", r.ev2 * 100) : @sprintf("%.1f%%", r.ev2 * 100))
        data[i, 10] = r.kelly1 > 0 ? @sprintf("%.2f%%", r.kelly1 * 100) : "----"
        data[i, 11] = r.kelly2 > 0 ? @sprintf("%.2f%%", r.kelly2 * 100) : "----"

        if (!isnan(r.ev1) && r.ev1 > 0) || (!isnan(r.ev2) && r.ev2 > 0)
            has_value = true
        end
    end

    return (data=data, hlines=hlines, has_value=has_value)
end

"""
    print_live_betting_dashboard_compare(ppd_1, label_1, ppd_2, label_2, redis_conn, todays_matches, market_id_lookup; kelly_fraction=0.5)

PrettyTables-formatted live betting dashboard comparing two models' probabilities, EV,
and Kelly stakes side by side against Betfair odds across 1X2, Over/Under, and BTTS markets.
"""
function print_live_betting_dashboard_compare(ppd_1::Predictions.PPD, label_1::String,
                                               ppd_2::Predictions.PPD, label_2::String,
                                               redis_conn, todays_matches::AbstractDataFrame,
                                               market_id_lookup; kelly_fraction=0.5)
    kelly_std = KellyCriterion(kelly_fraction)

    table_format = PrettyTables.TextTableFormat(borders = PrettyTables.text_table_borders__unicode_rounded)

    println("\n" * "="^135)
    println(" LIVE MATCHDAY MODEL COMPARISON | $label_1  vs  $label_2 | Kelly: $kelly_fraction | $(Dates.format(now(), "HH:MM:SS"))")
    println("="^135)

    for row in eachrow(todays_matches)
        mid = Int(row.match_id)
        home = String(row.home_team)
        away = String(row.away_team)

        result = _build_match_table_compare(ppd_1, ppd_2, redis_conn, market_id_lookup, home, away, mid, kelly_std)

        if result === nothing
            println("\n> $home vs $away (ID: $mid)")
            println("   [!] No model predictions or live markets found.")
            continue
        end

        value_tag = result.has_value ? " [\$]" : ""
        println("\n> $home vs $away (ID: $mid)$value_tag")

        pretty_table(
            result.data;
            column_labels = ["Market", "Selection", "$label_1 %", "$label_2 %", "Mid %", "Back", "Lay",
                              "$label_1 EV", "$label_2 EV", "$label_1 Kelly", "$label_2 Kelly"],
            table_format = table_format,
            alignment = [:l, :l, :r, :r, :r, :r, :r, :r, :r, :r, :r]
        )
    end

    println("\n" * "="^135)
    println(" * = Value Bet (EV > 0, either model) | [\$] = Match has value bets | Kelly = Std Kelly ($kelly_fraction frac)")
    println("="^135 * "\n")
end

function print_live_betting_dashboard_compare(ppd_1::Predictions.PPD, label_1::String,
                                               ppd_2::Predictions.PPD, label_2::String,
                                               redis_conn, todays_matches::AbstractDataFrame;
                                               kelly_fraction=0.5)
    market_id_lookup = get_live_market_mappings(redis_conn)
    return print_live_betting_dashboard_compare(ppd_1, label_1, ppd_2, label_2, redis_conn, todays_matches,
                                                market_id_lookup; kelly_fraction=kelly_fraction)
end
