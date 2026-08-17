# current_development/scottish_lower_portfolio/r04_matchday_sheet.jl
#
# MATCHDAY STAKING SHEET GENERATOR
#
# Given a Saturday Scottish match card and a current bankroll (e.g. £1,000):
# 1. Prices every market (1X2, BTTS, Over/Under) under the Champion model.
# 2. Solves the multi-market joint Kelly portfolio.
# 3. Applies the calibrated Risk Policy (Trust + Baker-McHale 800 + Drawdown Budget + Cap).
# 4. Emits the exact £ stake ticket per selection.

include("_setup_scottish.jl")

# Target Bankroll for the matchday
const BANKROLL = 1_000.0

# Calibrated Portfolio System for Scottish Lower
sys = PF.PortfolioSystem(
    PF.BookSpec(
        markets   = MARKETS,
        price     = PF.RawPrice(),                    # Exact Bet365 bookmaker settlement
        allocator = PF.KellyLogUtility(),             # Multi-market joint Kelly allocator
        shrink    = PF.BakerMcHale(n_draws = 800),    # 800-draw parameter uncertainty shrinkage
        exec      = PF.ExecutionConfig(
                        commission = PF.NoCommission(),
                        max_selection_stake = 0.50,
                        budget = 0.99,
                        require_complete_markets = true
                    )
    ),
    PF.PolicySpec(
        trust    = PF.FlatTrust(0.25),
        risk     = PF.SlateDrawdown(23.0),            # Calibrated ~20% drawdown budget
        cap      = PF.FixedCap(0.15),                 # 15% maximum simultaneous bankroll cap
        filter   = PF.KeepAll(),
        grouping = PF.DailySlate()
    )
)

println("\n", "="^100)
println("GENERATING MATCHDAY STAKE SHEET (Champion: funnel_pxg_apm)")
println("="^100)
@printf("Starting Bankroll: £%.2f | Max Slate Risk Cap: %.1f%% (£%.2f)\n",
        BANKROLL, sys.policy.cap.cap * 100, BANKROLL * sys.policy.cap.cap)

# Select a representative high-volume Saturday card from the dataset
_have = Set(latents_df.match_id)
_by_date = Dict{Date, Vector{Int}}()
for r in eachrow(ds.matches)
    r.match_id in _have || continue
    push!(get!(_by_date, Date(r.match_date), Int[]), r.match_id)
end

# Pick the date with the maximum simultaneous matches
best_date, card_matches = sort(collect(_by_date), by = x -> length(x[2]), rev = true)[1]
@info "Selected Matchday Card" date=best_date n_matches=length(card_matches)

# Generate matchday stake sheet
card_latents = filter(r -> r.match_id in card_matches, latents_df)
sheet = PF.stake_sheet(sys, card_latents, expr_champ, odds, ds; bankroll = BANKROLL)

if isempty(sheet)
    println("\nNo positive-edge bets found for this card matching the criteria.")
else
    # Enrich sheet with match team names
    m_info = Dict(r.match_id => "$(r.home_team) vs $(r.away_team)" for r in eachrow(ds.matches))
    sheet[!, :fixture] = [get(m_info, mid, "Match $mid") for mid in sheet.match_id]
    
    # Format ticket
    ticket = DataFrame(
        Fixture     = sheet.fixture,
        Market      = sheet.market,
        Selection   = string.(sheet.selection),
        BookOdds    = round.(sheet.p_market, digits = 3),
        ModelProb   = round.(sheet.p_model, digits = 3),
        EdgePct     = round.((sheet.p_model .- sheet.p_market) .* 100, digits = 1),
        Stake_GBP   = round.(sheet.stake, digits = 2),
        BankrollPct = round.((sheet.stake ./ BANKROLL) .* 100, digits = 2)
    )
    
    # Filter to placed bets (stake > 0.01)
    placed = filter(r -> r.Stake_GBP >= 0.50, ticket)
    sort!(placed, :Stake_GBP, rev = true)
    
    println("\n", "="^100)
    println("MATCHDAY BETTING TICKET — DATE: $best_date")
    println("="^100)
    show(placed; allrows = true, allcols = true, truncate = 0)
    println()
    
    total_stake = sum(placed.Stake_GBP)
    @printf("\nTotal Slate Exposure: £%.2f (%.2f%% of Bankroll across %d bets)\n",
            total_stake, (total_stake / BANKROLL) * 100, nrow(placed))
end
