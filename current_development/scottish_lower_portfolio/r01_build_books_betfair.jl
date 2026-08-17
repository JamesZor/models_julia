# current_development/scottish_lower_portfolio/r01_build_books_betfair.jl
#
# Builds MatchBooks for all 5 Scottish models under BETFAIR EXCHANGE odds with:
# - DeArb() settlement
# - KellyLogUtility() joint solver
# - BakerMcHale(n_draws = 800) shrinkage under parameter uncertainty
# - ExecutionConfig(commission = PerBetCommission(0.02), max_selection_stake = 0.50, budget = 0.99)
#
# Caches built books to `cache/books_<model_name>_betfair_bm800.jls`.

include("_setup_scottish_betfair.jl")

# Define the Betfair Exchange BookSpec with 2% commission and 800-draw Baker-McHale
spec = PF.BookSpec(
    markets   = MARKETS,
    price     = PF.DeArb(),                       # Exchange De-Arb settlement
    allocator = PF.KellyLogUtility(),             # Multi-market joint Kelly allocator
    shrink    = PF.BakerMcHale(n_draws = 800),    # 800 posterior draws (parameter uncertainty)
    exec      = PF.ExecutionConfig(
                    commission = PF.PerBetCommission(0.02), # 2% Betfair Exchange Commission
                    max_selection_stake = 0.50,
                    budget = 0.99,
                    require_complete_markets = true
                )
)

println("\n", "="^80)
println("BUILDING BETFAIR EXCHANGE MATCHBOOKS (800 Baker-McHale Draws per Match)")
println("="^80)
@info "BookSpec configured" price="DeArb" commission="2% PerBet" shrink="BakerMcHale(800 draws)" n_markets=length(MARKETS.markets)

books_map = Dict{String, Vector{PF.MatchBook}}()

for (m_name, exp) in all_exprs
    m_latents = latents_map[m_name]
    cache_file = joinpath(CACHE_DIR, "books_$(m_name)_betfair_bm800.jls")
    
    if isfile(cache_file) && get(ENV, "REBUILD_BOOKS", "0") != "1"
        @info "Reusing cached Betfair MatchBooks for: $m_name" cache_file
        books_map[m_name] = deserialize(cache_file)
    else
        @info "Building Betfair MatchBooks for: $m_name ($nrow(m_latents) matches)..."
        t0 = time()
        b = PF.build_books(spec, m_latents, exp, odds, ds)
        elapsed = round(time() - t0, digits = 1)
        @info "Completed Betfair MatchBooks for $m_name in $(elapsed)s" n_books=length(b)
        serialize(cache_file, b)
        books_map[m_name] = b
    end
end

println("\n", "="^80)
println("BETFAIR MATCHBOOK DIAGNOSTICS & SUMMARY")
println("="^80)

summary_df = DataFrame(
    model = String[],
    n_books = Int[],
    median_sels = Float64[],
    median_k_shrink = Float64[],
    mean_full_kelly = Float64[],
    max_kkt_error = Float64[]
)

for (m_name, b_list) in books_map
    isempty(b_list) && continue
    push!(summary_df, (
        m_name,
        length(b_list),
        median(length(b.sels) for b in b_list),
        round(median(b.k_shrink for b in b_list), digits = 3),
        round(100 * mean(sum(b.a_kelly) for b in b_list), digits = 1),
        maximum(b.kkt for b in b_list)
    ))
end

show(summary_df; allrows = true, allcols = true, truncate = 0)
println("\n\nAll Betfair MatchBooks built and cached successfully.")
