# current_development/scottish_lower_portfolio/r01_build_books.jl
#
# Builds MatchBooks for all 5 Scottish models with:
# - RawPrice() (Bet365 quoted odds)
# - KellyLogUtility() (Joint log-utility solver)
# - BakerMcHale(n_draws = 800) (800 posterior draws for parameter uncertainty shrinkage)
# - ExecutionConfig(commission = NoCommission(), max_selection_stake = 0.50, budget = 0.99)
#
# Caches built books to `cache/books_<model_name>.jls` so subsequent policy sweeps run in milliseconds.

include("_setup_scottish.jl")

# Define the production BookSpec with 800 posterior draws for Baker-McHale
spec = PF.BookSpec(
    markets   = MARKETS,
    price     = PF.RawPrice(),                    # Bet365 bookmaker settlement on quoted prices
    allocator = PF.KellyLogUtility(),             # Multi-market joint Kelly allocator
    shrink    = PF.BakerMcHale(n_draws = 800),    # 800 posterior draws (parameter uncertainty)
    exec      = PF.ExecutionConfig(
                    commission = PF.NoCommission(), # Bookmaker odds (no exchange commission)
                    max_selection_stake = 0.50,
                    budget = 0.99,
                    require_complete_markets = true
                )
)

println("\n", "="^80)
println("BUILDING MATCHBOOKS (800 Baker-McHale Draws per Match)")
println("="^80)
@info "BookSpec configured" price="RawPrice" shrink="BakerMcHale(800 draws)" n_markets=length(MARKETS.markets)

books_map = Dict{String, Vector{PF.MatchBook}}()

for (m_name, exp) in all_exprs
    m_latents = latents_map[m_name]
    cache_file = joinpath(CACHE_DIR, "books_$(m_name)_bm800.jls")
    
    if isfile(cache_file) && get(ENV, "REBUILD_BOOKS", "0") != "1"
        @info "Reusing cached MatchBooks for: $m_name" cache_file
        books_map[m_name] = deserialize(cache_file)
    else
      @info "Building MatchBooks for: $m_name ($(nrow(m_latents)) matches)..."
        t0 = time()
        b = PF.build_books(spec, m_latents, exp, odds, ds)
        elapsed = round(time() - t0, digits = 1)
        @info "Completed MatchBooks for $m_name in $(elapsed)s" n_books=length(b)
        serialize(cache_file, b)
        books_map[m_name] = b
    end
end

println("\n", "="^80)
println("MATCHBOOK DIAGNOSTICS & SUMMARY")
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
println("\n\nAll MatchBooks built and cached successfully. Ready for policy sweeps (r02) and benchmarking (r03).")
