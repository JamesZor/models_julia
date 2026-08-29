# src/Portfolio/compat.jl
#
# The bridge between the names this module has always used and the names the unified-framework
# briefing uses.
#
# ------------------------------------------------------------------------------
# THE ONE DESIGN DECISION THAT SHAPES EVERY OTHER FILE IN THE GRADUATION
# ------------------------------------------------------------------------------
#
# The briefing asks for `MarketSelection(family, group, line, sel, odds_close, odds_settle,
# prob_model, prob_market)` and `MarketBook(match_id, date, selections, p_grid, payoff_matrix,
# settle_vector, raw_alloc, shrink_k, kkt, converged)`. This module already has both, field for
# field, in the same positional order, under different names -- and adds nothing the briefing's
# version has.
#
# So the briefing's names are `const` ALIASES of the existing types, never second structs with the
# same shape. THREE REASONS, in descending order of how much they cost to get wrong:
#
#   1. TWO TYPES ANSWERING TO ONE NAME IN ONE SESSION IS THE FAILURE MODE THIS WHOLE GRADUATION
#      EXISTS TO REMOVE. A `MarketBook` that were a distinct struct would make every `MethodError`
#      in this repository ambiguous to read, and a `Vector{MatchBook}` unserialised from an
#      existing `.jls` cache would no longer be stakeable.
#
#   2. BACKWARD COMPATIBILITY BECOMES IDENTITY RATHER THAN EMULATION. A book built by
#      `build_books(spec, ::AbstractPosteriorLatents, ...)` IS a `MatchBook`, so `group`,
#      `stake_slate`, `simulate`, `stake_sheet`, `attribution` and every cached book on disk keep
#      working -- not because a bridge translates them, but because there is nothing to translate.
#
#   3. THE PARITY CLAIM GETS SHARPER. `test/unified_portfolio_tests.jl` compares two BUILDERS over
#      ONE set of types. If the types differed, every comparison would be field-by-field
#      transcription and a field the new builder forgot to copy would read as a pass.
#
# The field names the briefing spells differently are provided as ACCESSOR FUNCTIONS in §2.
# Functions rather than a `getproperty` overload, because `Base.getproperty(::MatchBook, ...)`
# would be visible to every package in the session and because a `MatchBook` field read sits inside
# `stake_slate`'s inner loop where an added dispatch layer is not free.

export MarketSelection, MarketBook, MatchedMarketOdds, PortfolioPolicy, LogUtility,
       UnsettledBooks
export book_match_id, book_date, book_selections, book_grid, book_payoff, book_settle,
       book_alloc, book_shrink, book_kkt, book_converged
export sel_name, sel_odds_close, sel_odds_settle, sel_prob_model, sel_prob_market, sel_edge

# ===================================================================
# 1. Type aliases
# ===================================================================

"The briefing's name for [`Selection`](@ref). The same type, not a look-alike."
const MarketSelection = Selection

"The briefing's name for [`MatchBook`](@ref). The same type, not a look-alike."
const MarketBook = MatchBook

"The briefing's name for [`OddsIndex`](@ref) -- quotes matched to fixtures, indexed by match."
const MatchedMarketOdds = OddsIndex

"The briefing's name for [`PolicySpec`](@ref).

The briefing's own `PolicySpec(allocator, caps, commission, risk, shrinkage, trust)` is NOT
adopted: `allocator`, `commission` and `shrinkage` change the `MatchBook` and therefore belong to
`BookSpec`, while `trust`, `risk`, `cap`, `filter` and `grouping` are pure post-multipliers on an
already-built book. That split is exactly the line between \"invalidates the book cache\" and
\"is a pure post-multiplier\", and it is the reason a policy sweep does not rebuild books -- which
is the reason walk-forward evaluation is affordable at all."
const PortfolioPolicy = PolicySpec

"The briefing's name for [`KellyLogUtility`](@ref), Jacot & Mochkovitch's non-mutually-exclusive
Kelly solve."
const LogUtility = KellyLogUtility

"""
    UnsettledBooks(books) -> Vector{MatchBook}

The briefing's name for [`unsettled_books`](@ref): the books `simulate` will refuse. A one-line
check that turns "`simulate` threw an assertion" into "these four fixtures have no result yet".
"""
const UnsettledBooks = unsettled_books

# ===================================================================
# 2. The briefing's field names, as accessors
# ===================================================================
#
# Prefixed (`book_`, `sel_`) so none of them collides with an existing export: `payoff_matrix`
# already names the `(sels, max_h, max_a, commission)` constructor, and shadowing it with a
# one-field getter would be a trap.

"`MatchBook.m_id` -- the briefing's `match_id`."
book_match_id(b::MatchBook) = b.m_id
"`MatchBook.date`."
book_date(b::MatchBook) = b.date
"`MatchBook.sels` -- the briefing's `selections`."
book_selections(b::MatchBook) = b.sels
"`MatchBook.p_grid` -- posterior-mean score grid, normalised to 1."
book_grid(b::MatchBook) = b.p_grid
"`MatchBook.R` -- the briefing's `payoff_matrix`. Jacot return matrix, `N x n`."
book_payoff(b::MatchBook) = b.R
"`MatchBook.settle` -- the briefing's `settle_vector`. `nothing` for an unplayed fixture."
book_settle(b::MatchBook) = b.settle
"`MatchBook.a_kelly` -- the briefing's `raw_alloc`. Full-size allocation on the posterior mean."
book_alloc(b::MatchBook) = b.a_kelly
"`MatchBook.k_shrink` -- the briefing's `shrink_k`."
book_shrink(b::MatchBook) = b.k_shrink
"`MatchBook.kkt` -- worst first-order-condition violation of `a_kelly`."
book_kkt(b::MatchBook) = b.kkt
"`MatchBook.converged` -- did the ALLOCATOR converge. Not the MCMC verdict; see `BuildReport`."
book_converged(b::MatchBook) = b.converged

"`Selection.selection` -- the briefing's `sel`."
sel_name(s::Selection) = s.selection
"`Selection.odds_quoted` -- the briefing's `odds_close`. The price as traded."
sel_odds_close(s::Selection) = s.odds_quoted
"""
`Selection.odds_used` -- the briefing's `odds_settle`. What we are settled at, after the
`AbstractPricePolicy` has been applied.
"""
sel_odds_settle(s::Selection) = s.odds_used
"`Selection.p_model` -- the briefing's `prob_model`."
sel_prob_model(s::Selection) = s.p_model
"`Selection.p_market` -- the briefing's `prob_market`. Vig-removed; a BENCHMARK, never a price."
sel_prob_market(s::Selection) = s.p_market
"`p_model - p_market`, in probability points on one de-vigged scale."
sel_edge(s::Selection) = s.p_model - s.p_market
