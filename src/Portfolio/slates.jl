# src/Portfolio/slates.jl
#
# Grouping matches into simultaneous settlement windows. This is the boundary that decides what
# "the bankroll at risk right now" means: bets inside one slate are settled against a common
# bankroll and compounded once, together.

export DailySlate, SingleMatchSlate, build_slates

"""
    DailySlate()

One slate per calendar date. The right default for a league programme where the bulk of a round
kicks off within a couple of hours of itself -- on ScottishLower this is a median of 8 matches
per slate.
"""
struct DailySlate <: AbstractSlateGrouping end

"""
    SingleMatchSlate()

One slate per match: sequential Kelly, no simultaneity.

Only correct if matches genuinely resolve one after another. Applied to a real weekend programme
it lets a whole round be staked as though each bet were the only one live, which is exactly how
the prototype drove its simulated bankroll negative.
"""
struct SingleMatchSlate <: AbstractSlateGrouping end

function group(::DailySlate, books::Vector{MatchBook})
    @assert issorted(books, by = b -> b.date) "books must be chronological -- build_books sorts them"
    slates = Slate[]
    for b in books
        if !isempty(slates) && slates[end].window == b.date
            push!(slates[end].books, b)
        else
            push!(slates, Slate(b.date, [b]))
        end
    end
    return slates
end

function group(::SingleMatchSlate, books::Vector{MatchBook})
    @assert issorted(books, by = b -> b.date) "books must be chronological"
    return [Slate(b.date, [b]) for b in books]
end

"Convenience: group with a system's configured grouping."
build_slates(sys::PortfolioSystem, books::Vector{MatchBook}) = group(sys.policy.grouping, books)
build_slates(books::Vector{MatchBook}) = group(DailySlate(), books)
