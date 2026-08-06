# src/Portfolio/display.jl
#
# Pretty printing. House style (src/features/display.jl, src/Data/display.jl): a rich
# `MIME"text/plain"` tree for the REPL, a compact one-liner for arrays and logging.
#
# The method worth caring about is PortfolioSystem: it draws the cache boundary, because
# "BookSpec is the cache key, PolicySpec is free to sweep" is the whole design and it is not
# visible from the type names.

# ==============================================================================
# 0. Helpers
# ==============================================================================

_pf_fmt(v::Float64) = string(round(v, digits = 6))
_pf_fmt(v::AbstractRange) = "$(first(v)):$(step(v)):$(last(v))"
_pf_fmt(v::AbstractString) = "\"$v\""

# Summarise long vectors. BakerMcHale's shrinkage grid is 51 elements and dumping it makes the
# spec unreadable, which defeats the point of the display.
function _pf_fmt(v::AbstractVector)
    length(v) <= 4 && return string(v)
    return "[$(first(v)) … $(last(v))] ($(length(v)) pts)"
end

_pf_fmt(v) = string(v)

"Print one `├──`/`└──` row."
function _pf_leaf(io::IO, label, value; last::Bool = false, indent = "  ",
                  value_color = :cyan)
    printstyled(io, indent, last ? "└── " : "├── ", color = :light_black)
    printstyled(io, rpad(label, 15), color = :white)
    printstyled(io, string(value), color = value_color)
    println(io)
end

_pct(x) = @sprintf("%.2f%%", 100x)

# ==============================================================================
# 1. Components
# ==============================================================================
#
# `component_name` previously returned the bare type name, so FlatTrust(0.25) and FlatTrust(1.0)
# printed identically -- in a module whose entire purpose is sweeping policies, that is a
# display that hides the thing you are sweeping. Fields are now included.

function component_name(x)
    name = string(nameof(typeof(x)))
    props = propertynames(x)
    isempty(props) && return name
    return name * "(" * join(("$p=$(_pf_fmt(getproperty(x, p)))" for p in props), ", ") * ")"
end

# FilterChain holds a tuple of filters; render it as a conjunction, which is what it is.
component_name(c::FilterChain) =
    "FilterChain(" * join(component_name.(c.filters), " & ") * ")"

const PortfolioComponent = Union{AbstractPricePolicy, AbstractCommissionModel, AbstractAllocator,
                                 AbstractShrinkage, AbstractTrustModel, AbstractRiskModel,
                                 AbstractExposureCap, AbstractSelectionFilter,
                                 AbstractSlateGrouping}

Base.show(io::IO, x::PortfolioComponent) = print(io, component_name(x))

function Base.show(io::IO, ::MIME"text/plain", x::PortfolioComponent)
    printstyled(io, string(nameof(typeof(x))), color = :green, bold = true)
    props = propertynames(x)
    if isempty(props)
        println(io, "()")
        return
    end
    println(io)
    for (i, p) in enumerate(props)
        _pf_leaf(io, string(p), _pf_fmt(getproperty(x, p)); last = i == length(props))
    end
end

# ==============================================================================
# 2. Configuration -- the cache boundary is the point
# ==============================================================================

function Base.show(io::IO, ::MIME"text/plain", e::ExecutionConfig)
    printstyled(io, "ExecutionConfig\n", color = :green, bold = true)
    props = propertynames(e)
    for (i, p) in enumerate(props)
        _pf_leaf(io, string(p), _pf_fmt(getproperty(e, p)); last = i == length(props))
    end
end

function Base.show(io::IO, ::MIME"text/plain", s::BookSpec)
    printstyled(io, "BookSpec", color = :cyan, bold = true)
    printstyled(io, "  → determines a MatchBook. THIS IS THE CACHE KEY.\n", color = :yellow)
    _pf_leaf(io, "markets", "$(length(s.markets.markets)): " *
             join(unique(string.(Data.market_group.(s.markets.markets))), ", "))
    _pf_leaf(io, "price", component_name(s.price))
    _pf_leaf(io, "allocator", component_name(s.allocator))
    _pf_leaf(io, "shrink", component_name(s.shrink))
    _pf_leaf(io, "commission", component_name(s.exec.commission))
    _pf_leaf(io, "budget", _pf_fmt(s.exec.budget))
    _pf_leaf(io, "cache key", string(book_cache_key(s), base = 16), last = true,
             value_color = :yellow)
    printstyled(io, "  change any of these and 600+ books must be rebuilt (~26s).\n",
                color = :light_black, italic = true)
end

Base.show(io::IO, s::BookSpec) =
    print(io, "BookSpec(", nameof(typeof(s.price)), ", ", nameof(typeof(s.allocator)), ", ",
          nameof(typeof(s.shrink)), ", key=", string(book_cache_key(s), base = 16), ")")

function Base.show(io::IO, ::MIME"text/plain", p::PolicySpec)
    printstyled(io, "PolicySpec", color = :cyan, bold = true)
    printstyled(io, "  → pure multipliers on an existing book. FREE TO SWEEP.\n", color = :green)
    _pf_leaf(io, "trust", component_name(p.trust))
    _pf_leaf(io, "risk", component_name(p.risk))
    _pf_leaf(io, "cap", component_name(p.cap))
    _pf_leaf(io, "filter", component_name(p.filter))
    _pf_leaf(io, "grouping", component_name(p.grouping), last = true)
    printstyled(io, "  note: once the drawdown constraint binds, `trust` reshapes the book but\n",
                color = :light_black, italic = true)
    printstyled(io, "  cannot rescale it -- risk_factor is homogeneous of degree 0.\n",
                color = :light_black, italic = true)
end

Base.show(io::IO, p::PolicySpec) =
    print(io, "PolicySpec(", component_name(p.trust), ", ", component_name(p.risk), ", ",
          component_name(p.cap), ")")

function Base.show(io::IO, ::MIME"text/plain", s::PortfolioSystem)
    printstyled(io, "PortfolioSystem\n", color = :cyan, bold = true)
    println(io, "="^68)
    printstyled(io, " BOOK  ", color = :magenta, bold = true)
    printstyled(io, "expensive · cached · key $(string(book_cache_key(s.book), base = 16))\n",
                color = :light_black)
    _pf_leaf(io, "markets", length(s.book.markets.markets))
    _pf_leaf(io, "price", component_name(s.book.price))
    _pf_leaf(io, "allocator", component_name(s.book.allocator))
    _pf_leaf(io, "shrink", component_name(s.book.shrink))
    _pf_leaf(io, "commission", component_name(s.book.exec.commission), last = true)

    printstyled(io, "─"^68, "\n", color = :yellow)
    printstyled(io, " ↑ above = CACHE KEY (rebuild)   ↓ below = FREE TO SWEEP\n",
                color = :yellow, bold = true)
    printstyled(io, "─"^68, "\n", color = :yellow)

    printstyled(io, " POLICY  ", color = :magenta, bold = true)
    printstyled(io, "cheap · milliseconds\n", color = :light_black)
    _pf_leaf(io, "trust", component_name(s.policy.trust))
    _pf_leaf(io, "risk", component_name(s.policy.risk))
    _pf_leaf(io, "cap", component_name(s.policy.cap))
    _pf_leaf(io, "filter", component_name(s.policy.filter))
    _pf_leaf(io, "grouping", component_name(s.policy.grouping), last = true)
    println(io, "="^68)
end

Base.show(io::IO, s::PortfolioSystem) =
    print(io, "PortfolioSystem(", s.book, ", ", s.policy, ")")

# ==============================================================================
# 3. Domain objects
# ==============================================================================

function Base.show(io::IO, ::MIME"text/plain", s::Selection)
    printstyled(io, "Selection ", color = :cyan, bold = true)
    printstyled(io, s.family, "\n", color = :white, bold = true)
    _pf_leaf(io, "odds quoted", round(s.odds_quoted, digits = 3))
    shrunk = s.odds_used < s.odds_quoted - 1e-12
    _pf_leaf(io, "odds used", "$(round(s.odds_used, digits = 3))" *
             (shrunk ? "  (de-arbed, $(_pct(1 - s.odds_used / s.odds_quoted)) shrink)" : ""))
    _pf_leaf(io, "p_model", _pct(s.p_model))
    _pf_leaf(io, "p_market", _pct(s.p_market))
    edge = s.p_model - s.p_market
    _pf_leaf(io, "edge", _pct(edge); last = true, value_color = edge > 0 ? :green : :red)
    printstyled(io, "  p_market is a FORECAST BENCHMARK, never a price.\n",
                color = :light_black, italic = true)
end

Base.show(io::IO, s::Selection) = print(io,
    "Selection($(s.family) @ $(round(s.odds_used, digits=3)), ",
    "p=$(round(s.p_model, digits=3)) vs mkt $(round(s.p_market, digits=3)))")

function Base.show(io::IO, ::MIME"text/plain", b::MatchBook)
    printstyled(io, "MatchBook ", color = :cyan, bold = true)
    printstyled(io, "$(b.m_id)", color = :white, bold = true)
    printstyled(io, "  $(b.date)", color = :yellow)
    printstyled(io, is_settled(b) ? "  [settled]\n" : "  [UNPLAYED — stakeable, not simulatable]\n",
                color = is_settled(b) ? :green : :yellow)

    _pf_leaf(io, "selections", length(b.sels))
    _pf_leaf(io, "state grid", "$(length(b.p_grid)) scorelines")
    _pf_leaf(io, "payoff R", "$(size(b.R, 1)) × $(size(b.R, 2))")
    _pf_leaf(io, "k_shrink", round(b.k_shrink, digits = 4))
    printstyled(io, "  ├── ", color = :light_black)
    printstyled(io, rpad("kkt", 15), color = :white)
    printstyled(io, @sprintf("%.2e", b.kkt), color = b.kkt < 1e-4 ? :cyan : :red)
    printstyled(io, b.kkt < 1e-4 ? "" : "   ← above 1e-4, solve is suspect", color = :red)
    println(io)
    _pf_leaf(io, "converged", b.converged; value_color = b.converged ? :cyan : :red)

    live = findall(>(1e-9), b.a_kelly)
    _pf_leaf(io, "Σ a_kelly", _pct(sum(b.a_kelly)), last = isempty(live))
    if !isempty(live)
        printstyled(io, "  └── ", color = :light_black)
        printstyled(io, "top stakes (full Kelly, pre-policy)\n", color = :white)
        order = sort(live, by = j -> -b.a_kelly[j])
        for j in first(order, min(5, length(order)))
            printstyled(io, "        ", rpad(b.sels[j].family, 20), color = :light_black)
            printstyled(io, @sprintf("%6s", _pct(b.a_kelly[j])), color = :cyan)
            printstyled(io, "  @ $(round(b.sels[j].odds_used, digits = 2))\n",
                        color = :light_black)
        end
        length(order) > 5 && printstyled(io, "        … $(length(order) - 5) more\n",
                                         color = :light_black)
    end
end

Base.show(io::IO, b::MatchBook) = print(io,
    "MatchBook($(b.m_id), $(b.date), $(length(b.sels)) sels, ",
    "Σa=$(round(sum(b.a_kelly), digits=3)), k=$(round(b.k_shrink, digits=3))",
    is_settled(b) ? "" : ", UNPLAYED", ")")

function Base.show(io::IO, ::MIME"text/plain", s::Slate)
    printstyled(io, "Slate ", color = :cyan, bold = true)
    printstyled(io, "$(s.window)\n", color = :yellow)
    _pf_leaf(io, "matches", length(s.books))
    _pf_leaf(io, "selections", sum(length(b.sels) for b in s.books; init = 0))
    _pf_leaf(io, "settled", "$(count(is_settled, s.books)) of $(length(s.books))", last = true)
    printstyled(io, "  these settle together, so they share one bankroll and one cap.\n",
                color = :light_black, italic = true)
end

Base.show(io::IO, s::Slate) =
    print(io, "Slate($(s.window), $(length(s.books)) matches)")

Base.show(io::IO, c::SlateContext) =
    print(io, "SlateContext(#$(c.idx), $(c.date), bankroll=$(round(c.bankroll, digits=2)))")

function Base.show(io::IO, ::MIME"text/plain", a::SlateAllocation)
    printstyled(io, "SlateAllocation\n", color = :cyan, bold = true)
    _pf_leaf(io, "matches", length(a.stakes))
    _pf_leaf(io, "live legs", sum(count(>(0), s) for s in a.stakes; init = 0))
    _pf_leaf(io, "exposure", _pct(a.exposure))
    _pf_leaf(io, "k_risk", round(a.k_risk, digits = 4))
    printstyled(io, "  └── ", color = :light_black)
    printstyled(io, rpad("capped", 15), color = :white)
    printstyled(io, a.capped, color = a.capped ? :yellow : :cyan)
    printstyled(io, a.capped ? "   ← the hard cap bound, not the drawdown budget" : "",
                color = :light_black)
    println(io)
end

Base.show(io::IO, a::SlateAllocation) = print(io,
    "SlateAllocation($(length(a.stakes)) matches, exposure=$(round(a.exposure, digits=4)), ",
    "k_risk=$(round(a.k_risk, digits=3))", a.capped ? ", CAPPED" : "", ")")

function Base.show(io::IO, ::MIME"text/plain", t::Trajectory)
    final = isempty(t.bankroll) ? 1.0 : t.bankroll[end]
    printstyled(io, "Trajectory ", color = :cyan, bold = true)
    printstyled(io, "$(length(t.slate_pl)) slates\n", color = :yellow)
    println(io, "="^60)
    _pf_leaf(io, "final", "$(round(final, digits = 4))×";
             value_color = final >= 1 ? :green : :red)
    _pf_leaf(io, "total staked", round(t.total_stake, digits = 3))
    _pf_leaf(io, "total P/L", round(t.total_pl, digits = 3);
             value_color = t.total_pl >= 0 ? :green : :red)
    t.total_stake > 0 && _pf_leaf(io, "flat ROI", _pct(t.total_pl / t.total_stake);
                                  value_color = t.total_pl >= 0 ? :green : :red)
    _pf_leaf(io, "bets", nrow(t.bets))
    if !isempty(t.bankroll)
        peak = accumulate(max, t.bankroll)
        _pf_leaf(io, "max drawdown", _pct(minimum(t.bankroll ./ peak) - 1); value_color = :red)
        _pf_leaf(io, "min bankroll", round(minimum(t.bankroll), digits = 4);
                 value_color = minimum(t.bankroll) > 0 ? :cyan : :red)
    end
    if !isempty(t.exposure)
        _pf_leaf(io, "exposure", "mean $(_pct(mean(t.exposure))), max $(_pct(maximum(t.exposure)))")
    end
    _pf_leaf(io, "slates capped", "$(t.n_capped) of $(length(t.slate_pl))", last = true)
    if !isempty(t.dates)
        printstyled(io, "  $(first(t.dates)) → $(last(t.dates))\n", color = :light_black)
    end
    printstyled(io, "  judge on growth, not ROI: a flat trust multiplier cancels out of ROI.\n",
                color = :light_black, italic = true)
end

Base.show(io::IO, t::Trajectory) = print(io,
    "Trajectory($(length(t.slate_pl)) slates, final $(round(isempty(t.bankroll) ? 1.0 : t.bankroll[end], digits=3))x, ",
    "$(nrow(t.bets)) bets)")
