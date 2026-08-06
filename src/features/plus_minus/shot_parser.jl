# src/features/plus_minus/shot_parser.jl
#
# BBC commentary -> structured shot records -> a zonal xG model.
#
# FAITHFUL PORT of current_development/plus_minus_ratings/l02_shot_parser.jl. Only the data source
# changed (`ds.bbc_events` instead of a bespoke live_text pull).
#
# BBC's live text is Opta-derived and follows a rigid template:
#
#   "Attempt saved. Kai Kennedy (Queen of the South) right footed shot from outside the box is
#    saved in the top right corner by Robbie Mutch (Cove Rangers)."
#
# so `<shooter> (<team>) <BODY PART> from <ZONE> <outcome> [... following a <CONTEXT>]`. That gives
# zone, body part and set-piece context — i.e. everything a pre-tracking-era xG model used — WITHOUT
# any coordinates. It is the only route to xG for tiers 56/57, which have no SofaScore shot data.
#
# The research measured this model at 98.4-99.8% parse coverage, a Brier ladder 11.1% better than
# the base rate (the base paper's COORDINATE model managed 14.7%, so ~76% of the coordinate gain is
# retained with no coordinates), and team-level correlation 0.817 against SofaScore xG.
#
# VOCABULARY IS EMPIRICAL, NOT GUESSED — scanned over all 45,201 shot-bearing rows across tiers
# 54-57. Counts are recorded next to each entry so drift is detectable on a re-scrape. Matching is
# on KEYWORDS, not capture position: an earlier positional regex silently conflated "header" with an
# empty body-part capture and swallowed "from a direct free kick" into the zone.

using DataFrames
using Statistics

# ==========================================
# 1. VOCABULARY
# ==========================================
# Ordered LONGEST PHRASE FIRST: "the left side of the six yard box" must be tested before "the left
# side of the box", and "a difficult angle and long range" before "a difficult angle".
const PM_ZONE_PATTERNS = [
    "the left side of the six yard box"  => :six_yard_side,      #    870
    "the right side of the six yard box" => :six_yard_side,      #    903
    "a difficult angle and long range"   => :difficult_long,     #    168
    "the centre of the box"              => :box_centre,         # 16,309
    "the left side of the box"           => :box_side,           #  3,462
    "the right side of the box"          => :box_side,           #  3,231
    "a difficult angle on the left"      => :difficult_angle,    #    384
    "a difficult angle on the right"     => :difficult_angle,    #    405
    "very close range"                   => :six_yard_centre,    #  1,980
    "more than 35 yards"                 => :very_long_range,    #    306
    "more than 40 yards"                 => :very_long_range,    #     33
    "long range on the left"             => :long_range,         #    100
    "long range on the right"            => :long_range,         #     78
    "outside the box"                    => :outside_box,        # 15,904
    "a free kick"                        => :free_kick_zone,     #    141  -> remapped, see below
]

# THE "a free kick with a ..." TRAP. 141 shots carry the phrase "from a free kick with a right
# footed shot" and similar. On the first research run these converted at 100.0% — BBC uses that
# phrasing ONLY in goal descriptions, so a cell keyed on it predicts xG ~ 1.0 and the model has
# effectively read the outcome off the wording rather than the chance quality. Left in, it was worth
# 2.1 Brier points of spurious gain and would have injected 141 phantom xG=1.0 events straight into
# the xGPM target.
#
# What we actually know about these shots is that they were DIRECT FREE KICKS of unstated location.
# So: keep the context, and give them the modal direct-free-kick location.
const PM_FREE_KICK_FALLBACK_ZONE = :outside_box

const PM_BODY_PATTERNS = [
    "header"       => :header,
    "right footed" => :right_foot,     # 23,400
    "left footed"  => :left_foot,      # 12,383
]

const PM_CONTEXT_PATTERNS = [
    "from a direct free kick"         => :direct_free_kick,   #   751
    "following a set piece situation" => :set_piece,          # 1,364
    "following a corner"              => :corner,             # 3,011
    "following a fast break"          => :fast_break,         #   645
]

# On target = the keeper had to deal with it, or it went in.
const PM_ON_TARGET_EVENTS = Set(["goal", "attempt_saved", "penalty_saved"])
const PM_GOAL_EVENTS      = Set(["goal"])

# ==========================================
# 2. PARSER
# ==========================================
_pm_first_match(text::AbstractString, patterns, default) = begin
    for (needle, label) in patterns
        occursin(needle, text) && return label
    end
    default
end

"""
    parse_shot(event_type, text) -> NamedTuple

`(zone, body_part, context, is_penalty, parsed)`.

`is_penalty` is taken from the event type OR the word "penalty" in the description. Penalties then
get their own bucket and a CONSTANT xG: the base paper tested four model families on 4,420
penalties and none beat the 0.1848 base rate, concluding the outcome is conditionally random given
anything observable.
"""
function parse_shot(event_type::AbstractString, text)
    if ismissing(text)
        return (zone = :unknown, body_part = :unknown, context = :open_play,
                is_penalty = startswith(event_type, "penalty"), parsed = false)
    end
    t = lowercase(String(text))
    is_pen = startswith(event_type, "penalty") || occursin("penalty", t)

    zone = _pm_first_match(t, PM_ZONE_PATTERNS, :unknown)
    body = _pm_first_match(t, PM_BODY_PATTERNS, :unknown)
    ctx  = _pm_first_match(t, PM_CONTEXT_PATTERNS, :open_play)

    # See PM_FREE_KICK_FALLBACK_ZONE above: this phrasing is outcome-confounded, so keep the context
    # it genuinely tells us and fall back to the modal free-kick location.
    if zone === :free_kick_zone
        ctx  = :direct_free_kick
        zone = PM_FREE_KICK_FALLBACK_ZONE
    end

    return (zone = zone, body_part = body, context = ctx, is_penalty = is_pen,
            parsed = zone !== :unknown || is_pen)
end

# ==========================================
# 3. THE SHOT TABLE
# ==========================================
"""
    build_shots(ds) -> DataFrame

One row per shot with its parsed descriptors, the side that took it, the match minute, and the
binary outcomes. `is_home` comes from `ds.bbc_events.is_home_event`, which the fetcher derives by
joining the BBC team slug to `match_meta`'s home/away slugs — deterministic, unlike inferring the
side from the running score.

Rows whose side could not be resolved keep `missing` and are dropped downstream (2.44% in the
research).
"""
function build_shots(ds::Data.DataStore)
    ev = ds.bbc_events
    (nrow(ev) == 0 || !("event_type" in names(ev))) &&
        return DataFrame(match_id = Int[], time = Union{Missing,Int}[],
                         added_time = Union{Missing,Int}[], event_type = String[],
                         is_home = Union{Missing,Bool}[], zone = Symbol[], body_part = Symbol[],
                         context = Symbol[], is_penalty = Bool[], parsed = Bool[],
                         is_goal = Bool[], is_on_target = Bool[], xg = Float64[])

    p = parse_shot.(String.(ev.event_type), ev.text)
    out = DataFrame(
        match_id   = Int.(ev.match_id),
        time       = ev.time,
        added_time = ev.added_time,
        event_type = String.(ev.event_type),
        is_home    = ev.is_home_event,
        zone       = [x.zone       for x in p],
        body_part  = [x.body_part  for x in p],
        context    = [x.context    for x in p],
        is_penalty = [x.is_penalty for x in p],
        parsed     = [x.parsed     for x in p],
    )
    out.is_goal      = in.(out.event_type, Ref(PM_GOAL_EVENTS))
    out.is_on_target = in.(out.event_type, Ref(PM_ON_TARGET_EVENTS))
    return out
end

# ==========================================
# 4. THE xG MODEL
# ==========================================
"""
    ShotXGModel

A lookup table, not a fitted GLM object: `P(goal | cell)` with empirical-Bayes shrinkage toward the
overall base rate. `k` is the pseudo-count — a cell with `k` shots is pulled halfway to the base
rate — which keeps rare cells (e.g. headers from long range) from producing 0 or 1.

Why a table rather than a logistic regression: the feature space is three small closed factors, so
the saturated cell model IS the full interaction, and shrinkage handles the sparse cells more
transparently than a regularised GLM would.
"""
struct ShotXGModel
    cells::Dict{Tuple{Symbol, Symbol, Symbol}, Float64}
    base_rate::Float64
    penalty_xg::Float64
    k::Float64
end

"""
    fit_shot_xg(shots; k=25.0) -> ShotXGModel

Fit the saturated (zone x body x context) cell table with empirical-Bayes shrinkage.
"""
function fit_shot_xg(shots::DataFrame; k::Float64 = 25.0)
    nrow(shots) == 0 && return ShotXGModel(Dict{Tuple{Symbol,Symbol,Symbol}, Float64}(),
                                           0.1, 0.76, k)
    open_play = shots[.!shots.is_penalty .& shots.parsed, :]
    base = isempty(open_play) ? 0.1 : mean(open_play.is_goal)

    num = Dict{Tuple{Symbol,Symbol,Symbol}, Float64}()
    den = Dict{Tuple{Symbol,Symbol,Symbol}, Float64}()
    for r in eachrow(open_play)
        kk = (r.zone, r.body_part, r.context)
        num[kk] = get(num, kk, 0.0) + (r.is_goal ? 1.0 : 0.0)
        den[kk] = get(den, kk, 0.0) + 1.0
    end
    cells = Dict{Tuple{Symbol,Symbol,Symbol}, Float64}()
    for (kk, n) in den
        cells[kk] = (num[kk] + k * base) / (n + k)          # empirical-Bayes shrinkage
    end

    pens = shots[shots.is_penalty, :]
    pen_xg = nrow(pens) == 0 ? 0.76 : mean(pens.is_goal)     # ONE constant, per the base paper
    return ShotXGModel(cells, base, pen_xg, k)
end

"""
    predict_xg(model, shots) -> Vector{Float64}

Unseen cells fall back to the base rate; unparsed shots also get the base rate, so a parse failure
degrades to "an average shot" rather than dropping the attempt entirely.
"""
function predict_xg(m::ShotXGModel, shots::DataFrame)
    return [r.is_penalty ? m.penalty_xg :
            (r.parsed ? get(m.cells, (r.zone, r.body_part, r.context), m.base_rate) : m.base_rate)
            for r in eachrow(shots)]
end
