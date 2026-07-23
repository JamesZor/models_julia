# current_development/plus_minus_ratings/l02_shot_parser.jl
#
# LOADER (temporary module). WP3a — turn BBC commentary into structured shot records.
#
# BBC's live text is Opta-derived and follows a rigid template:
#
#   "Attempt saved. Kai Kennedy (Queen of the South) right footed shot from outside the box
#    is saved in the top right corner by Robbie Mutch (Cove Rangers)."
#   "Goal! Cove Rangers 0, Queen of the South 1. Kurtis Guthrie (Queen of the South) header
#    from very close range to the bottom left corner. Assisted by Jack Hannah following a corner."
#
# so `<shooter> (<team>) <BODY PART> from <ZONE> <outcome> [... following a <CONTEXT>]`.
# That gives us zone, body part and set-piece context — i.e. everything a pre-tracking-era xG
# model used — WITHOUT any coordinates. This is the only route to xG for tiers 56/57, which
# have no SofaScore shot data at all.
#
# VOCABULARY IS EMPIRICAL, NOT GUESSED. Scanned over all 45,201 shot-bearing rows across tiers
# 54-57 (2026-07-23); the zone phrase matched on 98.1% of them and the vocabulary below is
# closed. Counts are recorded next to each entry so drift is detectable on a re-scrape.
#
# DELIBERATE DESIGN CHOICE — match on KEYWORDS, not on capture position. An earlier positional
# regex silently conflated "header" with an empty body-part capture, and swallowed
# "from a direct free kick" into the zone. Keyword search over a closed vocabulary is duller
# and far harder to get quietly wrong.

using DataFrames

include(joinpath(@__DIR__, "l00_pm_data.jl"))

# ==========================================
# 1. VOCABULARY  (counts from the 2026-07-23 scan, tiers 54-57)
# ==========================================
# Ordered LONGEST PHRASE FIRST: "the left side of the six yard box" must be tested before
# "the left side of the box", and "a difficult angle and long range" before "a difficult angle".
const ZONE_PATTERNS = [
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
    "a free kick"                        => :free_kick_zone,     #    141
]

const BODY_PATTERNS = [
    "header"       => :header,
    "right footed" => :right_foot,     # 23,400
    "left footed"  => :left_foot,      # 12,383
]

const CONTEXT_PATTERNS = [
    "from a direct free kick"         => :direct_free_kick,   #   751
    "following a set piece situation" => :set_piece,          # 1,364
    "following a corner"              => :corner,             # 3,011
    "following a fast break"          => :fast_break,         #   645
]

# Shot-bearing event types, and which of them put the keeper to work.
const SHOT_EVENTS = ["goal", "attempt_missed", "attempt_saved", "attempt_blocked", "post",
                     "penalty_missed", "penalty_saved"]
const ON_TARGET_EVENTS = Set(["goal", "attempt_saved", "penalty_saved"])
const GOAL_EVENTS      = Set(["goal"])

# ==========================================
# 2. PARSER
# ==========================================
_first_match(text::AbstractString, patterns, default) = begin
    for (needle, label) in patterns
        occursin(needle, text) && return label
    end
    default
end

"""
    parse_shot(event_type, text) -> NamedTuple

`(zone, body_part, context, is_penalty, parsed)`.

`is_penalty` is taken from the event type OR the word "penalty" in the description. Penalties
are then given their own bucket and a CONSTANT xG: the base paper tested four model families on
4,420 penalties and **none beat the 0.1848 base rate**, concluding the outcome is conditionally
random given anything we can observe. Modelling them is wasted effort (RESEARCH_rapm.md §5.1).
"""
function parse_shot(event_type::AbstractString, text)
    if ismissing(text)
        return (zone = :unknown, body_part = :unknown, context = :open_play,
                is_penalty = startswith(event_type, "penalty"), parsed = false)
    end
    t = lowercase(String(text))
    is_pen = startswith(event_type, "penalty") || occursin("penalty", t)

    zone = _first_match(t, ZONE_PATTERNS, :unknown)
    body = _first_match(t, BODY_PATTERNS, :unknown)
    ctx  = _first_match(t, CONTEXT_PATTERNS, :open_play)

    # A direct free kick is described as such, and its zone phrase is often the free-kick
    # boilerplate rather than a pitch location — normalise so the zone factor stays meaningful.
    zone === :free_kick_zone && (ctx = :direct_free_kick)

    return (zone = zone, body_part = body, context = ctx, is_penalty = is_pen,
            parsed = zone !== :unknown || is_pen)
end

"""
    strip_name(s) -> String

Normalise a player name for matching BBC's `player` field against SofaScore's `player_name`.
Lowercase, drop punctuation and accents, collapse whitespace. Used ONLY for the WP3 player-level
calibration, which doubles as the measurement of the name→player_id risk WP1 flagged.
"""
function strip_name(s)
    ismissing(s) && return ""
    t = lowercase(String(s))
    for (a, b) in ("á"=>"a","à"=>"a","â"=>"a","ä"=>"a","ã"=>"a","å"=>"a","é"=>"e","è"=>"e",
                   "ê"=>"e","ë"=>"e","í"=>"i","ì"=>"i","î"=>"i","ï"=>"i","ó"=>"o","ò"=>"o",
                   "ô"=>"o","ö"=>"o","õ"=>"o","ú"=>"u","ù"=>"u","û"=>"u","ü"=>"u","ñ"=>"n",
                   "ç"=>"c","ø"=>"o","š"=>"s","ž"=>"z","ć"=>"c","č"=>"c","đ"=>"d")
        t = replace(t, a => b)
    end
    t = replace(t, r"[^a-z ]" => "")
    return join(split(t), " ")
end

# ==========================================
# 3. THE SHOT TABLE
# ==========================================
"""
    build_shots(; tournaments) -> DataFrame

One row per shot with its parsed descriptors, the side that took it, the match minute, and the
binary outcome. `is_home_event` comes from joining BBC's team slug to `match_meta`'s home/away
slugs — deterministic, unlike inferring the side from the running score.
"""
function build_shots(; tournaments::Vector{Int} = PM_TIERS)
    ensure_pm_data!()
    lt = PM_LIVETEXT[]
    keep = coalesce.(in.(lt.event_type, Ref(SHOT_EVENTS)), false) .&
           coalesce.(in.(lt.tournament_id, Ref(tournaments)), false)
    sh = lt[keep, :]

    p = parse_shot.(String.(sh.event_type), sh.text)
    out = DataFrame(
        match_id      = Int.(sh.match_id),
        tournament_id = Int.(sh.tournament_id),
        season        = String.(sh.season),
        time          = sh.time,
        added_time    = sh.added_time,
        event_type    = String.(sh.event_type),
        is_home       = sh.is_home_event,
        shooter       = strip_name.(sh.player),
        zone          = [x.zone       for x in p],
        body_part     = [x.body_part  for x in p],
        context       = [x.context    for x in p],
        is_penalty    = [x.is_penalty for x in p],
        parsed        = [x.parsed     for x in p],
    )
    out.is_goal     = in.(out.event_type, Ref(GOAL_EVENTS))
    out.is_on_target = in.(out.event_type, Ref(ON_TARGET_EVENTS))
    return out
end

# ==========================================
# 4. THE xG MODEL
# ==========================================
"""
    ShotXGModel

A lookup table, not a fitted GLM object: `P(goal | cell)` with empirical-Bayes shrinkage toward
the overall base rate. `k` is the pseudo-count — a cell with `k` shots is pulled halfway to the
base rate — which keeps rare cells (e.g. headers from long range) from producing 0 or 1.

Why a table rather than a logistic regression: the feature space is three small closed factors,
so the saturated cell model IS the full interaction, and shrinkage handles the sparse cells more
transparently than a regularised GLM would. `r02` still ladders it against simpler nested forms
so the added structure has to earn its place.
"""
struct ShotXGModel
    cells::Dict{Tuple{Symbol, Symbol, Symbol}, Float64}
    base_rate::Float64
    penalty_xg::Float64
    k::Float64
end

cell_key(r) = (r.zone, r.body_part, r.context)

"""
    fit_shot_xg(shots; k=25.0, features=(:zone,:body_part,:context)) -> ShotXGModel

`features` selects which factors form the cell key; the excluded ones collapse to `:_`. That is
what lets `r02` run the nested ladder (base → zone → zone+body → zone+body+context) through one
code path.
"""
function fit_shot_xg(shots::DataFrame; k::Float64 = 25.0,
                     features::NTuple{3, Symbol} = (:zone, :body_part, :context))
    open_play = shots[.!shots.is_penalty .& shots.parsed, :]
    base = isempty(open_play) ? 0.1 : mean(open_play.is_goal)

    key(r) = (:zone      in features ? r.zone      : :_,
              :body_part in features ? r.body_part : :_,
              :context   in features ? r.context   : :_)

    num = Dict{Tuple{Symbol,Symbol,Symbol}, Float64}()
    den = Dict{Tuple{Symbol,Symbol,Symbol}, Float64}()
    for r in eachrow(open_play)
        kk = key(r)
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

Unseen cells fall back to the base rate; unparsed shots also get the base rate, so a parse
failure degrades to "an average shot" rather than dropping the attempt entirely.
"""
function predict_xg(m::ShotXGModel, shots::DataFrame;
                    features::NTuple{3, Symbol} = (:zone, :body_part, :context))
    key(r) = (:zone      in features ? r.zone      : :_,
              :body_part in features ? r.body_part : :_,
              :context   in features ? r.context   : :_)
    return [r.is_penalty ? m.penalty_xg :
            (r.parsed ? get(m.cells, key(r), m.base_rate) : m.base_rate)
            for r in eachrow(shots)]
end

brier(p, y) = mean((p .- y) .^ 2)
logloss(p, y; eps = 1e-9) = -mean(y .* log.(clamp.(p, eps, 1 - eps)) .+
                                  (1 .- y) .* log.(clamp.(1 .- p, eps, 1 - eps)))
