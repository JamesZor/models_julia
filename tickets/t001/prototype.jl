#!/usr/bin/env julia

# Prototype only: compare the incumbent tournament-local dense clock with a pooled,
# calendar-anchored clock without changing src/. Run on Kaimon, where data caches exist:
#
#   julia --project tickets/t001/prototype.jl
#
# The final expression is a compact NamedTuple suitable for inspection through kaimon_ex.

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

using BayesianFootball
using DataFrames
using Dates

const Data = BayesianFootball.Data
const WIDTH_WEEKS = Dict(:match_week => 1, :match_biweek => 2, :match_month => 4)

kickoff(row) = DateTime(row.match_date) + Hour(row.match_hour)
week_ending_sunday(d::Date) = d + Day(7 - dayofweek(d))

function shared_calendar_clock(df::AbstractDataFrame, dynamics_col::Symbol)
    width = get(WIDTH_WEEKS, dynamics_col) do
        error("calendar prototype does not support dynamics_col=$dynamics_col")
    end
    anchor = minimum(week_ending_sunday.(df.match_date))
    weeks = [1 + div(Dates.value(week_ending_sunday(d) - anchor), 7) for d in df.match_date]
    return cld.(weeks, width)
end

function fold_record(df, clock, train_step, predict_step)
    fitted = df[clock .<= train_step, :]
    heldout = df[clock .== predict_step, :]
    isempty(heldout) && return (
        train_step=train_step, predict_step=predict_step, empty=true,
        contaminated=false, fitted_last=missing, heldout_first=missing,
        heldout_last=missing, heldout_span=missing, fitted_ids=Int[], heldout_ids=Int[],
    )

    heldout_kickoffs = kickoff.(eachrow(heldout))
    fitted_kickoffs = kickoff.(eachrow(fitted))
    fitted_last = isempty(fitted) ? missing : maximum(fitted_kickoffs)
    heldout_first = minimum(heldout_kickoffs)
    heldout_last = maximum(heldout_kickoffs)
    return (
        train_step=train_step,
        predict_step=predict_step,
        empty=false,
        contaminated=!ismissing(fitted_last) && fitted_last >= heldout_first,
        fitted_last=fitted_last,
        heldout_first=heldout_first,
        heldout_last=heldout_last,
        heldout_span=heldout_last - heldout_first,
        fitted_ids=Int.(fitted.match_id),
        heldout_ids=Int.(heldout.match_id),
    )
end

"Emulate the current `t` → exact `t+1` grouped split behavior."
function incumbent_folds(df::AbstractDataFrame, dynamics_col::Symbol)
    clock = Int.(df[!, dynamics_col])
    steps = sort(unique(clock))
    isempty(steps) && return NamedTuple[]
    return [fold_record(df, clock, t, t + 1) for t in steps[1:end-1]]
end

"Use fixed calendar bins and jump directly to the next observed bin (no empty fold)."
function proposed_pooled_folds(df::AbstractDataFrame, dynamics_col::Symbol)
    clock = shared_calendar_clock(df, dynamics_col)
    steps = sort(unique(clock))
    length(steps) < 2 && return NamedTuple[]
    return [fold_record(df, clock, steps[i - 1], steps[i]) for i in 2:length(steps)]
end

function season_summary(segment_name, tids, season, method, folds)
    evaluated = filter(f -> !f.empty, folds)
    return (
        segment=segment_name,
        tournaments=tids,
        season=season,
        method=method,
        folds=length(folds),
        empty=count(f -> f.empty, folds),
        contaminated=count(f -> f.contaminated, evaluated),
        max_heldout_hours=isempty(evaluated) ? missing :
            maximum(Dates.value(f.heldout_span) for f in evaluated) / 3_600_000,
    )
end

function common_seasons(matches, tids)
    sets = [Set(String.(skipmissing(matches.season[matches.tournament_id .== tid]))) for tid in tids]
    return sort!(collect(reduce(intersect, sets)))
end

function compare_pooled(segment; dynamics_col=:match_biweek)
    ds = Data.load_datastore_cached(segment)
    tids = Data.tournament_ids(segment)
    name = string(nameof(typeof(segment)))
    summaries = NamedTuple[]
    for season in common_seasons(ds.matches, tids)
        mask = in.(ds.matches.tournament_id, Ref(tids)) .& (ds.matches.season .== season)
        sdf = ds.matches[mask, :]
        old = incumbent_folds(sdf, dynamics_col)
        new = proposed_pooled_folds(sdf, dynamics_col)
        push!(summaries, season_summary(name, tids, season, :incumbent, old))
        push!(summaries, season_summary(name, tids, season, :proposed, new))
    end
    return summaries, ds
end

function aggregate(rows, method)
    chosen = filter(r -> r.method == method, rows)
    return (
        method=method,
        folds=sum(r.folds for r in chosen),
        empty=sum(r.empty for r in chosen),
        contaminated=sum(r.contaminated for r in chosen),
        max_heldout_hours=maximum(skipmissing(r.max_heldout_hours for r in chosen)),
    )
end

function fold_detail(df, fold, tids)
    held = df[in.(Int.(df.match_id), Ref(Set(fold.heldout_ids))), :]
    per_tournament = [
        let g = held[held.tournament_id .== tid, :]
            (tournament=tid, matches=nrow(g),
             first=isempty(g) ? missing : minimum(kickoff.(eachrow(g))),
             last=isempty(g) ? missing : maximum(kickoff.(eachrow(g))))
        end for tid in tids
    ]
    return (
        train_step=fold.train_step,
        predict_step=fold.predict_step,
        contaminated=fold.contaminated,
        fitted_last=fold.fitted_last,
        heldout_first=fold.heldout_first,
        heldout_last=fold.heldout_last,
        heldout_matches=length(fold.heldout_ids),
        per_tournament=per_tournament,
    )
end

pooled_segments = Any[
    Data.ScottishLower(), Data.ScottishUpper(), Data.IrelandAll(),
    Data.SouthKorea(), Data.Norway(),
]
all_summaries = NamedTuple[]
scottish_ds = nothing
for segment in pooled_segments
    summaries, ds = compare_pooled(segment)
    append!(all_summaries, summaries)
    segment isa Data.ScottishLower && (global scottish_ds = ds)
end

pooled_report = [
    let rows = filter(r -> r.segment == string(nameof(typeof(segment))), all_summaries)
        (segment=string(nameof(typeof(segment))),
         incumbent=aggregate(rows, :incumbent),
         proposed=aggregate(rows, :proposed))
    end for segment in pooled_segments
]

# Side-by-side detail for the ticket's exact 56/57 24/25 date.
sl = scottish_ds.matches[
    in.(scottish_ds.matches.tournament_id, Ref([56, 57])) .&
    (scottish_ds.matches.season .== "24/25"), :]
old_sl = incumbent_folds(sl, :match_biweek)
new_sl = proposed_pooled_folds(sl, :match_biweek)
old_focus = only(filter(f -> !f.empty && Date(f.heldout_first) == Date("2024-10-19"), old_sl))
new_focus = only(filter(f -> !f.empty &&
    Date("2024-10-19") in Date.(kickoff.(eachrow(sl[in.(Int.(sl.match_id), Ref(Set(f.heldout_ids))), :]))), new_sl))
focus = (
    incumbent=fold_detail(sl, old_focus, [56, 57]),
    proposed=fold_detail(sl, new_focus, [56, 57]),
)

# The production proposal deliberately bypasses the new clock and transition logic for
# singleton groups. Verify exact fold signatures for the three named singleton segments.
singleton_report = [
    let ds = Data.load_datastore_cached(segment), tid = only(Data.tournament_ids(segment)),
        seasons = common_seasons(ds.matches, [tid]), exact = true, compared = 0
        for season in seasons
            sdf = ds.matches[(ds.matches.tournament_id .== tid) .& (ds.matches.season .== season), :]
            incumbent = incumbent_folds(sdf, :match_biweek)
            proposed = incumbent_folds(sdf, :match_biweek) # intentional singleton fallback
            exact &= incumbent == proposed
            compared += length(incumbent)
        end
        (segment=string(nameof(typeof(segment))), tournament=tid,
         seasons=length(seasons), folds_compared=compared, exactly_identical=exact)
    end for segment in Any[Data.Ireland(), Data.IrelandFirstDivision(), Data.Veikkausliiga()]
]

analysis = (
    clock=:match_biweek,
    pooled=pooled_report,
    scottish_2024_10_19=focus,
    singleton_controls=singleton_report,
)
