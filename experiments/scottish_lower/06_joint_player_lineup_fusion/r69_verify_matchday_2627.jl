# ==============================================================================
# r69 — Post-extension audit: database persistence + MatchDay operational readiness
# ==============================================================================
#
# Companion to `r68_extend_joint_player_2627.jl`. r68 samples Folds 41-43 and writes them back
# to the immutable run UUID; this runner is the independent check that the write landed and that
# Saturday's pricing path can condition on it.
#
# It separates two questions r68's own audit line conflates:
#
#   * Did the EXTENSION converge? — folds 41-43 only. This is what the extension is answerable
#     for, and the only part of the fit that changed today.
#   * Does the WHOLE 43-fold run clear the gate MatchDay applies? — folds 1-43. A historical fold
#     that has always been marginal fails this without the extension having done anything wrong,
#     and the distinction decides whether the right response is "resample fold 41" or "this run
#     was never loadable with require_converged = true".
#
# Read-only against both databases. Run it on mcmc-beast, in the same REPL as r68:
#
#   include("experiments/scottish_lower/06_joint_player_lineup_fusion/r69_verify_matchday_2627.jl")
#   r69_verify_matchday_2627()

# The deserialization shim and the widened splitter live in r68; reuse them rather than keeping a
# second copy that can drift.
isdefined(Main, :r68_splitter_2627) ||
    include(joinpath(@__DIR__, "r68_extend_joint_player_2627.jl"))

using BayesianFootball
using DataFrames, Dates, Printf, Statistics, UUIDs
import LibPQ

const EV = BayesianFootball.Evaluation

const R69_EXPERIMENT = "scottish_lower_joint_player_2426"

const R69_RUNS = [
    ("m12_joint_hybrid_synergy",    "132df5c2-c742-4e95-8693-3aeb2b2cbaef"),
    ("m05_joint_production_wealth", "ed541a7c-01e2-447e-a771-783517728d47"),
]

# Saturday 2026-09-05, 15:00 BST = 14:00 UTC. `AS_OF` is T-25, the instant the runbook prices at:
# the confirmed XI has landed (T-13..T-42) and it is still clear of the T-12 submission target.
const R69_KO_DAY = Date(2026, 9, 5)
const R69_AS_OF  = DateTime(2026, 9, 5, 13, 35)

const R69_HISTORICAL_FOLDS = 40
const R69_EXPECTED_FOLDS   = 43
const R69_EXPECTED_OOS     = 759        # 710 through 25/26 + 49 played in August 2026
const R69_NEW_FOLDS        = 41:43

# The thresholds the work brief names. They are NOT the library's gates -- `convergence_verdict`
# asks for tail ESS > 400 -- so both are reported, separately and by name, rather than one being
# quietly substituted for the other.
const R69_MAX_RHAT = 1.05
const R69_MIN_ESS  = 100.0

r69_mark(ok) = ok ? "PASS" : "FAIL"

# ==============================================================================
# 1. The reloaded fit
# ==============================================================================
"""
    r69_audit_fit(db, uuid) -> NamedTuple

Reload the run from `fit_artifacts` and unroll its per-fold convergence record.

The per-fold numbers live on `Fit.diagnostics.folds` (a `Vector{FoldConvergence}`), not on the
`FoldFit`s. They are unrolled rather than read off the summary's reductions because a reduction
cannot say WHICH fold is bad, and "fold 21" versus "fold 42" is the whole difference between a
pre-existing condition and a broken extension.
"""
function r69_audit_fit(db, uuid::AbstractString)
    fit = TT.load_fit(db, uuid)
    passed, gates, detail = EV.convergence_verdict(fit)

    rows = DataFrame(fold = Int[], new = Bool[], applicable = Bool[], rhat = Float64[],
                     ess_bulk = Float64[], ess_tail = Float64[], divergences = Int[],
                     worst_rhat_param = Symbol[])
    for f in fit.diagnostics.folds
        push!(rows, (f.fold, f.fold in R69_NEW_FOLDS, f.applicable, f.max_rhat,
                     f.min_ess_bulk, f.min_ess_tail, f.n_divergent, f.worst_rhat_param))
    end

    return (; fit, n_folds = length(fit.folds), n_oos = TT.n_matches(fit.latents),
              passed, gates, detail, folds = rows)
end

"Reduce a per-fold diagnostic frame to the four numbers the brief gates on."
function r69_reduce(rows::AbstractDataFrame)
    isempty(rows) && return (; max_rhat = NaN, min_bulk = NaN, min_tail = NaN, div = 0,
                               worst_rhat_fold = 0, worst_tail_fold = 0,
                               divergent_folds = Int[])
    return (; max_rhat = maximum(rows.rhat),
              min_bulk = minimum(rows.ess_bulk),
              min_tail = minimum(rows.ess_tail),
              div      = sum(rows.divergences),
              worst_rhat_fold = rows.fold[argmax(rows.rhat)],
              worst_tail_fold = rows.fold[argmin(rows.ess_tail)],
              divergent_folds = rows.fold[rows.divergences .> 0])
end

# ==============================================================================
# 2. The relational rows
# ==============================================================================
"""
    r69_audit_db(db, uuid) -> NamedTuple

Count what `mcmc_experiments` holds for this run, independently of the serialized artifact.

`fold_results` and `match_latents` are the queryable half of a run; `fit_artifacts` is the exact
half. MatchDay reads the artifact and every diagnostic query reads the rows, so a run where
`extend_fit` wrote one and not the other would price from a 43-fold posterior while every report
about it described 40 folds -- and nothing would raise. That is the failure this counts for.
"""
function r69_audit_db(db, uuid::AbstractString)
    conn = LibPQ.Connection(db.conn_str)
    try
        folds = DataFrame(LibPQ.execute(conn, """
            SELECT fold_idx, r_hat_max, ess_bulk_min, ess_tail_min, divergences, converged,
                   n_matches, first_match_date, last_match_date
            FROM fold_results WHERE run_id = \$1 ORDER BY fold_idx;""", (uuid,)))
        latents = DataFrame(LibPQ.execute(conn, """
            SELECT count(*) AS n
            FROM match_latents ml JOIN fold_results fr ON ml.fold_id = fr.fold_id
            WHERE fr.run_id = \$1;""", (uuid,)))
        artifact = DataFrame(LibPQ.execute(conn, """
            SELECT octet_length(fit_blob) AS bytes FROM fit_artifacts WHERE run_id = \$1;""",
            (uuid,)))
        run = DataFrame(LibPQ.execute(conn, """
            SELECT name, status, git_branch, git_commit, created_at, finished_at,
                   duration_seconds
            FROM runs WHERE run_id = \$1;""", (uuid,)))
        return (; folds, n_latents = isempty(latents) ? 0 : Int(latents.n[1]),
                  artifact_bytes = isempty(artifact) ? 0 : Int(artifact.bytes[1]),
                  run)
    finally
        close(conn)
    end
end

# ==============================================================================
# 3. Tomorrow's card
# ==============================================================================
"""
    r69_tomorrow_fixtures() -> Vector{MD.Fixture}

The unplayed 2026-09-05 slate, straight from `sofascore.events` via `SofaScoreEvents`.

A 12-hour horizon off a 13:35 `as_of` covers the 14:00 UTC kick-offs and any evening fixture
without reaching into Sunday.
"""
r69_tomorrow_fixtures() =
    MD.fixtures(MD.SofaScoreEvents(horizon = Hour(12)), DD.ScottishLower(), R69_AS_OF)

"""
    r69_audit_matchday(cf, ds, fixtures) -> NamedTuple

The two questions Saturday actually turns on: which fold, and does it price.

`select_split` is called exactly as `matchday_latents` calls it -- positive identification via
`get_next_matches`, `exclude` as the fallback -- because verifying a different call would verify
a different code path than the one that runs at T-25.

The `matchday_latents` call is caught rather than allowed to propagate. A crash there is a
finding about the SERVING path, and letting it abort the run would also destroy the fold
selection and feature-coverage evidence that says whether the TRAINING side is sound -- which is
the distinction this whole runner exists to draw.

Also reports coverage of the two per-match maps by name. `:player_lineup_ratings_map` is what
`Models.PreGame`'s builder engine reads for a `PlayerLineupPillar` at OOS
(`builder/engine.jl`, `get(d, :player_lineup_ratings_map, Dict())`), and it is NOT in
`MatchDay.INJECTABLE_KEYS` -- so it is neither materialised for an unseen fixture nor checked by
`check_coverage`. An uncovered fixture falls back to `_pm_empty_lineup_aggregate()`, i.e. the
lineup pillar contributes exactly zero, silently. That is worth measuring explicitly rather than
inferring from a stack trace.
"""
function r69_audit_matchday(cf, ds, fx::Vector{MD.Fixture})
    boundaries = DD.create_id_boundaries(ds, cf.config.splitter)
    ids = [f.m_id for f in fx]
    sel = MD.select_split(cf.fit, boundaries; exclude = ids, ds = ds,
                          config = cf.config.splitter, fixture_ids = ids)

    spec = MD.MatchDaySpec(
        fixtures = MD.ExplicitFixtures(fx),
        identity = MD.ResolverChain(MD.MatchMetaCrosswalk(), MD.LiveNameMatch()),
        lineups  = MD.SourceChain(MD.ProvisionalDB(), MD.LastHistorical(ds)))
    cards = MD.build_cards(spec, DD.ScottishLower(), R69_AS_OF)

    fcol = BayesianFootball.Features.create_features(boundaries, ds, cf.config.model,
                                                     cf.config.splitter)
    fs = fcol[sel.idx][1]

    team_map = fs.data[:team_map]
    teams = unique(vcat([f.home for f in fx], [f.away for f in fx]))
    missing_teams = [t for t in teams if !haskey(team_map, t)]

    map_cover = Pair{Symbol,Int}[]
    for key in (:player_ratings_map, :player_lineup_ratings_map, :league_lookup)
        haskey(fs.data, key) || continue
        m = fs.data[key]
        push!(map_cover, key => count(i -> haskey(m, i), ids))
    end

    # `matchday_latents` takes the odds frame only for `MarketPillarFromBook`, which is not in the
    # default materialiser chain these models need; an empty frame keeps the exchange out of a
    # check that is about the posterior, not about the book.
    latents, diag, err = try
        l, d = MD.matchday_latents(spec, cf.fit, ds, cards, DataFrame(), R69_AS_OF)
        (l, d, nothing)
    catch e
        (DataFrame(), nothing, e)
    end

    return (; sel, cards, latents, diag, err, boundaries, fs,
              n_boundaries = length(boundaries), team_map_size = length(team_map),
              missing_teams, map_cover)
end

# ==============================================================================
# Driver
# ==============================================================================
function r69_verify_matchday_2627(; refresh::Bool = false)
    println("="^100)
    println("  POST-EXTENSION AUDIT — 43-FOLD FITS AND SATURDAY ", R69_KO_DAY, " READINESS")
    println("="^100)

    db = TT.PostgresStorage(R69_EXPERIMENT)
    ds = refresh ? DD.load_datastore_cached(DD.ScottishLower(); force = true) :
                   DD.load_datastore_cached(DD.ScottishLower(); max_age_hours = 100_000)
    println("  storage  : ", db)
    println("  DataStore: ", length(ds.matches.match_id), " matches")

    results = Dict{String,Any}()
    extension_ok = true          # did folds 41-43 land, converged, and persist?

    for (name, uuid) in R69_RUNS
        println("\n", "-"^100)
        println("  ", name, "  (", uuid, ")")
        println("-"^100)

        a = r69_audit_fit(db, uuid)
        d = r69_audit_db(db, uuid)

        new_rows = a.folds[a.folds.new, :]
        old_rows = a.folds[.!a.folds.new, :]
        new = r69_reduce(new_rows)
        old = r69_reduce(old_rows)
        all_ = r69_reduce(a.folds)

        # --- persistence: the extension's own responsibility ---------------------------------
        persistence = [
            ("Fit carries $(R69_EXPECTED_FOLDS) folds",       a.n_folds == R69_EXPECTED_FOLDS),
            ("OOS fixtures == $(R69_EXPECTED_OOS)",           a.n_oos == R69_EXPECTED_OOS),
            ("fold_results rows == $(R69_EXPECTED_FOLDS)",    nrow(d.folds) == R69_EXPECTED_FOLDS),
            ("match_latents rows == $(R69_EXPECTED_OOS)",     d.n_latents == R69_EXPECTED_OOS),
            ("fit_artifacts blob rewritten",                  d.artifact_bytes > 0),
            ("fold_results agrees with Fit on R̂",             nrow(d.folds) == a.n_folds &&
                 all(isapprox.(d.folds.r_hat_max, a.folds.rhat; atol = 1e-6))),
        ]

        # --- convergence of the NEW folds -----------------------------------------------------
        new_gates = [
            ("folds 41-43 present",                new.div isa Int && nrow(new_rows) == 3),
            ("max split R̂ <= $(R69_MAX_RHAT)",     new.max_rhat <= R69_MAX_RHAT),
            ("min bulk ESS >= $(R69_MIN_ESS)",     new.min_bulk >= R69_MIN_ESS),
            ("min tail ESS >= $(R69_MIN_ESS)",     new.min_tail >= R69_MIN_ESS),
            ("divergences == 0",                   new.div == 0),
        ]

        # --- the whole run, against the brief's gates and the library's ------------------------
        run_gates = [
            ("all 43: max R̂ <= $(R69_MAX_RHAT)",   all_.max_rhat <= R69_MAX_RHAT),
            ("all 43: min bulk ESS >= $(R69_MIN_ESS)", all_.min_bulk >= R69_MIN_ESS),
            ("all 43: min tail ESS >= $(R69_MIN_ESS)", all_.min_tail >= R69_MIN_ESS),
            ("all 43: divergences == 0",           all_.div == 0),
            ("Evaluation.convergence_verdict",     a.passed),
        ]

        println("\n  PERSISTENCE (what the extension wrote)")
        for (l, ok) in persistence
            @printf("    %-46s %s\n", l, r69_mark(ok)); extension_ok &= ok
        end

        println("\n  CONVERGENCE — NEW FOLDS 41-43 ONLY (the extension's own chains)")
        for (l, ok) in new_gates
            @printf("    %-46s %s\n", l, r69_mark(ok)); extension_ok &= ok
        end
        @printf("    max R̂ %.4f | bulk ESS %.1f | tail ESS %.1f | divergences %d\n",
                new.max_rhat, new.min_bulk, new.min_tail, new.div)

        println("\n  CONVERGENCE — ALL 43 FOLDS (what MatchDay's gate sees)")
        for (l, ok) in run_gates
            @printf("    %-46s %s\n", l, r69_mark(ok))
        end
        @printf("    max R̂ %.4f (fold %d) | bulk ESS %.1f | tail ESS %.1f (fold %d) | divergences %d\n",
                all_.max_rhat, all_.worst_rhat_fold, all_.min_bulk,
                all_.min_tail, all_.worst_tail_fold, all_.div)
        isempty(all_.divergent_folds) ||
            println("    folds carrying divergences: ", join(all_.divergent_folds, ", "),
                    "   (historical: ", join(old.divergent_folds, ", "),
                    " | new: ", join(new.divergent_folds, ", "), ")")
        a.passed || println("    convergence_verdict failed on: ", join(a.gates, ", "),
                            "\n      ", join(a.detail, "\n      "))

        println("\n  Folds 41-43 as stored in fold_results")
        show(stdout, MIME"text/plain"(), last(d.folds, 3); allrows = true)
        println()

        results[name] = (; audit = a, db = d, new, old, all = all_,
                           persistence, new_gates, run_gates)
    end

    # ------------------------------------------------------------------
    # MatchDay readiness, on the production standard
    # ------------------------------------------------------------------
    println("\n", "="^100)
    println("  MATCHDAY READINESS — ", R69_KO_DAY, " (as_of ", R69_AS_OF, " UTC)")
    println("="^100)

    prod_name, prod_uuid = R69_RUNS[1]
    cf = MD.canonical_fit(db, prod_uuid; require_converged = true)
    MD.matchday_fit_report(cf)

    fx = r69_tomorrow_fixtures()
    println("\n  fixtures on the card: ", length(fx))
    for f in fx
        @printf("    %-9d %-24s vs %-24s  %s  t%d\n",
                f.m_id, f.home, f.away, f.kickoff, f.tournament_id)
    end

    m = r69_audit_matchday(cf, ds, fx)

    println("\n  select_split")
    @printf("    chose fold %d of %d rebuilt boundaries (fit carries %d folds)\n",
            m.sel.idx, m.n_boundaries, cf.n_folds)
    println("    warning: ", isempty(m.sel.warning) ? "(none)" : m.sel.warning)

    with_lineup = count(c -> c.lineup !== nothing, m.cards)
    resolved_id = count(MD.resolved, m.cards)

    println("\n  Fold ", m.sel.idx, " FeatureSet coverage of tomorrow's ", length(fx), " fixtures")
    @printf("    %-46s %d teams, missing: %s\n", "team_map",
            m.team_map_size, isempty(m.missing_teams) ? "(none)" : join(m.missing_teams, ", "))
    for (key, n) in m.map_cover
        injectable = key in MD.INJECTABLE_KEYS
        @printf("    %-46s %d/%d covered%s\n", string(key), n, length(fx),
                injectable ? "  (materialised by MatchDay)" :
                             "  (NOT in INJECTABLE_KEYS — carried forward from training)")
    end

    if m.err !== nothing
        println("\n  matchday_latents RAISED:")
        for line in split(sprint(showerror, m.err), "\n")
            println("    ", line)
        end
    end

    # `extract_parameters` returns POSTERIOR DRAWS per fixture (`:λ_h`, `:λ_a` are
    # `Vector{Float64}`, one entry per draw), not point estimates -- so every check reduces over
    # the draws rather than over the column. A single non-finite draw is a broken posterior even
    # when the mean looks reasonable, which is exactly what a mean would hide.
    _finite(x::AbstractVector) = all(isfinite, x)
    _finite(x::Real)           = isfinite(x)
    _positive(x::AbstractVector) = all(>(0), x)
    _positive(x::Real)           = x > 0

    value_cols = [c for c in propertynames(m.latents) if c !== :match_id]
    rate_cols  = [c for c in value_cols
                  if occursin("λ", String(c)) || occursin("lambda", String(c))]
    finite_rates = !isempty(rate_cols) && nrow(m.latents) > 0 &&
        all(all(_finite, m.latents[!, c]) for c in value_cols) &&
        all(all(_positive, m.latents[!, c]) for c in rate_cols)

    md_gates = [
        ("canonical_fit(require_converged=true) loads", true),   # reaching here proves it
        ("canonical_fit reports converged",             cf.converged),
        ("canonical_fit folds == $(R69_EXPECTED_FOLDS)", cf.n_folds == R69_EXPECTED_FOLDS),
        ("card has 10 fixtures",                        length(fx) == 10),
        ("select_split chose fold $(R69_EXPECTED_FOLDS)", m.sel.idx == R69_EXPECTED_FOLDS),
        ("select_split warning empty",                  isempty(m.sel.warning)),
        ("identity resolved for every fixture",         resolved_id == length(fx)),
        ("lineup available for every fixture",          with_lineup == length(fx)),
        ("every team on the card is in team_map",       isempty(m.missing_teams)),
        ("matchday_latents ran without raising",        m.err === nothing),
        ("matchday_latents priced every fixture",       nrow(m.latents) == length(fx)),
        ("all rates finite and positive",               finite_rates),
    ]
    println("\n  Operational gates")
    matchday_ok = true
    for (l, ok) in md_gates
        @printf("    %-46s %s\n", l, r69_mark(ok)); matchday_ok &= ok
    end

    if !isempty(rate_cols)
        println("\n  Posterior rates (mean over draws)")
        by_id = Dict(f.m_id => f for f in fx)
        summ = DataFrame(match_id = m.latents.match_id,
                         fixture = [string(by_id[i].home, " v ", by_id[i].away)
                                    for i in m.latents.match_id])
        for c in rate_cols
            summ[!, Symbol("mean_", c)] = [mean(v) for v in m.latents[!, c]]
        end
        # Only for the goals arm. A two-arm observation can carry a second pair of rates, and
        # adding those into one "total" would print a number that means nothing.
        if hasproperty(m.latents, :λ_h) && hasproperty(m.latents, :λ_a)
            summ.total_goals = summ.mean_λ_h .+ summ.mean_λ_a
        end
        show(stdout, MIME"text/plain"(), summ; allrows = true)
        println()
    end

    println("\n", "="^100)
    println("  EXTENSION : ", extension_ok ? "CLEAN — folds 41-43 converged and persisted." :
                                             "PROBLEM — see the FAIL lines above.")
    println("  MATCHDAY  : ", matchday_ok ? "READY — $(prod_name) prices tomorrow's card." :
                                            "NOT READY — see the FAIL lines above.")
    println("="^100)

    return (; db, ds, results, cf, fixtures = fx, matchday = m,
              extension_ok, matchday_ok)
end

if abspath(PROGRAM_FILE) == @__FILE__
    r69_verify_matchday_2627()
end
