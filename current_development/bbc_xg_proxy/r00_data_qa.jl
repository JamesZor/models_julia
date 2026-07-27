#=
r00 — WP0 data QA for the bbc xG-proxy stream.

Answers, straight from Postgres (schema `bbc` + `sofascore`), no modelling:
  1. Join integrity: which bbc.match_meta rows fail the sofascore.matches ⋈ seasons
     join, and why (missing season row? cup tie? id drift?). Score agreement check
     (`scores_match`) on the joined set.
  2. Coverage matrix: tier × season fill-rate for every candidate feature stat,
     with `filled=true` treated as missing.
  3. Distribution sanity per tier: home/away means for shots, SoT, corners,
     possession vs known benchmarks (shots/match ≈ 20–26, SoT/match ≈ 8).
  4. bbc ↔ sofascore per-stat agreement on Prem/Champ (both sources present):
     correlation, exact-match rate, MAE. This is the data-quality acceptance gate.
  5. xG target overlap: tier × season counts of matches with bbc stats AND
     sofascore expectedGoals (the WP1 training set).

Run in the kaimon REPL (needs ENV["BF_DB_URL"]):
    include(joinpath(pkgdir(BayesianFootball), "current_development/bbc_xg_proxy/r00_data_qa.jl"))
Results are printed AND collected in the global `QA::Dict{Symbol,Any}`.
=#

using LibPQ
using DataFrames
using Statistics

const _conn = LibPQ.Connection(ENV["BF_DB_URL"])
_q(sql) = DataFrame(execute(_conn, sql))

QA = Dict{Symbol,Any}()

# Candidate universal feature stats (n=4,394 in bbc.match_stats).
# NB some stat_types appear under two stat_cats (e.g. shotsOnTarget in attack +
# goalkeeper view) — always aggregate with max(home_value) after de-dup on
# (match_id, stat_type); values are identical duplicates.
const FEATURE_STATS = ["shotsTotal", "shotsOnTarget", "shotsOffTarget", "shotsBlocked",
                       "hitWoodwork", "cornersWon", "possessionPercentage",
                       "foulsCommitted", "totalYellowCard"]
const _stats_in = join(("'" * s * "'" for s in FEATURE_STATS), ",")

# ==========================================
# 1. JOIN INTEGRITY
# ==========================================
println("="^70, "\n1. JOIN INTEGRITY — bbc.match_meta vs sofascore.matches ⋈ seasons\n", "="^70)

QA[:join] = _q("""
    SELECT count(*)                                  AS n_bbc,
           count(m.match_id)                         AS n_in_matches,
           count(s.season_id)                        AS n_full_join,
           count(*) FILTER (WHERE b.scores_match)    AS n_scores_match
    FROM bbc.match_meta b
    LEFT JOIN sofascore.matches m ON m.match_id = b.match_id
    LEFT JOIN sofascore.seasons s ON s.season_id = m.season_id
                                 AND s.tournament_id = m.tournament_id
""")
show(QA[:join], allcols=true); println()

# Where do the unjoined rows fall? Group by presence + kickoff year.
QA[:unjoined] = _q("""
    SELECT (m.match_id IS NULL)          AS missing_from_matches,
           m.tournament_id,
           extract(year FROM b.kickoff)::int AS ko_year,
           count(*)                      AS n
    FROM bbc.match_meta b
    LEFT JOIN sofascore.matches m ON m.match_id = b.match_id
    LEFT JOIN sofascore.seasons s ON s.season_id = m.season_id
                                 AND s.tournament_id = m.tournament_id
    WHERE s.season_id IS NULL
    GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
""")
println("\nUnjoined rows by cause × tournament × kickoff year:")
show(QA[:unjoined], allrows=true, allcols=true); println()

# Sample of unjoined fixtures for eyeballing (slugs + date).
QA[:unjoined_sample] = _q("""
    SELECT b.match_id, b.kickoff::date AS ko, b.bbc_home_slug, b.bbc_away_slug,
           (m.match_id IS NULL) AS missing_from_matches, m.tournament_id, m.season_id
    FROM bbc.match_meta b
    LEFT JOIN sofascore.matches m ON m.match_id = b.match_id
    LEFT JOIN sofascore.seasons s ON s.season_id = m.season_id
                                 AND s.tournament_id = m.tournament_id
    WHERE s.season_id IS NULL
    ORDER BY b.kickoff LIMIT 15
""")
println("\nSample of unjoined fixtures:")
show(QA[:unjoined_sample], allrows=true, allcols=true); println()

# ==========================================
# 2. COVERAGE MATRIX — tier × season × stat, filled=TRUE ⇒ missing
# ==========================================
println("\n", "="^70, "\n2. COVERAGE — genuine (non-imputed) fill-rate per tier × season × stat\n", "="^70)

QA[:coverage] = _q("""
    WITH st AS (
        SELECT DISTINCT ON (ms.match_id, ms.stat_type)
               ms.match_id, ms.stat_type, ms.filled
        FROM bbc.match_stats ms
        WHERE ms.stat_type IN ($_stats_in)
        ORDER BY ms.match_id, ms.stat_type, ms.filled  -- prefer filled=false row
    )
    SELECT m.tournament_id, s.name AS season, st.stat_type,
           count(*) AS n,
           round(100.0 * count(*) FILTER (WHERE NOT st.filled) / count(*), 1) AS pct_genuine
    FROM st
    JOIN sofascore.matches m ON m.match_id = st.match_id
    JOIN sofascore.seasons s ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
    GROUP BY 1, 2, 3
""")
cov_wide = unstack(QA[:coverage], [:tournament_id, :season], :stat_type, :pct_genuine)
sort!(cov_wide, [:tournament_id, :season])
QA[:coverage_wide] = cov_wide
show(cov_wide, allrows=true, allcols=true); println()

# ==========================================
# 3. DISTRIBUTION SANITY per tier (genuine values only)
# ==========================================
println("\n", "="^70, "\n3. DISTRIBUTION SANITY — per-tier means (genuine values only)\n", "="^70)

QA[:dist] = _q("""
    WITH st AS (
        SELECT DISTINCT ON (ms.match_id, ms.stat_type)
               ms.match_id, ms.stat_type, ms.home_value, ms.away_value
        FROM bbc.match_stats ms
        WHERE ms.stat_type IN ($_stats_in) AND NOT ms.filled
        ORDER BY ms.match_id, ms.stat_type
    )
    SELECT m.tournament_id, st.stat_type,
           count(*)                              AS n,
           round(avg(st.home_value), 2)          AS home_mean,
           round(avg(st.away_value), 2)          AS away_mean,
           round(avg(st.home_value + st.away_value), 2) AS match_mean,
           max(st.home_value + st.away_value)    AS match_max
    FROM st
    JOIN sofascore.matches m ON m.match_id = st.match_id
    WHERE m.tournament_id IN (54, 55, 56, 57)
    GROUP BY 1, 2 ORDER BY 2, 1
""")
show(QA[:dist], allrows=true, allcols=true); println()

# ==========================================
# 4. BBC ↔ SOFASCORE AGREEMENT (Prem/Champ, both sources)
# ==========================================
println("\n", "="^70, "\n4. BBC ↔ SOFASCORE PER-STAT AGREEMENT (acceptance gate)\n", "="^70)

# stat_type mapping bbc → sofascore stat_key
const STAT_MAP = [
    ("shotsOnTarget",        "shotsOnGoal"),
    ("shotsOffTarget",       "shotsOffGoal"),
    ("shotsBlocked",         "blockedScoringAttempt"),
    ("cornersWon",           "cornerKicks"),
    ("possessionPercentage", "ballPossession"),
    ("foulsCommitted",       "fouls"),
    ("totalYellowCard",      "yellowCards"),
]

agree_rows = NamedTuple[]
for (bbc_key, ss_key) in STAT_MAP
    pair = _q("""
        WITH b AS (
            SELECT DISTINCT ON (match_id) match_id, home_value, away_value
            FROM bbc.match_stats WHERE stat_type = '$bbc_key' AND NOT filled
            ORDER BY match_id
        )
        SELECT b.home_value AS bh, b.away_value AS ba,
               ss.home_value::float8 AS sh, ss.away_value::float8 AS sa
        FROM b
        JOIN sofascore.match_statistics ss ON ss.match_id = b.match_id
             AND ss.stat_key = '$ss_key' AND ss.period = 'ALL'
        JOIN sofascore.matches m ON m.match_id = b.match_id
        WHERE m.tournament_id IN (54, 55)
          AND ss.home_value IS NOT NULL AND ss.away_value IS NOT NULL
    """)
    nrow(pair) == 0 && (push!(agree_rows, (bbc=bbc_key, ss=ss_key, n=0, cor=NaN, exact=NaN, mae=NaN)); continue)
    x = vcat(Float64.(pair.bh), Float64.(pair.ba))
    y = vcat(Float64.(pair.sh), Float64.(pair.sa))
    push!(agree_rows, (bbc = bbc_key, ss = ss_key, n = nrow(pair),
                       cor   = round(cor(x, y), digits=3),
                       exact = round(100 * mean(x .== y), digits=1),
                       mae   = round(mean(abs.(x .- y)), digits=2)))
end
QA[:agreement] = DataFrame(agree_rows)
show(QA[:agreement], allrows=true, allcols=true); println()

# ==========================================
# 5. TRAINING OVERLAP — bbc stats ∧ sofascore xG
# ==========================================
println("\n", "="^70, "\n5. WP1 TRAINING SET — matches with bbc stats AND sofascore xG\n", "="^70)

QA[:overlap] = _q("""
    SELECT m.tournament_id, s.name AS season, count(DISTINCT b.match_id) AS n_train
    FROM bbc.match_meta b
    JOIN sofascore.matches m ON m.match_id = b.match_id
    JOIN sofascore.seasons s ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
    JOIN sofascore.match_statistics xg ON xg.match_id = b.match_id
         AND xg.stat_key = 'expectedGoals' AND xg.period = 'ALL'
         AND xg.home_value IS NOT NULL AND xg.away_value IS NOT NULL
    GROUP BY 1, 2 ORDER BY 1, 2
""")
show(QA[:overlap], allrows=true, allcols=true); println()
println("\nTotal training matches: ", sum(QA[:overlap].n_train))

println("\n[INFO] r00 QA complete — record verdicts in NOTES.md. Tables live in `QA`.")
