#=
r00_position_eda.jl — Phase 1 EDA across ALL betdb leagues for position-aware player ratings.

Runs the four decision gates per league (2023+ starters only) and prints a cross-league summary:
  Gate 1 Coverage          — is the position/rating data even usable?
  Gate 2 Multi-positionality— do players actually appear off their modal position? (MAKE-OR-BREAK)
  Gate 3 Out-of-position Δ  — within-player, is there a real sign-consistent rating penalty? (|t|≫2)
  Gate 4 A vs B             — which construction predicts the held-out next rating better?

Decision: build the MVP only if Gate 2 AND Gate 3 pass on >= 1 league. A negative result is valid.

Run on the server REPL (kaimon): git pull, restart REPL, then
    include("current_development/position_aware_ratings/r00_position_eda.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Printf

include("current_development/position_aware_ratings/l00_position_helpers.jl")

const Data = BayesianFootball.Data

# every segment defined in src/Data/fetchers/segments.jl (skip the legacy shim)
const SEGMENTS = [
    Data.ScottishLower(), Data.Ireland(), Data.IrelandFirstDivision(),
    Data.SouthKorea(), Data.Norway(), Data.Veikkausliiga(),
]

const MIN_APPS = 5      # min real appearances for a player to enter multipos / modal stats
seg_name(s) = string(nameof(typeof(s)))

# collectors for the cross-league summary
summary = DataFrame(league=String[], n_starter=Int[], pct_real_pos=Float64[], pct_defaultM=Float64[],
                    off_modal_share=Float64[], pct_multipos=Float64[],
                    g3_offmodal_coef=Float64[], g3_t=Float64[],
                    g4_rmse_overall=Float64[], g4_rmse_A=Float64[], g4_rmse_B=Float64[])

for seg in SEGMENTS
    name = seg_name(seg)
    println("\n", "#"^84, "\n# LEAGUE: $name\n", "#"^84)

    ds = try
        Data.load_datastore_cached(seg)
    catch e
        println("[SKIP] load_datastore_cached failed: $(sprint(showerror, e))"); continue
    end

  try   # one bad league must not abort the whole sweep
    df = prepare_starter_lineups(ds; from_year=2023)
    if nrow(df) == 0
        println("[SKIP] no 2023+ starter lineups."); continue
    end

    # ---- Gate 1: coverage ----
    cov = coverage_stats(df)
    println("\n[Gate 1 — COVERAGE]")
    @printf("  starter player-matches=%d   players=%d   matches=%d\n", cov.n, cov.n_players, cov.n_matches)
    @printf("  match-date range: %s … %s\n", string(minimum(df.match_date)), string(maximum(df.match_date)))
    @printf("  %% real position=%.1f   %% rated=%.1f   %% defaulted-M=%.1f\n",
            cov.pct_real_pos, cov.pct_rated, cov.pct_defaultM)
    println("  real position mix (%): ", cov.mix_real)

    # ---- Gate 2: multi-positionality ----
    add_modal_position!(df; min_apps=MIN_APPS)
    mp = multipositionality_stats(df; min_apps=MIN_APPS)
    println("\n[Gate 2 — MULTI-POSITIONALITY]  (players with >= $MIN_APPS real apps)")
    if mp.n_players == 0
        println("  no eligible players.");
    else
        @printf("  eligible players=%d   mean distinct pos=%.3f   %% multi-pos players=%.1f   mean entropy=%.3f bits\n",
                mp.n_players, mp.mean_distinct_pos, mp.pct_multipos_players, mp.mean_entropy_bits)
        @printf("  OFF-MODAL share of player-matches=%.2f%%   (n_off_modal=%d)  <-- make-or-break\n",
                mp.off_modal_share, mp.n_off_modal)
    end

    # gates 3 & 4 need realised ratings; many minor leagues have none.
    g3coef = NaN; g3t = NaN; g4o = NaN; g4a = NaN; g4b = NaN
  if !any(df.has_rating)
    println("\n[Gate 3 & 4]  no realised ratings in this league — N/A.")
  else
    # ---- Gate 3: out-of-position Δ ----
    df_os = add_opponent_strength!(df)
    (m3, tbl3) = out_of_position_regression(df_os)
    println("\n[Gate 3 — OUT-OF-POSITION Δ]  within-player FE: rating ~ off_modal + is_home + minutes + opp_str")
    if m3 === nothing
        println("  too few rows / no opponent strength — skipped.")
    else
        show(tbl3; allrows=true, allcols=true); println()
        r = tbl3[tbl3.term .== "off_modal", :]
        if nrow(r) == 1
            g3coef = r.coef[1]; g3t = r.t[1]
            @printf("  => off-modal Δ = %.3f rating pts (t=%.2f)  %s\n", g3coef, g3t,
                    abs(g3t) >= 2 ? "** |t|>=2: real **" : "(|t|<2: weak)")
        end
    end

    # ---- Gate 4: A vs B held-out next-match rating ----
    ab = ab_holdout_eval(df; test_frac=0.3, min_pos_apps=4)
    println("\n[Gate 4 — A vs B]  chronological holdout, RMSE of pre-match estimate vs realised rating")
    if hasproperty(ab, :note)
        println("  ", ab.note)
    else
        println("  test cut=$(ab.cut)  n_test=$(ab.n_test)")
        ao = ab.all_rows; om = ab.off_modal
        @printf("  ALL test rows (n=%d):       overall=%.4f  A=%.4f  B=%.4f\n", ao.n, ao.rmse_overall, ao.rmse_A, ao.rmse_B)
        @printf("  OFF-MODAL test rows (n=%d): overall=%.4f  A=%.4f  B=%.4f   <-- where they differ\n",
                om.n, om.rmse_overall, om.rmse_A, om.rmse_B)
        @printf("  %% test rows where A differs >0.25 from overall=%.2f   B differs=%.2f\n",
                ab.pct_A_differs, ab.pct_B_differs)
        g4o = om.rmse_overall; g4a = om.rmse_A; g4b = om.rmse_B
    end
  end  # rating-gated block

    push!(summary, (name, cov.n, cov.pct_real_pos, cov.pct_defaultM,
                    mp.n_players == 0 ? NaN : mp.off_modal_share,
                    mp.n_players == 0 ? NaN : mp.pct_multipos_players,
                    g3coef, g3t, g4o, g4a, g4b))
  catch e
    println("[ERROR] $name gates failed: $(sprint(showerror, e))")
  end  # per-league try
end

# ============================================================================
# CROSS-LEAGUE SUMMARY + decision gate
# ============================================================================
println("\n", "="^84, "\nCROSS-LEAGUE SUMMARY\n", "="^84)
show(summary; allrows=true, allcols=true, truncate=0); println()

println("""

$("="^84)
DECISION GATE (build MVP only if BOTH pass on >= 1 league):
  Gate 2 pass: off_modal_share materially > 0 (players really do move position).
  Gate 3 pass: |g3_t| >= 2 with a consistent (negative) off-modal coef.
If Gate 4 also shows A or B beating `overall` RMSE on OFF-MODAL rows, prefer that construction
for the l01 feature. If Gate 2 ~ 0 everywhere -> STOP: per-position ratings ≡ the single rating.
Record the verdict in NOTES.md (timestamped, newest-first).
$("="^84)
""")
