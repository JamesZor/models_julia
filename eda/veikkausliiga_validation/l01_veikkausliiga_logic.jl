# eda/veikkausliiga_validation/l01_veikkausliiga_logic.jl
#
# Loader (math / functions only — NO top-level execution).
#
# Stage-A EDA for the NEW competition Veikkausliiga (Finnish top flight,
# tournament 31). Goal: characterise 31 as a data-generating process so the L1
# Bayesian engines know which goals likelihood family to use (Poisson / NB /
# COM / Weibull), PLUS three extras the 718 study did not cover:
#   (a) per-team ATTACK/DEFENCE goal & xG distributions vs the league average,
#       ranked with a formal test + empirical-Bayes shrinkage;
#   (b) a player-rating COVERAGE audit (is the rating feature usable?);
#   (c) per-team player-rating distributions vs the league average (squad quality).
#
# Pairs with r01_veikkausliiga_runner.jl. Standalone (no contrast league).
#
# REUSE: the count fitters / summaries / GoF / DC ladder / league diagnostics
# already exist and are include-safe. Pull them in transitively via the
# first_division loader (which itself includes the two ireland_validation files):
#   - feature_coverage_by_season, datastore_overview, get_goals,
#     analyze_goal_models, analyze_heavyweight_models, fit_dc_ladder
#   - summarise_count, compare_count_models, compare_nb1_nb2, rootogram_data,
#     chi_square_gof, fit_poisson_entry, fit_negbin_entry, fit_mle(...)
#   - test_overdispersion, test_home_advantage_mean/_variance,
#     test_team_volatility, test_temporal_stability.

using DataFrames
using Distributions
using Statistics
using StatsBase
using HypothesisTests
using Printf
using BayesianFootball

include("../first_division_validation/l01_first_division_logic.jl")

# ============================================================================
# 0. SMALL HELPERS
# ============================================================================

"""
    clean_pos(pos) -> "G"|"D"|"M"|"F"

Normalise raw SofaScore/DB position labels to the four buckets. Unknown → "M".
(Copied from current_development/match_day_inference/src/ratings.jl to keep this
EDA loader free of the inference module's dependencies.)
"""
function clean_pos(pos::AbstractString)
    if pos == "G" || pos == "Goalkeeper" || pos == "GK"
        return "G"
    elseif pos == "D" || pos == "Defender" || pos == "DF"
        return "D"
    elseif pos == "M" || pos == "Midfielder" || pos == "MF"
        return "M"
    elseif pos == "F" || pos == "Forward" || pos == "FW" || pos == "A"
        return "F"
    else
        return "M"
    end
end

"""
    bh_adjust(pvals) -> Vector{Float64}

Benjamini–Hochberg FDR-adjusted p-values (monotone, clamped to ≤ 1). With ~12
teams a raw per-team p-value is one of many simultaneous comparisons against the
league average, so we always report the adjusted column alongside the raw one.
"""
function bh_adjust(pvals::AbstractVector{<:Real})
    m = length(pvals)
    m == 0 && return Float64[]
    ord = sortperm(pvals)
    adj = Vector{Float64}(undef, m)
    prev = 1.0
    for i in m:-1:1
        idx = ord[i]
        val = min(prev, pvals[idx] * m / i)
        adj[idx] = val
        prev = val
    end
    return clamp.(adj, 0.0, 1.0)
end

# ============================================================================
# 1. PER-TEAM LONG TABLE (goals + xG, for/against)
# ============================================================================

"""
    build_team_match_long(ds) -> DataFrame

One row per (played match, side) with columns
`match_id, season, match_date, team, is_home, goals_for, goals_against,
xg_for, xg_against`. xG is `NaN` where the `ALL`-period stats row is missing it
(pre-2023 for Veikkausliiga). This is the substrate for the per-team
attack/defence fits.
"""
function build_team_match_long(ds::Data.DataStore)
    stats = nrow(ds.statistics) == 0 ? ds.statistics :
            filter(r -> r.period == "ALL", ds.statistics)
    scols = Set(propertynames(stats))
    has_xg = (:expectedGoals_home in scols) && (:expectedGoals_away in scols)
    smap = Dict(r.match_id => r for r in eachrow(stats))

    rows = NamedTuple[]
    for mr in eachrow(ds.matches)
        ismissing(mr.home_score) && continue
        s = get(smap, mr.match_id, nothing)
        xgh = (has_xg && s !== nothing && !ismissing(s.expectedGoals_home)) ? Float64(s.expectedGoals_home) : NaN
        xga = (has_xg && s !== nothing && !ismissing(s.expectedGoals_away)) ? Float64(s.expectedGoals_away) : NaN
        season = string(mr.season)
        push!(rows, (match_id = mr.match_id, season = season, match_date = mr.match_date,
                     team = mr.home_team, is_home = true,
                     goals_for = Int(mr.home_score), goals_against = Int(mr.away_score),
                     xg_for = xgh, xg_against = xga))
        push!(rows, (match_id = mr.match_id, season = season, match_date = mr.match_date,
                     team = mr.away_team, is_home = false,
                     goals_for = Int(mr.away_score), goals_against = Int(mr.home_score),
                     xg_for = xga, xg_against = xgh))
    end
    return DataFrame(rows)
end

# ============================================================================
# 2. PER-TEAM COUNT FIT vs LEAGUE (Gamma–Poisson empirical Bayes + rate test)
# ============================================================================

"""
    _team_count_table(long_df, valcol; min_matches, higher_is_better)

Generic per-team count analysis of `valcol` (an integer goals column) against the
pooled league rate. For each team with ≥ `min_matches` rows:

  - raw mean / var / index-of-dispersion / n / total k,
  - **rate ratio** RR = team_mean / league_mean,
  - **quasi-Poisson log-rate z-test** vs the league rate: se = √(φ/k) where φ is
    the pooled league dispersion index (so over-dispersion does not manufacture
    false significance); 95% CI from the same Wald se; BH-adjusted p,
  - **Gamma–Poisson empirical-Bayes shrunk rate**: a league prior
    Gamma(α₀, β₀) is fit by moment-matching the team rates (β₀ = μ/τ²,
    α₀ = μ·β₀, τ² = between-team variance net of sampling), giving the posterior
    mean (α₀ + k)/(β₀ + n) — this is what keeps a hot 15-game start from topping
    the table over a steady full-season side.

Sorted by the shrunk rate (descending if `higher_is_better`, else ascending).
"""
function _team_count_table(long_df::DataFrame, valcol::Symbol; min_matches::Int = 15,
                           higher_is_better::Bool = true)
    vals = Float64.(long_df[!, valcol])
    μ = mean(vals)                      # league rate
    φ = max(var(vals) / μ, 1e-6)        # league dispersion index (quasi-Poisson scale)

    g = combine(groupby(long_df, :team),
                valcol => mean => :mean,
                valcol => var  => :var,
                valcol => sum  => :k,
                nrow           => :n)
    filter!(r -> r.n >= min_matches, g)

    g.di = g.var ./ g.mean
    g.rate_ratio = g.mean ./ μ

    # quasi-Poisson Wald z on the log rate, with CI
    z = similar(g.mean); pval = similar(g.mean); lo = similar(g.mean); hi = similar(g.mean)
    for i in 1:nrow(g)
        k = max(g.k[i], 1e-6)
        se = sqrt(φ / k)               # var(log λ̂) ≈ φ/k under quasi-Poisson
        lr = log(max(g.mean[i], 1e-6) / μ)
        z[i] = lr / se
        pval[i] = 2 * ccdf(Normal(), abs(z[i]))
        lo[i] = μ * exp(lr - 1.96se)
        hi[i] = μ * exp(lr + 1.96se)
    end
    g.z = z; g.p = pval; g.p_adj = bh_adjust(pval); g.ci_lo = lo; g.ci_hi = hi

    # Gamma–Poisson empirical-Bayes shrinkage
    τ2 = max(var(g.mean) - mean(g.mean ./ g.n), 1e-6)
    β0 = μ / τ2
    α0 = μ * β0
    g.shrunk_rate = (α0 .+ g.k) ./ (β0 .+ g.n)

    sort!(g, :shrunk_rate, rev = higher_is_better)
    return g, (league_rate = μ, phi = φ, alpha0 = α0, beta0 = β0, tau2 = τ2)
end

"Pretty-print a per-team count table with the league rate / EB prior banner."
function _print_team_count(g::DataFrame, meta, title::String)
    println("\n" * "═"^96)
    @printf(" %s  (league rate μ=%.3f | dispersion φ=%.3f | EB prior Gamma(α₀=%.2f, β₀=%.2f))\n",
            uppercase(title), meta.league_rate, meta.phi, meta.alpha0, meta.beta0)
    println("═"^96)
    @printf("%-22s | %4s | %6s | %5s | %6s | %6s | %7s | %7s | %s\n",
            "team", "n", "mean", "DI", "RR", "shrunk", "z", "p_adj", "95% CI")
    println("-"^96)
    for r in eachrow(g)
        sig = r.p_adj < 0.05 ? "*" : " "
        @printf("%-22s | %4d | %6.3f | %5.2f | %6.2f | %6.3f | %+7.2f | %6.3f%s | [%.2f, %.2f]\n",
                first(string(r.team), 22), r.n, r.mean, r.di, r.rate_ratio,
                r.shrunk_rate, r.z, r.p_adj, sig, r.ci_lo, r.ci_hi)
    end
    nsig = count(<(0.05), g.p_adj)
    println("-"^96)
    @printf("Teams: %d | significantly ≠ league (BH p_adj<0.05): %d\n", nrow(g), nsig)
    return nothing
end

"""
    fit_team_attack_defence(long_df; min_matches=15) -> (attack, defence)

Per-team **goals** attack (`goals_for`) and defence (`goals_against`) tables vs
the league average. Attack sorted best-first (high scoring), defence sorted
best-first (low conceding). Each row carries the rate ratio, quasi-Poisson test,
and Gamma–Poisson shrunk rate (see `_team_count_table`). Prints both; returns the
two DataFrames.
"""
function fit_team_attack_defence(long_df::DataFrame; min_matches::Int = 15)
    atk, atk_meta = _team_count_table(long_df, :goals_for;     min_matches = min_matches, higher_is_better = true)
    def, def_meta = _team_count_table(long_df, :goals_against; min_matches = min_matches, higher_is_better = false)
    _print_team_count(atk, atk_meta, "PER-TEAM ATTACK — goals scored / match (best first)")
    _print_team_count(def, def_meta, "PER-TEAM DEFENCE — goals conceded / match (best=fewest first)")
    return (attack = atk, defence = def)
end

# ============================================================================
# 3. PER-TEAM NORMAL FIT vs LEAGUE (Welch test + Normal–Normal shrinkage)
#    Used for the continuous metrics: xG (for/against) and player ratings.
# ============================================================================

"""
    _team_normal_table(df, valcol; min_matches, higher_is_better, group=:team)

Generic per-team analysis of a continuous `valcol` against the league mean. For
each team with ≥ `min_matches` non-missing rows:

  - mean / sd / n,
  - **Welch two-sample t-test** of the team's values vs all OTHER teams' values
    (unequal variance), with BH-adjusted p,
  - **Normal–Normal hierarchical shrinkage**: reliability
    λ = τ²/(τ² + σ²/n), shrunk mean = μ + λ·(team_mean − μ), where τ² is the
    between-team variance net of sampling noise.

Sorted by the shrunk mean (descending if `higher_is_better`).
"""
function _team_normal_table(df::DataFrame, valcol::Symbol; min_matches::Int = 10,
                            higher_is_better::Bool = true, group::Symbol = :team)
    sub = filter(r -> !isnan(Float64(r[valcol])), df)
    μ = mean(Float64.(sub[!, valcol]))
    σ2_pool = var(Float64.(sub[!, valcol]))

    g = combine(groupby(sub, group),
                valcol => (x -> mean(Float64.(x))) => :mean,
                valcol => (x -> (length(x) > 1 ? std(Float64.(x)) : NaN)) => :sd,
                nrow => :n)
    filter!(r -> r.n >= min_matches, g)

    # Welch team-vs-rest
    pval = fill(NaN, nrow(g))
    for i in 1:nrow(g)
        tname = g[i, group]
        x = Float64.(sub[sub[!, group] .== tname, valcol])
        y = Float64.(sub[sub[!, group] .!= tname, valcol])
        if length(x) > 1 && length(y) > 1 && std(x) > 0 && std(y) > 0
            pval[i] = pvalue(UnequalVarianceTTest(x, y))
        end
    end
    g.p = pval; g.p_adj = bh_adjust(coalesce.(pval, 1.0))

    # Normal–Normal shrinkage
    τ2 = max(var(g.mean) - mean((g.sd .^ 2) ./ g.n), 1e-6)
    g.lambda = τ2 ./ (τ2 .+ (g.sd .^ 2) ./ g.n)
    g.shrunk = μ .+ g.lambda .* (g.mean .- μ)

    sort!(g, :shrunk, rev = higher_is_better)
    return g, (league_mean = μ, pooled_sd = sqrt(σ2_pool), tau = sqrt(τ2))
end

"Pretty-print a per-team continuous table."
function _print_team_normal(g::DataFrame, meta, title::String; group::Symbol = :team)
    println("\n" * "═"^88)
    @printf(" %s  (league mean μ=%.3f | pooled sd=%.3f | between-team τ=%.3f)\n",
            uppercase(title), meta.league_mean, meta.pooled_sd, meta.tau)
    println("═"^88)
    @printf("%-22s | %4s | %7s | %6s | %7s | %6s | %7s\n",
            string(group), "n", "mean", "sd", "shrunk", "λ", "p_adj")
    println("-"^88)
    for r in eachrow(g)
        sig = (!isnan(r.p_adj) && r.p_adj < 0.05) ? "*" : " "
        @printf("%-22s | %4d | %7.3f | %6.3f | %7.3f | %6.2f | %6.3f%s\n",
                first(string(r[group]), 22), r.n, r.mean, r.sd, r.shrunk, r.lambda, r.p_adj, sig)
    end
    nsig = count(p -> !isnan(p) && p < 0.05, g.p_adj)
    println("-"^88)
    @printf("Teams: %d | significantly ≠ league (BH p_adj<0.05): %d\n", nrow(g), nsig)
    return nothing
end

"""
    fit_team_xg_attack_defence(long_df; min_matches=15) -> (attack, defence)

Per-team **xG** attack (`xg_for`) and defence (`xg_against`) tables vs the league
average (continuous → Welch + Normal–Normal shrinkage). Rows with missing xG
(pre-2023) are dropped automatically. Returns the two DataFrames.
"""
function fit_team_xg_attack_defence(long_df::DataFrame; min_matches::Int = 15)
    atk, atk_meta = _team_normal_table(long_df, :xg_for;     min_matches = min_matches, higher_is_better = true)
    def, def_meta = _team_normal_table(long_df, :xg_against; min_matches = min_matches, higher_is_better = false)
    _print_team_normal(atk, atk_meta, "PER-TEAM xG ATTACK — xG for / match (best first)")
    _print_team_normal(def, def_meta, "PER-TEAM xG DEFENCE — xG against / match (best=lowest first)")
    return (attack = atk, defence = def)
end

# ============================================================================
# 4. PLAYER-RATING COVERAGE AUDIT
# ============================================================================

"""
    rating_coverage_audit(ds) -> DataFrame

Per-season usability audit of the SofaScore player `rating` field in `ds.lineups`:

  - played matches, lineup_matches (matches with any lineup row),
  - matches_any_rating + its fraction of played matches,
  - mean rated starters per team per match (starters = `is_substitute==false`,
    `rating` non-missing) — the XI-coverage number that decides whether the
    positional rating pillar is feasible.

The decision rule: ratings are model-usable for a season only if frac_any_rating
≈ 1 and mean_rated_starters_per_team is close to 11.
"""
function rating_coverage_audit(ds::Data.DataStore)
    lu = ds.lineups
    season_of = Dict(r.match_id => string(r.season) for r in eachrow(ds.matches))
    played = filter(r -> !ismissing(r.home_score), ds.matches)
    seasons = sort(unique(string.(ds.matches.season)))

    has_rating(r) = !ismissing(r.rating) && !isnan(Float64(r.rating))
    is_starter(r) = (:is_substitute in propertynames(lu)) ? (!ismissing(r.is_substitute) && !r.is_substitute) : true

    # match_id → set of sides with ≥1 rating ; (match_id,side) → rated-starter count
    matches_with_rating = Set{Int}()
    starter_counts = Dict{Tuple{Int,String},Int}()
    for r in eachrow(lu)
        has_rating(r) || continue
        push!(matches_with_rating, r.match_id)
        if is_starter(r)
            key = (r.match_id, String(r.team_side))
            starter_counts[key] = get(starter_counts, key, 0) + 1
        end
    end
    lineup_match_ids = Set(lu.match_id)

    rows = NamedTuple[]
    for s in seasons
        ps = filter(r -> string(r.season) == s, played)
        ids = Set(ps.match_id)
        nplayed = nrow(ps)
        n_lineup = count(in(lineup_match_ids), ids)
        n_rated = count(in(matches_with_rating), ids)
        # rated starters per team per match, averaged over (match,side) units in season
        units = [v for ((mid, _side), v) in starter_counts if mid in ids]
        push!(rows, (
            season = s,
            played = nplayed,
            lineup_matches = n_lineup,
            matches_any_rating = n_rated,
            frac_any_rating = nplayed == 0 ? 0.0 : round(n_rated / nplayed; digits = 3),
            mean_rated_starters_per_team = isempty(units) ? 0.0 : round(mean(units); digits = 2),
        ))
    end
    return DataFrame(rows)
end

"""
    rating_position_coverage(ds) -> DataFrame

Rated-starter coverage broken down by cleaned position (G/D/M/F), pooled over
seasons that carry ratings. Confirms ratings are not systematically missing for a
position bucket (e.g. goalkeepers), which would bias a positional rating pillar.
"""
function rating_position_coverage(ds::Data.DataStore)
    lu = ds.lineups
    is_starter(r) = (:is_substitute in propertynames(lu)) ? (!ismissing(r.is_substitute) && !r.is_substitute) : true
    starters = filter(r -> is_starter(r) && !ismissing(r.position), lu)
    starters.cpos = [clean_pos(String(p)) for p in starters.position]
    starters.has_rating = [!ismissing(x) && !isnan(Float64(x)) for x in starters.rating]
    g = combine(groupby(starters, :cpos),
                :has_rating => sum => :rated,
                nrow => :starters)
    g.frac_rated = round.(g.rated ./ g.starters; digits = 3)
    sort!(g, :cpos)
    return g
end

# ============================================================================
# 5. PER-TEAM PLAYER-RATING DISTRIBUTIONS
# ============================================================================

"""
    build_team_rating_long(ds) -> DataFrame

One row per (match, side) with a single **minute-weighted team match rating** (the
weighted mean of rated players' SofaScore ratings, weight = minutes played, ≥1).
Drops sides with no rated players (i.e. pre-2023 matches). Columns:
`match_id, season, team, is_home, team_rating, n_rated`.
"""
function build_team_rating_long(ds::Data.DataStore)
    lu = ds.lineups
    mmap = Dict(r.match_id => r for r in eachrow(ds.matches))
    has_min = :minutes_played in propertynames(lu)

    rows = NamedTuple[]
    for sub in groupby(lu, [:match_id, :team_side])
        mid = sub.match_id[1]; side = String(sub.team_side[1])
        haskey(mmap, mid) || continue
        mr = mmap[mid]
        ismissing(mr.home_score) && continue
        rated = filter(r -> !ismissing(r.rating) && !isnan(Float64(r.rating)), sub)
        nrow(rated) == 0 && continue
        ratings = Float64.(rated.rating)
        w = has_min ? [ismissing(m) ? 1.0 : max(Float64(m), 1.0) for m in rated.minutes_played] :
                      ones(length(ratings))
        team_rating = sum(w .* ratings) / sum(w)
        team = side == "home" ? mr.home_team : mr.away_team
        push!(rows, (match_id = mid, season = string(mr.season), team = team,
                     is_home = (side == "home"), team_rating = team_rating, n_rated = nrow(rated)))
    end
    return DataFrame(rows)
end

"""
    fit_team_rating_dist(rating_long; min_matches=10) -> DataFrame

Per-team distribution of the minute-weighted team match rating vs the league
average (Welch team-vs-rest + Normal–Normal shrinkage), ranked best squad first.
Returns the table.
"""
function fit_team_rating_dist(rating_long::DataFrame; min_matches::Int = 10)
    g, meta = _team_normal_table(rating_long, :team_rating; min_matches = min_matches, higher_is_better = true)
    _print_team_normal(g, meta, "PER-TEAM SQUAD QUALITY — minute-weighted team rating (best first)")
    return g
end
