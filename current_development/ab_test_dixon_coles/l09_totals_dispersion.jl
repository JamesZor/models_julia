# current_development/ab_test_dixon_coles/l09_totals_dispersion.jl
#
# Totals dispersion / calibration diagnostic.   [VALIDATED on server 2026-06-19]
#
# Question: is the model's expected-total-goals forecast COMPRESSED vs the market
# and vs realised goals? And is that compression a bug or the source of the edge?
#
# Three quantities per match:
#   model_Etot   = median_samples(λ_h + λ_a)   (posterior MEDIAN — robust; the
#                  posterior MEAN blows up on ~24/281 matches where a few divergent
#                  samples hit the clamp(log_λ, 20) AD-safety ceiling → λ in the
#                  thousands. Median is immune; gives sane totals 1.8–3.1.)
#   market_Etot  = Σ P_fair(Over k.5) + geometric tail   (de-vigged bookmaker O/U
#                  ladder; Betfair carries NO totals for Ireland, only 1X2/DC, so
#                  the market reference is ds.odds, NOT summarize_betfair_market.)
#   realized_tot = home_score + away_score
#
# Readouts (totals_dispersion_report):
#   disp_ratio  = sd(model_Etot)/sd(market_Etot).  <1 ⇒ model totals compressed.
#   slope_model = OLS slope realized ~ model_Etot.
#   slope_mkt   = OLS slope realized ~ market_Etot.
#       Calibrated forecast ⇒ slope ≈ 1.
#       COMPRESSED / under-dispersed forecast ⇒ slope > 1 (narrow forecast range,
#         realised swings more).   OVER-dispersed forecast ⇒ slope < 1.
#       (Empirically confirmed: mw=0 pure model is over-dispersed → slope 0.20;
#        mw=0.4 is compressed → slope 1.22.)
#   cor_*_real  = correlation of each forecast with realised goals — the "who is
#                 actually right" metric. On Ireland BOTH ≈ 0.11 (totals are
#                 near-irreducibly noisy at the match level).

using DataFrames
using Statistics
using GLM

# ------------------------------------------------------------------
# 1. MODEL expected totals (robust posterior median of λ_h + λ_a)
# ------------------------------------------------------------------
function model_expected_totals(latents)
    df = latents.df
    rows = NamedTuple[]
    for r in eachrow(df)
        tot = r.λ_h .+ r.λ_a
        push!(rows, (match_id = Int(r.match_id),
                     model_Etot   = median(tot),          # robust to divergent samples
                     model_lam_h  = median(r.λ_h),
                     model_lam_a  = median(r.λ_a),
                     model_mean_tot = mean(tot)))         # kept to flag blow-ups
    end
    return DataFrame(rows)
end

# ------------------------------------------------------------------
# 2. MARKET expected totals (de-vigged O/U ladder → E[N])
# ------------------------------------------------------------------
function market_expected_totals(odds_df; over_market::String="OverUnder",
                                prob_col::Symbol=:prob_fair_close)
    ou = filter(:market_name => ==(over_market), odds_df)
    ou = filter(:selection => s -> startswith(String(s), "over_"), ou)
    isempty(ou) && error("No '$over_market' over-selections in odds — check naming. " *
                         "NB Betfair has no totals for Ireland; pass ds.odds (bookmaker).")
    out = NamedTuple[]
    for g in groupby(ou, :match_id)
        order = sortperm(g.market_line)
        P = clamp.(Float64.(g[!, prob_col][order]), 1e-9, 1.0 - 1e-9)
        Etot = sum(P)                                       # Σ P(N>k) over the ladder
        if length(P) >= 2 && P[end] < P[end-1]              # geometric tail beyond top line
            r = clamp(P[end] / P[end-1], 0.0, 0.95)
            Etot += P[end] * r / (1.0 - r)
        end
        push!(out, (match_id = Int(g.match_id[1]), market_Etot = Etot, n_lines = length(P)))
    end
    return DataFrame(out)
end

# ------------------------------------------------------------------
# 3. REALISED totals
# ------------------------------------------------------------------
function realized_totals(ds)
    out = NamedTuple[]
    for r in eachrow(ds.matches)
        (ismissing(r.home_score) || ismissing(r.away_score)) && continue
        push!(out, (match_id = Int(r.match_id),
                    realized_tot = Float64(r.home_score) + Float64(r.away_score)))
    end
    return DataFrame(out)
end

# ------------------------------------------------------------------
# 4. JOIN → per-match calibration frame
# ------------------------------------------------------------------
"""
    build_totals_calibration_df(ds, odds_df, exp; latents=nothing,
                                min_lines=5, etot_range=(0.5, 6.0))

`ds` = full DataStore (features rebuilt for OOS extraction).
`odds_df` = de-vigged bookmaker odds (ds.odds) for the market pillar.
Filters incomplete O/U ladders and junk market totals.
"""
function build_totals_calibration_df(ds, odds_df, exp; latents=nothing,
                                     min_lines::Int=5, etot_range=(0.5, 6.0))
    latents = isnothing(latents) ? Experiments.extract_oos_predictions(ds, exp) : latents
    md = model_expected_totals(latents)
    mk = filter(:n_lines => >=(min_lines), market_expected_totals(odds_df))
    rz = realized_totals(ds)

    df = innerjoin(md, mk, on=:match_id)
    df = innerjoin(df, rz, on=:match_id)
    filter!(:market_Etot => x -> etot_range[1] <= x <= etot_range[2], df)
    df.model_name = fill(String(exp.config.name), nrow(df))
    return df
end

# ------------------------------------------------------------------
# 5. REPORT (one summary row per model)
# ------------------------------------------------------------------
function totals_dispersion_report(df::AbstractDataFrame; model_label::String="")
    βm = coef(lm(@formula(realized_tot ~ model_Etot), df))
    βk = coef(lm(@formula(realized_tot ~ market_Etot), df))
    rnd(x) = round(x, digits=3)
    return (; model = isempty(model_label) ? df.model_name[1] : model_label,
            n = nrow(df),
            mean_model = rnd(mean(df.model_Etot)), mean_mkt = rnd(mean(df.market_Etot)),
            mean_real  = rnd(mean(df.realized_tot)),
            sd_model = rnd(std(df.model_Etot)), sd_mkt = rnd(std(df.market_Etot)),
            sd_real  = rnd(std(df.realized_tot)),
            disp_ratio = rnd(std(df.model_Etot) / std(df.market_Etot)),   # <1 ⇒ compressed
            slope_model = rnd(βm[2]), slope_mkt = rnd(βk[2]),             # >1 ⇒ compressed
            cor_model_real = rnd(cor(df.model_Etot, df.realized_tot)),
            cor_mkt_real   = rnd(cor(df.market_Etot, df.realized_tot)),
            corr_mm = rnd(cor(df.model_Etot, df.market_Etot)))
end

# ------------------------------------------------------------------
# 6. Per-bucket calibration table (binned by where the MARKET sits)
# ------------------------------------------------------------------
function totals_calibration_buckets(df::AbstractDataFrame; nbins::Int=5, by::Symbol=:market_Etot)
    edges = quantile(df[!, by], range(0, 1; length=nbins+1))
    edges[1] -= 1e-9; edges[end] += 1e-9
    tmp = copy(df); tmp.bucket = map(x -> searchsortedlast(edges, x), df[!, by])
    g = combine(groupby(tmp, :bucket),
        :market_Etot  => mean => :mkt_Etot,
        :model_Etot   => mean => :model_Etot,
        :realized_tot => mean => :realized,
        nrow => :n)
    return Base.sort(g, :bucket)
end

# ------------------------------------------------------------------
# 7. Multi-model driver
# ------------------------------------------------------------------
function run_totals_dispersion(ds, odds_df, experiments::AbstractVector)
    reports = NamedTuple[]
    permatch = DataFrame[]
    for (i, exp) in enumerate(experiments)
        name = String(exp.config.name)
        df = build_totals_calibration_df(ds, odds_df, exp)
        push!(permatch, df)
        push!(reports, totals_dispersion_report(df; model_label=name))
    end
    return DataFrame(reports), vcat(permatch...)
end
