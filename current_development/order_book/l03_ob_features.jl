# current_development/order_book/l03_ob_features.jl
# Derived features for order book entry-timing analysis

using DataFrames
using Statistics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_safe_sum3(a, b, c) = (isnan(a) ? 0.0 : a) + (isnan(b) ? 0.0 : b) + (isnan(c) ? 0.0 : c)

function _safe_cor(x::Vector{Float64}, y::Vector{Float64})::Float64
    mask = .!isnan.(x) .& .!isnan.(y)
    sum(mask) < 5 && return NaN
    return cor(x[mask], y[mask])
end

# ---------------------------------------------------------------------------
# add_ob_features: augments ob DataFrame with all derived columns
# ---------------------------------------------------------------------------

"""
    add_ob_features(ob) -> DataFrame

Adds the following columns to a copy of ob:
  Scalar:  mid_price, spread_abs, spread_pct, bid_depth, ask_depth,
           total_depth, OBI, depth_ratio, VWMP, time_bucket
  Rolling: price_vel_back (5-snap backward), price_vel_fwd (5-snap forward),
           price_vol_10m (rolling std), OBI_stability (5-snap std of OBI),
           is_price_jump (|back vel| > 2% of mid)

Sorted by [:market_id, :selection, :ts] before rolling is computed.
"""
function add_ob_features(ob::DataFrame)::DataFrame
    df = copy(ob)
    sort!(df, [:market_id, :selection, :ts])

    # --- Point-in-time scalars ---
    df.mid_price   = (df.bid_price_1 .+ df.ask_price_1) ./ 2.0
    df.spread_abs  = df.ask_price_1 .- df.bid_price_1
    df.spread_pct  = df.spread_abs ./ df.mid_price .* 100.0

    df.bid_depth   = _safe_sum3.(df.bid_vol_1, df.bid_vol_2, df.bid_vol_3)
    df.ask_depth   = _safe_sum3.(df.ask_vol_1, df.ask_vol_2, df.ask_vol_3)
    df.total_depth = df.bid_depth .+ df.ask_depth
    df.OBI         = (df.bid_depth .- df.ask_depth) ./ df.total_depth
    df.depth_ratio = df.bid_depth ./ df.ask_depth

    # Volume-weighted mid — skews toward the heavier side
    denom          = df.bid_vol_1 .+ df.ask_vol_1
    df.VWMP        = (df.bid_price_1 .* df.bid_vol_1 .+ df.ask_price_1 .* df.ask_vol_1) ./ denom

    # 30-minute time bucket for aggregation
    df.time_bucket = floor.(df.minutes_to_kickoff ./ 30.0) .* 30.0

    # --- Rolling features (per market_id × selection, index-based) ---
    n = nrow(df)
    price_vel_back = fill(NaN, n)
    price_vel_fwd  = fill(NaN, n)
    price_vol_10m  = fill(NaN, n)
    obi_stab_5m    = fill(NaN, n)

    for grp in groupby(df, [:market_id, :selection])
        idxs = parentindices(grp)[1]
        m    = length(idxs)
        m < 2 && continue

        mid = Float64.(grp.mid_price)
        obi = Float64.(grp.OBI)

        # Backward velocity: how much did price move in last 5 snapshots?
        for i in 6:m
            price_vel_back[idxs[i]] = mid[i] - mid[i-5]
        end

        # Forward velocity: how much will price move in next 5 snapshots?
        # Used to test whether OBI predicts direction.
        for i in 1:(m-5)
            price_vel_fwd[idxs[i]] = mid[i+5] - mid[i]
        end

        # Local volatility: rolling 10-snapshot std of mid
        for i in 10:m
            price_vol_10m[idxs[i]] = std(mid[i-9:i])
        end

        # Book stability: rolling 5-snapshot std of OBI
        for i in 5:m
            obi_stab_5m[idxs[i]] = std(obi[i-4:i])
        end
    end

    df.price_vel_back = price_vel_back
    df.price_vel_fwd  = price_vel_fwd
    df.price_vol_10m  = price_vol_10m
    df.OBI_stability  = obi_stab_5m

    # Price jump flag: backward move exceeds 2% of mid (detects lineup/news events)
    df.is_price_jump = .!isnan.(df.price_vel_back) .&
                       (abs.(df.price_vel_back) ./ df.mid_price .> 0.02)

    return df
end

# ---------------------------------------------------------------------------
# add_entry_criteria!: adds entry_ok boolean column
# ---------------------------------------------------------------------------

"""
    add_entry_criteria!(df; spread_thresh, depth_thresh, obi_stab_thresh, obi_max)

Marks each row as a valid entry opportunity when ALL of:
  spread_pct  < spread_thresh   (default 2.0%)
  total_depth > depth_thresh    (default £500 — adjust to your typical stake)
  OBI_stability < obi_stab_thresh (default 0.15 — book not churning)
  |OBI|       < obi_max          (default 0.4  — not a one-sided panic)
"""
function add_entry_criteria!(
    df::DataFrame;
    spread_thresh    = 2.0,
    depth_thresh     = 500.0,
    obi_stab_thresh  = 0.15,
    obi_max          = 0.4,
)
    df.entry_ok = (
        df.spread_pct  .< spread_thresh                          .&&
        df.total_depth .> depth_thresh                           .&&
        coalesce.(df.OBI_stability, Inf) .< obi_stab_thresh      .&&
        abs.(df.OBI)   .< obi_max
    )
    return df
end

# ---------------------------------------------------------------------------
# ou25_aggregate_stats: per-bucket summary for OverUnder 2.5
# ---------------------------------------------------------------------------

"""
    ou25_aggregate_stats(obf; lookback=360) -> DataFrame

Returns a per-(time_bucket, selection) summary table for OU2.5 markets.
Columns: n, med/q25/q75 spread_pct, med/q25/q75 depth, med OBI,
         med OBI_stability, jump_freq%, OBI_fwd_cor, entry_pct%.
"""
function ou25_aggregate_stats(obf::DataFrame; time_window::Tuple{Real, Real} = (-360.0, 0.0))::DataFrame
    ou = filter(r ->
        r.market_name      == "OverUnder" &&
        r.market_line      == 2.5         &&
        time_window[1] .<= r.minutes_to_kickoff .<= time_window[2],
        obf)

    isempty(ou) && error("No OU2.5 data in the provided DataFrame")

    nanv(v)       = filter(!isnan, Float64.(v))
    nanmed(v)     = (w = nanv(v); isempty(w) ? NaN : median(w))
    nanq(v, q)    = (w = nanv(v); isempty(w) ? NaN : quantile(w, q))
    nanmean_bool(v) = mean(Bool.(v))

    stats = combine(
        groupby(sort(ou, :time_bucket), [:time_bucket, :selection]),
        nrow                                            => :n,
        :spread_pct   => nanmed                        => :med_spread_pct,
        :spread_pct   => (v -> nanq(v, 0.25))          => :q25_spread_pct,
        :spread_pct   => (v -> nanq(v, 0.75))          => :q75_spread_pct,
        :total_depth  => nanmed                        => :med_depth,
        :total_depth  => (v -> nanq(v, 0.25))          => :q25_depth,
        :total_depth  => (v -> nanq(v, 0.75))          => :q75_depth,
        :OBI          => nanmed                        => :med_OBI,
        :OBI_stability => nanmed                       => :med_OBI_stab,
        :is_price_jump => nanmean_bool                 => :jump_freq,
        AsTable([:OBI, :price_vel_fwd]) =>
            (nt -> _safe_cor(Float64.(nt.OBI), Float64.(nt.price_vel_fwd))) => :OBI_fwd_cor,
        :entry_ok     => nanmean_bool                  => :entry_pct,
    )

    sort!(stats, [:selection, :time_bucket])
    return stats
end
