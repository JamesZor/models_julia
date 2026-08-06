# src/Portfolio/stake.jl
#
# Stage B: a built book plus a policy becomes sized stakes.
#
#     a_kelly  ->  x trust  ->  x shrink  ->  x risk  ->  cap  ->  filter
#
# Order matters, and one property of it is worth stating explicitly because it is
# counter-intuitive and drives most of the system's behaviour:
#
#   `risk_factor` is HOMOGENEOUS OF DEGREE 0 in the stakes it is handed. It solves for the
#   factor that makes those stakes satisfy the drawdown constraint, so handing it twice the
#   stakes returns half the factor and `k .* stakes` is unchanged. Once the constraint binds,
#   trust and shrinkage can therefore only RESHAPE the book -- they cannot resize it.
#
# Measured: at lambda = 20 a stake multiplier of 0.25, 1.0 or 4.0 all produce mean slate
# exposure 0.1088. Trust settings of 0.25, 0.5 and 1.0 produce identical final wealth. To move
# exposure, move lambda (`calibrate_lambda`), not trust.

export stake_slate

"""
    stake_slate(policy, slate, ctx; use_shrink = true, scale = 1.0) -> SlateAllocation

`scale` is a global multiplier applied before the risk step; it exists for `calibrate_scale`
and is very nearly a no-op whenever the drawdown constraint is active, per the note above.
"""
function stake_slate(policy::PolicySpec, slate::Slate, ctx::SlateContext;
                     use_shrink::Bool = true, scale::Float64 = 1.0)
    L = length(slate.books)
    stakes = Vector{Vector{Float64}}(undef, L)

    # --- trust + parameter-uncertainty shrinkage (both per match) ---
    @inbounds for i in 1:L
        b = slate.books[i]
        a = copy(b.a_kelly)
        w = trust_vector(policy.trust, b, ctx)
        for j in eachindex(a)
            a[j] *= w[j]
        end
        use_shrink && (a .*= b.k_shrink)
        a .*= scale
        stakes[i] = a
    end

    # --- drawdown budget ---
    probs = [b.p_grid for b in slate.books]
    rets  = [slate.books[i].R * stakes[i] for i in 1:L]
    kf    = risk_factor(policy.risk, probs, rets)

    k_report = if kf isa AbstractVector          # isolated: one factor per match
        @inbounds for i in 1:L
            stakes[i] .*= kf[i]
        end
        isempty(kf) ? 1.0 : mean(kf)
    else                                          # slate-wide: one factor for everything
        @inbounds for s in stakes
            s .*= kf
        end
        kf
    end

    # --- hard exposure cap (never optional) ---
    stakes, capped = apply_cap(policy.cap, stakes)

    # --- curation, last: can only remove exposure ---
    if !(policy.filter isa KeepAll)
        @inbounds for i in 1:L
            b = slate.books[i]
            for j in eachindex(stakes[i])
                stakes[i][j] > 0 && !keep(policy.filter, b.sels[j], stakes[i][j], ctx) &&
                    (stakes[i][j] = 0.0)
            end
        end
    end

    exposure = isempty(stakes) ? 0.0 : sum(sum(s) for s in stakes)
    return SlateAllocation(stakes, k_report, exposure, capped)
end
