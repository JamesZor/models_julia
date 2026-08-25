# Side-effect-free phase-1 diagnostics for audit issue 02.  This loader requires
# ExistingChainTeamBridge to have been included first; identity mapping is intentionally not repeated.
module TauExtractionDiagnostics
using Statistics, MCMCChains
import Main.ExistingChainTeamBridge

export chain_matrix, chain_vector, tau_effects, tau_summary, team_effect_summary,
       assert_tau_reconstruction!, reconstruct_pxg_tau_only, affected_method_manifest

const affected_method_manifest = [
    (file="l03_recombination_models.jl", method="NegBin chain/FeatureSet extractor", lines="1100-1169", tau=:required),
    (file="l03_recombination_models.jl", method="NegBin DataFrame OOS extractor", lines="1171-1274", tau=:required),
    (file="l04_recomb_wealth_models.jl", method="Wealth DataFrame OOS extractor", lines="403-501", tau=:required),
    (file="l04_recomb_wealth_models.jl", method="Wealth FeatureSet/chain extractor", lines="504-608", tau=:required),
    (file="l05_recomb_pxg_models.jl", method="pxG DataFrame OOS extractor", lines="340-449", tau=:required),
    (file="l05_recomb_pxg_models.jl", method="pxG FeatureSet/chain extractor", lines="451-564", tau=:required),
    (file="l03_recombination_models.jl", method="Poisson control/open-play/integrated extractors", lines="766-1098", tau=:not_applicable),
]

"""Stack MCMCChains iterations × parameters × chains into draw × parameter.
Also accepts the 1-column vector/2-D forms emitted by MCMCChains versions."""
function chain_matrix(chain::Chains, labels::Vector{String})
    a = Array(chain[labels])
    if ndims(a) == 1
        length(labels) == 1 || error("1-D chain array cannot represent $(length(labels)) columns")
        return reshape(a, :, 1)
    elseif ndims(a) == 2
        size(a, 2) == length(labels) || error("unexpected 2-D shape $(size(a))")
        return Matrix(a)
    elseif ndims(a) == 3
        size(a, 2) == length(labels) || error("unexpected 3-D shape $(size(a))")
        return reshape(permutedims(a, (1, 3, 2)), :, length(labels))
    end
    error("unexpected chain dimensions $(size(a))")
end
chain_vector(chain::Chains, label::String) = vec(Array(chain[label]))
_has(chain::Chains, label::String) = label in Set(string.(names(chain)))

function tau_effects(chain::Chains, n_teams::Integer)
    raw_alpha = chain_matrix(chain, ["raw_alpha[$i]" for i in 1:n_teams])
    raw_beta = chain_matrix(chain, ["raw_beta[$i]" for i in 1:n_teams])
    tau_present = _has(chain, "tau_alpha") && _has(chain, "tau_beta")
    tau_present || error("tau_alpha/tau_beta absent: this model is tau-free by design or the artifact is incompatible")
    tau_alpha, tau_beta = chain_vector(chain, "tau_alpha"), chain_vector(chain, "tau_beta")
    draws = size(raw_alpha, 1)
    length(tau_alpha) == draws == length(tau_beta) || error("draw-count mismatch")
    centered_alpha, centered_beta = raw_alpha .- mean(raw_alpha, dims=2), raw_beta .- mean(raw_beta, dims=2)
    exact_alpha = centered_alpha .* reshape(tau_alpha, :, 1)
    exact_beta = centered_beta .* reshape(tau_beta, :, 1)
    return (; raw_alpha, raw_beta, centered_alpha, centered_beta, exact_alpha, exact_beta, tau_alpha, tau_beta)
end

"Assert zero-sum per draw, inputs untouched, and exact scale equation; this never mutates a chain array."
function assert_tau_reconstruction!(effects; atol=1e-11)
    @assert all(abs.(vec(sum(effects.centered_alpha, dims=2))) .<= atol)
    @assert all(abs.(vec(sum(effects.centered_beta, dims=2))) .<= atol)
    @assert effects.exact_alpha == effects.centered_alpha .* reshape(effects.tau_alpha, :, 1)
    @assert effects.exact_beta == effects.centered_beta .* reshape(effects.tau_beta, :, 1)
    @assert size(effects.raw_alpha) == size(effects.centered_alpha) == size(effects.exact_alpha)
    @assert size(effects.raw_beta) == size(effects.centered_beta) == size(effects.exact_beta)
    return true
end
_q(x) = (q05=quantile(x,.05), median=median(x), mean=mean(x), q95=quantile(x,.95))
tau_summary(effects) = (tau_alpha=_q(effects.tau_alpha), tau_beta=_q(effects.tau_beta))
function team_effect_summary(effects)
    raw = vcat(vec(effects.centered_alpha), vec(effects.centered_beta))
    exact = vcat(vec(effects.exact_alpha), vec(effects.exact_beta))
    return (centered_log_effect=_q(raw), exact_log_effect=_q(exact),
            exact_rate_multiplier=_q(exp.(exact)),
            max_abs_centered=maximum(abs.(raw)), max_abs_exact=maximum(abs.(exact)))
end

"""Tau-only counterpart to issue-01's mapping-only pxG reconstruction.
It retains its bridge, kappa, penalty/noise, wealth, month, tournament-57 league fallback, and no
clamp/floor semantics. Only centered raw alpha/beta become centered raw .* draw-wise tau."""
function reconstruct_pxg_tau_only(df, feature_set, chain::Chains; bridge=ExistingChainTeamBridge.build_name_to_existing_column(feature_set))
    base = ExistingChainTeamBridge.reconstruct_pxg_mapping_only(df, feature_set, chain; bridge=bridge)
    e = tau_effects(chain, feature_set.data[:n_teams]); assert_tau_reconstruction!(e)
    # Snapshot proves transforms did not alter the selected posterior parameter columns.
    raw_a_before, raw_b_before = copy(e.raw_alpha), copy(e.raw_beta)
    out = Dict{Int,NamedTuple}()
    for row in eachrow(df)
        hi = ExistingChainTeamBridge.corrected_team_index(row, :home, bridge)
        ai = ExistingChainTeamBridge.corrected_team_index(row, :away, bridge)
        old = base.latents[Int(row.match_id)]
        # Mapping-only's team term is centered_alpha[h] - centered_beta[a].
        z = zeros(length(old.lambda_open_h))
        old_h = (hi > 0 ? e.centered_alpha[:,hi] : z) .- (ai > 0 ? e.centered_beta[:,ai] : z)
        old_a = (ai > 0 ? e.centered_alpha[:,ai] : z) .- (hi > 0 ? e.centered_beta[:,hi] : z)
        new_h = (hi > 0 ? e.exact_alpha[:,hi] : z) .- (ai > 0 ? e.exact_beta[:,ai] : z)
        new_a = (ai > 0 ? e.exact_alpha[:,ai] : z) .- (hi > 0 ? e.exact_beta[:,hi] : z)
        open_h, open_a = old.lambda_open_h .* exp.(new_h .- old_h), old.lambda_open_a .* exp.(new_a .- old_a)
        noise_h = old.λ_h .- old.lambda_open_h; noise_a = old.λ_a .- old.lambda_open_a
        out[Int(row.match_id)] = merge(old, (lambda_open_h=open_h, lambda_open_a=open_a, λ_h=open_h .+ noise_h, λ_a=open_a .+ noise_a))
    end
    @assert e.raw_alpha == raw_a_before && e.raw_beta == raw_b_before "diagnostic mutated parameter columns"
    return (latents=out, diagnostics=base.diagnostics, effects=e)
end
end # module
