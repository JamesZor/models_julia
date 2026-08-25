# Side-effect-free audit issue 05 helpers. Include l03 and issue 01's bridge first.
module LeagueIndexDiagnostics

using BayesianFootball
using DataFrames, Dates, Statistics, MCMCChains, Distributions
import Main.ExistingChainTeamBridge

export LegacyPooledLeagueContract, league_column, assert_league_contract!,
       chain_vector, posterior_summary, reconstruct_l05_league_paths,
       score_grid, market_summary, tournament_counts, league_contract_self_check!

"""Saved-chain league-column contract.

One fitted column is the legacy pooled ScottishLower artifact: tournament IDs 56 and
57 both belong to posterior column 1.  The current l05 DataFrame prediction bug instead
tries 57 as column 2 and therefore gets zero when this artifact has one column.  A future
2-column artifact uses the stored canonical map 56=>1, 57=>2.  Any other ID is an error.
"""
struct LegacyPooledLeagueContract
    n_columns::Int
end

function league_column(contract::LegacyPooledLeagueContract, tournament_id::Integer; semantics::Symbol=:artifact)
    tournament_id in (56, 57) || error("unknown tournament_id=$tournament_id; league assignment is intentionally explicit")
    contract.n_columns in (1, 2) || error("unsupported saved-chain league shape: $(contract.n_columns) columns (expected 1 or 2)")
    if semantics === :artifact
        return contract.n_columns == 1 ? 1 : (tournament_id == 56 ? 1 : 2)
    elseif semantics === :legacy_l05
        # Exact current l05 DataFrame lookup: 57=>2, then out-of-range=>zero.
        return tournament_id == 57 ? 2 : 1
    end
    error("unknown league semantics=$semantics")
end

function league_contract_self_check!()
    one, two = LegacyPooledLeagueContract(1), LegacyPooledLeagueContract(2)
    @assert (league_column(one, 56), league_column(one, 57)) == (1, 1)
    @assert (league_column(two, 56), league_column(two, 57)) == (1, 2)
    @assert league_column(one, 57; semantics=:legacy_l05) == 2
    try
        league_column(one, 999)
        error("unknown tournament accepted")
    catch err
        occursin("unknown tournament_id=999", sprint(showerror, err)) || rethrow()
    end
    return true
end

function assert_league_contract!(contract::LegacyPooledLeagueContract, df::AbstractDataFrame)
    :tournament_id in propertynames(df) || error("fixture data has no tournament_id; refusing implicit league mapping")
    unknown = sort(unique(Int(x) for x in df.tournament_id if !ismissing(x) && Int(x) ∉ (56, 57)))
    isempty(unknown) || error("unknown tournament IDs $(unknown); refusing silent league mapping")
    any(ismissing, df.tournament_id) && error("missing tournament_id; refusing implicit league mapping")
    return true
end

function chain_vector(chain::Chains, label::String)
    label in string.(names(chain)) || error("saved chain lacks $label")
    return vec(Array(chain[label]))
end
function _matrix(chain::Chains, labels::Vector{String})
    a = Array(chain[labels])
    ndims(a) == 1 && length(labels) == 1 && return reshape(a, :, 1)
    ndims(a) == 2 && return Matrix(a)
    ndims(a) == 3 && return reshape(permutedims(a, (1, 3, 2)), :, length(labels))
    error("unexpected saved-chain array shape $(size(a))")
end
_has(chain::Chains, label::String) = label in string.(names(chain))
_optional(chain, stem, n, draws) = _has(chain, "$stem[1]") ? _matrix(chain, ["$stem[$i]" for i in 1:n]) : zeros(draws, n)
_summary(x) = (q05=quantile(x, .05), median=median(x), mean=mean(x), q95=quantile(x, .95))
posterior_summary(x) = _summary(x)
tournament_counts(df::AbstractDataFrame) = combine(groupby(DataFrame(df), :tournament_id), nrow => :fixtures)

"""Reconstruct l05 pxG paths differing *only* in league-column semantics.

Both arms use issue-01's original-chain name bridge and l03's permanent
`_tau_scaled_team_effects`; saved chain arrays are only read. Kappa, penalty/referee
noise, wealth, month, and l05's lack of prediction-time clamp/floor are shared verbatim.
`:legacy_l05` preserves 57=>missing column=>zero for a one-column artifact; `:artifact`
uses the chain-aware contract.
"""
function reconstruct_l05_league_paths(df::AbstractDataFrame, feature_set, chain::Chains;
                                      bridge=ExistingChainTeamBridge.build_name_to_existing_column(feature_set))
    data = feature_set.data
    contract = LegacyPooledLeagueContract(data[:n_leagues])
    assert_league_contract!(contract, df)
    ExistingChainTeamBridge.assert_bridge_invariants!(feature_set, bridge)
    nt, nr = data[:n_teams], data[:n_refs]
    base, ha, wealth = chain_vector(chain, "base_mu"), chain_vector(chain, "ha_home"), chain_vector(chain, "w_wealth")
    draws = length(base)
    # Deliberately call the permanent production helper, rather than duplicating tau arithmetic.
    effects = Main._tau_scaled_team_effects(chain, nt; context="issue-05 l05 league diagnostic")
    alpha, beta = effects.alpha, effects.beta
    kappa = exp.(_optional(chain, "raw_kappa", nt, draws))
    months = _optional(chain, "delta_month", data[:n_months], draws)
    leagues = _optional(chain, "delta_league", contract.n_columns, draws)
    pen, ha_pen, sigma = chain_vector(chain, "pen_base_mu"), chain_vector(chain, "ha_pen"), chain_vector(chain, "sigma_ref")
    gamma = _optional(chain, "raw_gamma_ref", nr, draws) .* reshape(sigma, :, 1)
    apd, bpf = _optional(chain, "alpha_pen_draw", nt, draws), _optional(chain, "beta_pen_foul", nt, draws)
    paths = Dict(:legacy_l05 => Dict{Int,NamedTuple}(), :artifact => Dict{Int,NamedTuple}())
    getv(mat, i, default) = i > 0 ? mat[:, i] : fill(default, draws)
    for row in eachrow(df)
        hi = ExistingChainTeamBridge.corrected_team_index(row, :home, bridge)
        ai = ExistingChainTeamBridge.corrected_team_index(row, :away, bridge)
        # Unknown team behavior is retained from l05; unknown tournaments were rejected above.
        αh, βh, αa, βa = getv(alpha, hi, 0.0), getv(beta, hi, 0.0), getv(alpha, ai, 0.0), getv(beta, ai, 0.0)
        κh, κa = getv(kappa, hi, 1.0), getv(kappa, ai, 1.0)
        mi = month(row.match_date)
        δm = 1 <= mi <= data[:n_months] ? months[:, mi] : zeros(draws)
        ws = wealth .* get(data[:wealth_map], Int(row.match_id), 0.0)
        ri = hasproperty(row, :referee_id) && !ismissing(row.referee_id) ? get(data[:ref_map], Int(row.referee_id), -1) : -1
        γ = getv(gamma, ri, 0.0)
        ph = exp.(pen .+ ha_pen .+ γ .+ getv(apd, hi, 0.0) .+ getv(bpf, ai, 0.0))
        pa = exp.(pen .- ha_pen .+ γ .+ getv(apd, ai, 0.0) .+ getv(bpf, hi, 0.0))
        for semantics in (:legacy_l05, :artifact)
            li = league_column(contract, Int(row.tournament_id); semantics)
            # The exact l05 out-of-range fallback is zero, not a remapped column.
            δl = 1 <= li <= contract.n_columns ? leagues[:, li] : zeros(draws)
            core = base .+ δm .+ δl
            true_h = exp.(core .+ ha .+ αh .- βa .+ ws)
            true_a = exp.(core .+ αa .- βh .- ws)
            open_h, open_a = κh .* true_h, κa .* true_a
            paths[semantics][Int(row.match_id)] = (lambda_open_h=open_h, lambda_open_a=open_a,
                lambda_pen_h=ph, lambda_pen_a=pa, λ_h=open_h .+ .768 .* ph .+ .0276,
                λ_a=open_a .+ .768 .* pa .+ .0276, true_xg_h=true_h, true_xg_a=true_a)
        end
    end
    return (paths=paths, contract=contract, effects=effects,
            unknown_teams=ExistingChainTeamBridge.unknown_team_diagnostics(df, bridge))
end

function score_grid(latent; max_goals::Int=12)
    n = length(latent.λ_h); grid = zeros(Float64, max_goals, max_goals, n)
    for d in 1:n
        nh, na = .768 * latent.lambda_pen_h[d] + .0276, .768 * latent.lambda_pen_a[d] + .0276
        ph = [pdf(Poisson(max(1e-4, latent.lambda_open_h[d])), g) for g in 0:max_goals-1]
        pa = [pdf(Poisson(max(1e-4, latent.lambda_open_a[d])), g) for g in 0:max_goals-1]
        qh = [pdf(Poisson(max(1e-4, nh)), g) for g in 0:max_goals-1]
        qa = [pdf(Poisson(max(1e-4, na)), g) for g in 0:max_goals-1]
        h = [sum(ph[k+1] * qh[g-k+1] for k in 0:g) for g in 0:max_goals-1]
        a = [sum(pa[k+1] * qa[g-k+1] for k in 0:g) for g in 0:max_goals-1]
        h ./= sum(h); a ./= sum(a); grid[:, :, d] = h * a'
    end
    return grid
end
function market_summary(grid)
    p = dropdims(mean(grid, dims=3), dims=3); n = size(p, 1)
    return (home_win=sum(p[i,j] for i in 1:n for j in 1:n if i > j), draw=sum(p[i,i] for i in 1:n),
            away_win=sum(p[i,j] for i in 1:n for j in 1:n if i < j), btts=sum(p[i,j] for i in 2:n for j in 2:n),
            over_2_5=sum(p[i,j] for i in 1:n for j in 1:n if i+j >= 5))
end
end # module
