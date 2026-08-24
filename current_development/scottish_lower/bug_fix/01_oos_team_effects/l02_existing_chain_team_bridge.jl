# Artifact-compatible Phase 2 bridge for issue 01.
# This module is deliberately local to the investigation: it does not extend/override production dispatch.
module ExistingChainTeamBridge

using BayesianFootball
using DataFrames, Dates, Statistics, MCMCChains, Distributions

export existing_team_map, build_name_to_existing_column, corrected_team_index,
       legacy_team_index, unknown_team_diagnostics, mapping_comparison,
       assert_bridge_invariants!, bridge_self_check!, reconstruct_pxg_mapping_only,
       reconstruct_pxg_fitted, score_matrix

existing_team_map(feature_set)::AbstractDict = feature_set.data[:team_map]

"""Return canonical name => *existing posterior column* without renumbering anything.

The values are copied directly from the FeatureSet's integer `team_map`; consequently saved-chain
`raw_alpha[i]`, `raw_beta[i]`, kappa, and penalty columns retain their fitted ordering. `-1` is not
stored for a known identity: it is returned only by `corrected_team_index` for an unknown name.
"""
function build_name_to_existing_column(feature_set)::Dict{String,Int}
    team_map = existing_team_map(feature_set)
    clean_df = feature_set.data[:clean_df]
    required = (:home_team, :away_team, :home_team_id, :away_team_id)
    missing_columns = setdiff(collect(required), propertynames(clean_df))
    isempty(missing_columns) || error("clean_df is missing $(missing_columns)")
    out = Dict{String,Int}()
    for row in eachrow(clean_df), (name_col, id_col) in ((:home_team, :home_team_id), (:away_team, :away_team_id))
        name, id = getproperty(row, name_col), getproperty(row, id_col)
        (ismissing(name) || ismissing(id)) && continue
        column = get(team_map, Int(id), nothing)
        isnothing(column) && continue # known globally, but genuinely not fitted in this fold
        key = String(name)
        if haskey(out, key) && out[key] != column
            error("Team name '$key' maps to multiple existing posterior columns")
        end
        out[key] = column
    end
    return out
end

function corrected_team_index(row, side::Symbol, name_to_column::AbstractDict{String,<:Integer})::Int
    field = side === :home ? :home_team : side === :away ? :away_team : error("side must be :home or :away")
    hasproperty(row, field) || return -1
    name = getproperty(row, field)
    return ismissing(name) ? -1 : get(name_to_column, String(name), -1)
end

"Exact broken two-stage l05 lookup, retained solely for comparison."
function legacy_team_index(row, side::Symbol, team_map::AbstractDict)::Int
    id_field, name_field = side === :home ? (:home_team_id, :home_team) : (:away_team_id, :away_team)
    id = if hasproperty(row, id_field)
        value = getproperty(row, id_field); ismissing(value) ? -1 : Int(value)
    elseif hasproperty(row, name_field)
        value = getproperty(row, name_field); ismissing(value) ? -1 : get(team_map, value, -1)
    else
        -1
    end
    return get(team_map, id, -1)
end

function unknown_team_diagnostics(df::AbstractDataFrame, name_to_column::AbstractDict{String,<:Integer})
    names = String[]
    for side in (:home, :away), row in eachrow(df)
        field = side === :home ? :home_team : :away_team
        hasproperty(row, field) || continue
        value = getproperty(row, field)
        !ismissing(value) && get(name_to_column, String(value), -1) <= 0 && push!(names, String(value))
    end
    unique_names = sort(unique(names))
    return (unknown_sides = length(names), unknown_names = unique_names,
            unknown_matches = count(eachrow(df)) do row
                corrected_team_index(row, :home, name_to_column) <= 0 || corrected_team_index(row, :away, name_to_column) <= 0
            end)
end

function mapping_comparison(df::AbstractDataFrame, feature_set)
    team_map = existing_team_map(feature_set); bridge = build_name_to_existing_column(feature_set)
    out = DataFrame(match_id=Int[], home_team=String[], away_team=String[], legacy_home_idx=Int[], corrected_home_idx=Int[], legacy_away_idx=Int[], corrected_away_idx=Int[])
    for row in eachrow(df)
        push!(out, (Int(row.match_id), String(row.home_team), String(row.away_team),
                    legacy_team_index(row, :home, team_map), corrected_team_index(row, :home, bridge),
                    legacy_team_index(row, :away, team_map), corrected_team_index(row, :away, bridge)))
    end
    return out
end

"Fail if the bridge has changed an artifact's existing column ordering."
function assert_bridge_invariants!(feature_set, bridge=build_name_to_existing_column(feature_set))
    team_map = existing_team_map(feature_set)
    all(i -> 1 <= i <= length(team_map), values(bridge)) || error("bridge contains invalid posterior column")
    clean = feature_set.data[:clean_df]
    for row in eachrow(clean), (n, id) in ((row.home_team, row.home_team_id), (row.away_team, row.away_team_id))
        (ismissing(n) || ismissing(id) || !haskey(team_map, Int(id))) && continue
        get(bridge, String(n), -1) == team_map[Int(id)] || error("posterior-column permutation for $(n)")
    end
    return true
end

# Chain extraction explicitly requests [1:n] in fitted column order and stacks chains as draw rows.
function _matrix(chain::Chains, labels::Vector{String})
    a = Array(chain[labels])
    ndims(a) == 2 && return Matrix(a)
    ndims(a) == 3 && return reshape(permutedims(a, (1, 3, 2)), :, length(labels))
    error("unexpected chain dimensions $(size(a))")
end
_vector(chain::Chains, name::String) = vec(Array(chain[name]))
_has(chain::Chains, name::String) = name in Set(string.(names(chain)))
_optional_matrix(chain, stem, n, draws) = _has(chain, "$stem[1]") ? _matrix(chain, ["$stem[$i]" for i in 1:n]) : zeros(draws, n)

"""Shared pxG reconstruction engine.

`fitted=false` deliberately preserves the current l05 dataframe extractor's arithmetic and
fallbacks; only the name-to-existing-column lookup differs. `fitted=true` applies the transform
actually used by `_turing_recomb_pxg_wealth`.
"""
function _reconstruct_pxg(df::AbstractDataFrame, feature_set, chain::Chains, bridge; fitted::Bool)
    assert_bridge_invariants!(feature_set, bridge)
    d = feature_set.data; nt, nr = d[:n_teams], d[:n_refs]
    base, ha, wealth = _vector(chain,"base_mu"), _vector(chain,"ha_home"), _vector(chain,"w_wealth")
    draws = length(base)
    raw_a, raw_b = _matrix(chain,["raw_alpha[$i]" for i in 1:nt]), _matrix(chain,["raw_beta[$i]" for i in 1:nt])
    # The mapping-only path intentionally retains l05's unscaled centered raw effects.
    alpha, beta = raw_a .- mean(raw_a,dims=2), raw_b .- mean(raw_b,dims=2)
    if fitted
        alpha .*= reshape(_vector(chain,"tau_alpha"),:,1)
        beta  .*= reshape(_vector(chain,"tau_beta"),:,1)
    end
    kappa = exp.(_optional_matrix(chain,"raw_kappa",nt,draws))
    dm = _optional_matrix(chain,"delta_month",d[:n_months],draws); dl = _optional_matrix(chain,"delta_league",d[:n_leagues],draws)
    pen, ha_pen, sigma = _vector(chain,"pen_base_mu"), _vector(chain,"ha_pen"), _vector(chain,"sigma_ref")
    gamma = _optional_matrix(chain,"raw_gamma_ref",nr,draws) .* reshape(sigma,:,1)
    apd, bpf = _optional_matrix(chain,"alpha_pen_draw",nt,draws), _optional_matrix(chain,"beta_pen_foul",nt,draws)
    out = Dict{Int,NamedTuple}()
    for row in eachrow(df)
        hi, ai = corrected_team_index(row,:home,bridge), corrected_team_index(row,:away,bridge)
        getv(mat,i,default) = i > 0 ? mat[:,i] : fill(default,draws)
        αh, βh, αa, βa = getv(alpha,hi,0.0), getv(beta,hi,0.0), getv(alpha,ai,0.0), getv(beta,ai,0.0)
        κh, κa = getv(kappa,hi,1.0), getv(kappa,ai,1.0)
        mi = month(row.match_date)
        # l05's dataframe extractor tries tournament 57 as league 2; training has only league 1.
        li = fitted ? 1 : (hasproperty(row,:tournament_id) && row.tournament_id == 57 ? 2 : 1)
        δm = 1 <= mi <= d[:n_months] ? dm[:,mi] : zeros(draws); δl = 1 <= li <= d[:n_leagues] ? dl[:,li] : zeros(draws)
        ws = wealth .* get(d[:wealth_map],Int(row.match_id),0.0); core = base .+ δm .+ δl
        log_h = core .+ ha .+ αh .- βa .+ ws; log_a = core .+ αa .- βh .- ws
        true_h = fitted ? exp.(clamp.(log_h,-10.0,10.0)) .+ 1e-6 : exp.(log_h)
        true_a = fitted ? exp.(clamp.(log_a,-10.0,10.0)) .+ 1e-6 : exp.(log_a)
        ri = hasproperty(row,:referee_id) && !ismissing(row.referee_id) ? get(d[:ref_map],Int(row.referee_id),-1) : -1
        γ = ri > 0 ? gamma[:,ri] : zeros(draws)
        lph = pen .+ ha_pen .+ γ .+ getv(apd,hi,0.0) .+ getv(bpf,ai,0.0)
        lpa = pen .- ha_pen .+ γ .+ getv(apd,ai,0.0) .+ getv(bpf,hi,0.0)
        ph = fitted ? exp.(clamp.(lph,-10.0,5.0)) .+ 1e-6 : exp.(lph)
        pa = fitted ? exp.(clamp.(lpa,-10.0,5.0)) .+ 1e-6 : exp.(lpa)
        oh, oa = κh .* true_h, κa .* true_a
        out[Int(row.match_id)] = (λ_h=oh .+ .768 .* ph .+ .0276, λ_a=oa .+ .768 .* pa .+ .0276,
            r_h=fill(100.0,draws), r_a=fill(100.0,draws), true_xg_h=true_h, true_xg_a=true_a,
            lambda_pen_h=ph, lambda_pen_a=pa, lambda_open_h=oh, lambda_open_a=oa)
    end
    return (latents=out, diagnostics=unknown_team_diagnostics(df,bridge))
end

"""Reproduce l05's current dataframe extraction exactly, except for the corrected name bridge.

This intentionally uses unscaled centered raw alpha/beta, l05's tournament-57 league lookup and
out-of-range fallback, and no prediction-time clamps or `+1e-6` floors.
"""
reconstruct_pxg_mapping_only(df::AbstractDataFrame, feature_set, chain::Chains; bridge=build_name_to_existing_column(feature_set)) =
    _reconstruct_pxg(df, feature_set, chain, bridge; fitted=false)

"""Reconstruct the transform actually fitted by Turing, using the corrected name bridge.

Applies tau scaling, training log clamps and floors, and training league semantics (league index 1).
"""
reconstruct_pxg_fitted(df::AbstractDataFrame, feature_set, chain::Chains; bridge=build_name_to_existing_column(feature_set)) =
    _reconstruct_pxg(df, feature_set, chain, bridge; fitted=true)

"""l05-compatible truncated open-play/noise Poisson convolution score grid."""
function score_matrix(latent; max_goals::Int=12)
    n=length(latent.λ_h); s=zeros(Float64,max_goals,max_goals,n)
    for k in 1:n
        oh, oa = latent.lambda_open_h[k], latent.lambda_open_a[k]
        nh, na = .768 * latent.lambda_pen_h[k] + .0276, .768 * latent.lambda_pen_a[k] + .0276
        poh=[pdf(Poisson(max(1e-4,oh)),g) for g in 0:max_goals-1]; pnh=[pdf(Poisson(max(1e-4,nh)),g) for g in 0:max_goals-1]
        poa=[pdf(Poisson(max(1e-4,oa)),g) for g in 0:max_goals-1]; pna=[pdf(Poisson(max(1e-4,na)),g) for g in 0:max_goals-1]
        h=[sum(poh[m+1]*pnh[g-m+1] for m in 0:g) for g in 0:max_goals-1]
        a=[sum(poa[m+1]*pna[g-m+1] for m in 0:g) for g in 0:max_goals-1]
        h ./= sum(h); a ./= sum(a); s[:,:,k]=h*a'
    end
    return BayesianFootball.Predictions.ScoreMatrix(s)
end

function bridge_self_check!()
    clean=DataFrame(home_team=["B","A"],away_team=["A","C"],home_team_id=[20,10],away_team_id=[10,30])
    fs=(data=Dict(:team_map=>Dict(20=>1,10=>2),:clean_df=>clean),); b=build_name_to_existing_column(fs)
    @assert b == Dict("B"=>1,"A"=>2) # original map order, not alphabetical name order
    @assert corrected_team_index((home_team="A",),:home,b)==2 && corrected_team_index((home_team="Z",),:home,b)==-1
    unknown=unknown_team_diagnostics(DataFrame(home_team=["A"],away_team=["Z"]),b)
    @assert unknown.unknown_names == ["Z"] && unknown.unknown_sides == 1
    @assert assert_bridge_invariants!(fs,b); return true
end

end # module
