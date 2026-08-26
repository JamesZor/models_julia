# ==============================================================================
# Shared protocol — GATE 4 : EXTRACTION
# ==============================================================================
# Gate 3 proves the fitted density.  This gate separately proves that the chain
# reader and pricing path preserve that fitted posterior.  All model arithmetic
# and extraction shape are adapter-owned; ordering, IDs, depth, and finiteness
# are protocol invariants.

using MCMCChains
using DataFrames
using Statistics
using Printf

# ------------------------------------------------------------------------------
# 1. Synthetic chain construction
# ------------------------------------------------------------------------------

"""
    sl_synthetic_chain(adapter, draws; n_chains=2) -> Chains

Place deliberately distinct draws in the exact column-major ordering used by
`vec(Array(chain[site]))`: iteration changes before chain.  More than one chain
is required, otherwise a row/column ordering bug has no chance to be exposed.
"""
function sl_synthetic_chain(adapter::AbstractSLModelAdapter, draws; n_chains::Int = 2)
    isempty(draws) && error("synthetic draws must be non-empty")
    n_chains > 1 || error("n_chains must exceed one to test cross-chain draw ordering")
    length(draws) % n_chains == 0 || error("draw count must divide evenly into n_chains")

    n_iter  = length(draws) ÷ n_chains
    n_teams = sl_synthetic_n_teams(adapter, first(draws))
    sites   = Symbol.(sl_sampled_sites(adapter, n_teams))
    array   = zeros(Float64, n_iter, length(sites), n_chains)
    for (k, draw) in enumerate(draws)
        iteration = (k - 1) % n_iter + 1
        chain     = (k - 1) ÷ n_iter + 1
        row = sl_parameter_row(adapter, draw)
        length(row) == length(sites) || error("adapter parameter row has $(length(row)) values for $(length(sites)) chain columns")
        array[iteration, :, chain] = row
    end
    return MCMCChains.Chains(array, sites)
end

function _sl_extraction_schema(adapter)
    caps = sl_capabilities(adapter)
    hasproperty(caps, :extraction_schema) || return nothing
    schema = caps.extraction_schema
    all(hasproperty(schema, x) for x in (:posterior_fields, :positive_fields)) || return nothing
    return schema
end

function _sl_vector_values(df, fields)
    values = Float64[]
    for field in fields
        field in propertynames(df) || return nothing
        for value in df[!, field]
            value isa AbstractVector{<:Real} || return nothing
            append!(values, value)
        end
    end
    return values
end

# ------------------------------------------------------------------------------
# 2. GATE 4a — synthetic-chain parity
# ------------------------------------------------------------------------------

function sl_gate_extraction_synthetic(adapter::AbstractSLModelAdapter, fs;
                                      n_draws::Int = 8, n_chains::Int = 2,
                                      tol::Float64 = 1e-12)
    n_draws > 0 && n_chains > 1 && n_draws % n_chains == 0 ||
        return [sl_result("synthetic extraction inputs", false, "positive draws divisible by at least two chains required")]
    sl_assert_model_contract(adapter)

    draws = sl_synthetic_draws(adapter, Int(fs.data[:n_teams]), n_draws)
    isempty(draws) && return [sl_result("synthetic extraction inputs", false, "adapter returned no draws")]
    chain = sl_synthetic_chain(adapter, draws; n_chains)
    fixtures = sl_synthetic_fixtures(adapter, fs.data[:team_map]; n = 6)
    priced = sl_extract_parameters(adapter, fixtures, fs, chain)

    results = Any[
        sl_result("every fixture priced",
            length(priced) == nrow(fixtures) && all(haskey(priced, Int(row.match_id)) for row in eachrow(fixtures)),
            "$(length(priced)) of $(nrow(fixtures)) fixtures returned"),
    ]

    # The adapter reference is scalar-per-draw; extraction is vector-per-draw.
    # Compare every declared reference field and retain the separate draw-ordering
    # check below so a coincidental identity permutation cannot pass unnoticed.
    errors = Float64[]
    fields = nothing
    for fixture in eachrow(fixtures)
        got = get(priced, Int(fixture.match_id), nothing)
        got === nothing && continue
        for (k, params) in enumerate(draws)
            expected = sl_reference_extract(adapter, params, fixture, fs)
            expected_fields = propertynames(expected)
            fields === nothing ? (fields = expected_fields) : fields == expected_fields || error("adapter reference extraction fields change between draws")
            for field in expected_fields
                hasproperty(got, field) || error("extractor omitted reference field $field")
                observed = getproperty(got, field)
                observed isa AbstractVector || error("extractor field $field is not posterior-vector valued")
                k <= length(observed) || error("extractor field $field lost posterior draw $k")
                push!(errors, abs(observed[k] - getproperty(expected, field)))
            end
        end
    end
    fields === nothing && return vcat(results, [sl_result("extraction parity vs reference", false, "no comparable fields")])
    first_price = priced[Int(first(fixtures.match_id))]
    preserved = all(hasproperty(first_price, field) && length(getproperty(first_price, field)) == n_draws for field in fields)
    push!(results, sl_result("posterior draw depth preserved", preserved,
                             "$n_draws draws expected across $n_chains chains"))
    push!(results, sl_result("extraction parity vs reference", !isempty(errors) && maximum(errors) <= tol,
                             isempty(errors) ? "no comparable fields" : @sprintf("max |Δ| = %.3e over %d draws x %d fixtures", maximum(errors), n_draws, nrow(fixtures))))

    # Distinct synthetic parameters must result in a distinct price for at least
    # one adapter-declared reference field.  This catches collapsed and transposed
    # extraction even when the arithmetic comparison is otherwise correct.
    distinct = false
    for field in fields
        values = getproperty(first_price, field)
        distinct |= length(unique(round.(values; digits = 12))) == n_draws
    end
    push!(results, sl_result("draws not collapsed", distinct,
                             distinct ? "$n_draws distinct posterior prices" : "no priced field retained all $n_draws distinct draws"))
    return vcat(results, sl_adapter_check(adapter, :extraction_synthetic, fs, draws, fixtures, priced))
end

# ------------------------------------------------------------------------------
# 3. GATE 4c — population and index fallbacks
# ------------------------------------------------------------------------------

function sl_gate_extraction_fallbacks(adapter::AbstractSLModelAdapter, fs; n_draws::Int = 8)
    n_draws > 0 || return [sl_result("fallback inputs", false, "n_draws must be positive")]
    draws = sl_synthetic_draws(adapter, Int(fs.data[:n_teams]), n_draws)
    fixtures = sl_synthetic_fixtures(adapter, fs.data[:team_map]; n = 2, unmapped = true)
    chain = sl_synthetic_chain(adapter, draws; n_chains = 2)
    priced = sl_extract_parameters(adapter, fixtures, fs, chain)

    unmapped = fixtures[end, :]
    fallback = get(priced, Int(unmapped.match_id), nothing)
    reference = fallback === nothing ? nothing : [sl_reference_extract(adapter, p, unmapped, fs) for p in draws]
    fields = reference === nothing || isempty(reference) ? () : propertynames(first(reference))
    errors = Float64[]
    if fallback !== nothing
        for (k, expected) in enumerate(reference), field in fields
            hasproperty(fallback, field) || continue
            value = getproperty(fallback, field)
            k <= length(value) || continue
            push!(errors, abs(value[k] - getproperty(expected, field)))
        end
    end
    out = Any[
        sl_result("unmapped fixture returned", fallback !== nothing, fallback === nothing ? "unmapped synthetic match_id missing" : "match_id $(unmapped.match_id) returned"),
        sl_result("fallback posterior depth preserved", fallback !== nothing && all(length(getproperty(fallback, f)) == n_draws for f in fields), "$n_draws draws required for fallback fixture"),
        sl_result("fallback parity vs reference", !isempty(errors) && maximum(errors) <= 1e-12,
                  isempty(errors) ? "no comparable fallback values" : @sprintf("max |Δ| = %.3e", maximum(errors))),
    ]
    return vcat(out, sl_adapter_check(adapter, :extraction_fallback, fs, draws, fixtures, priced))
end

# ------------------------------------------------------------------------------
# 4. GATE 4b — persisted real chain, ordinary OOS loader
# ------------------------------------------------------------------------------

function sl_gate_extraction_real(ds, results, adapter::AbstractSLModelAdapter, contract::SLContract)
    latents = SLExperiments.extract_oos_predictions(ds, results; force = true)
    df = latents.df
    chains = [c for (c, _) in results.training_results]
    schema = _sl_extraction_schema(adapter)
    out = Any[]

    # Support both 1-fold smoke artifacts and multi-fold grid artifacts dynamically
    n_folds = length(results.training_results)
    splitter = n_folds == 1 ? _sl_smoke_splitter(contract) : _sl_grid_splitter(contract)
    boundaries = SLData.create_id_boundaries(ds, splitter)
    oos_list = [DataFrame(SLData.get_next_matches(ds, b, splitter)) for b in boundaries[1:min(n_folds, length(boundaries))]]
    oos = isempty(oos_list) ? DataFrame() : vcat(oos_list...)
    push!(out, sl_result("OOS fixtures priced", nrow(df) == nrow(oos) && nrow(df) > 0,
                         "$(nrow(df)) rows priced across $(n_folds) fold(s), $(nrow(oos)) OOS fixtures expected"))
    ids_present = :match_id in propertynames(df) && :match_id in propertynames(oos)
    missing_ids = ids_present ? length(setdiff(Set(oos.match_id), Set(df.match_id))) : nrow(oos)
    extra_ids = ids_present ? length(setdiff(Set(df.match_id), Set(oos.match_id))) : nrow(df)
    push!(out, sl_result("match ids match the OOS set", ids_present && missing_ids == 0 && extra_ids == 0,
                         "$missing_ids missing, $extra_ids unexpected"))

    expected_depth = isempty(chains) ? 0 : size(first(chains), 1) * size(first(chains), 3)
    schema_ok = schema !== nothing
    posterior_fields = schema_ok ? schema.posterior_fields : Symbol[]
    positive_fields = schema_ok ? schema.positive_fields : Symbol[]
    push!(out, sl_result("extraction schema declared", schema_ok,
                         schema_ok ? "posterior fields $(join(string.(posterior_fields), ", ")); positive fields $(join(string.(positive_fields), ", "))" : "adapter capabilities lack extraction_schema(posteriors_fields, positive_fields)"))
    depth_ok = schema_ok && expected_depth > 0 && all(field in propertynames(df) && all(v -> v isa AbstractVector && length(v) == expected_depth, df[!, field]) for field in posterior_fields)
    push!(out, sl_result("posterior depth preserved", depth_ok,
                         "$expected_depth draws per fixture (chain is $(isempty(chains) ? "missing" : "$(size(first(chains),1)) x $(size(first(chains),3))"))"))

    values = schema_ok ? _sl_vector_values(df, posterior_fields) : nothing
    finite_ok = values !== nothing && !isempty(values) && all(isfinite, values)
    push!(out, sl_result("numeric posterior values finite", finite_ok,
                         values === nothing ? "missing/non-vector posterior field" : isempty(values) ? "no posterior values" : @sprintf("%d values, range [%.3g, %.3g]", length(values), minimum(values), maximum(values))))
    positive = schema_ok ? _sl_vector_values(df, positive_fields) : nothing
    positive_ok = positive !== nothing && !isempty(positive) && all(isfinite, positive) && all(>(0), positive)
    push!(out, sl_result("schema-positive posterior values", positive_ok,
                         positive === nothing ? "positive field missing/non-vector" : isempty(positive) ? "no schema-positive values" : @sprintf("%d values, range [%.3g, %.3g]", length(positive), minimum(positive), maximum(positive))))
    return vcat(out, sl_adapter_check(adapter, :extraction_real, ds, results, latents, contract)), latents
end
