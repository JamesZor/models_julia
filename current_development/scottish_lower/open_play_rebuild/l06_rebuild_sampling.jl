module RebuildSampling

using BayesianFootball
using MCMCChains
using DataFrames
using Statistics
using Serialization
using Dates
using UUIDs
using Turing

const Samplers = BayesianFootball.Samplers

export PriorInit, atomic_serialize, diagnostics, hard_smoke_pass, generic_dataframe,
       dataframe_roundtrip_ok, sampler_metadata

"""Project sampler initialisation strategy: each chain asks Turing for an independent prior draw."""
struct PriorInit <: Samplers.AbstractInitStrategy end
Samplers.get_init_params(model, ::PriorInit, n_chains::Int) =
    [Turing.InitFromPrior() for _ in 1:n_chains]

function atomic_serialize(path::AbstractString, value)
    ispath(path) && throw(ArgumentError("refusing to overwrite existing artifact: $path"))
    mkpath(dirname(path)); tmp = path * ".tmp-" * string(uuid4())
    try
        serialize(tmp, value)
        mv(tmp, path; force=false)
    finally
        ispath(tmp) && rm(tmp; force=true)
    end
    return path
end

# A deliberately small mirror of generic Dict -> DataFrame persistence: scalar provenance/status
# remains a scalar column, while vectors/matrices remain cell values.
_cell(x) = x isa Symbol ? String(x) : x
function generic_dataframe(rows)
    isempty(rows) && return DataFrame()
    keys0 = sort!(collect(union((Set(keys(r)) for r in rows)...)))
    cols = Dict{Symbol,Vector{Any}}()
    for k in keys0
        cols[k] = [_cell(get(r, k, missing)) for r in rows]
    end
    return DataFrame(cols)
end
function dataframe_roundtrip_ok(df::DataFrame)
    io = IOBuffer(); serialize(io, df); seekstart(io); restored = deserialize(io)
    nrow(restored) == nrow(df) && names(restored) == names(df) &&
        all(isequal.(eachcol(restored), eachcol(df)))
end

const _SAMPLER = Set(Symbol.("lp n_steps acceptance_rate tree_depth numerical_error step_size nom_step_size is_accept hamiltonian_energy hamiltonian_energy_error max_hamiltonian_energy_error" |> split))
_get(row, syms) = begin
    for s in syms
        hasproperty(row, s) && begin v = getproperty(row, s); v isa Number && return Float64(v); end
    end
    missing
end
function diagnostics(chain::Chains; max_depth::Int)
    labels = Symbol.(MCMCChains.names(chain, :parameters)); summary = DataFrame(MCMCChains.summarize(chain))
    parameter_rows = filter(r -> Symbol(r.parameters) in labels, eachrow(summary))
    metric(which, syms, op) = begin
        vals = [(String(r.parameters), _get(r, syms)) for r in parameter_rows]
        good = filter(x -> !ismissing(x[2]) && isfinite(x[2]), vals)
        isempty(good) ? (value=missing, label=missing) : begin x = op(good; by=last); (value=x[2], label=x[1]) end
    end
    maxr = metric(:rhat, (:rhat,), maximum); bulk = metric(:bulk, (:ess_bulk,:ess), minimum); tail = metric(:tail, (:ess_tail,), minimum)
    available(s) = s in Symbol.(MCMCChains.names(chain))
    flat(s) = available(s) ? vec(Float64.(Array(chain[s]))) : Float64[]
    div = available(:numerical_error) ? sum(flat(:numerical_error) .!= 0) : missing
    depth = flat(:tree_depth); caps = isempty(depth) ? missing : sum(depth .>= max_depth)
    capfrac = ismissing(caps) ? missing : caps / length(depth)
    lp = flat(:lp); finite_lp = isempty(lp) ? missing : all(isfinite, lp)
    accept = available(:acceptance_rate) ? mean(flat(:acceptance_rate)) : missing
    bfmi = if available(:hamiltonian_energy)
        e = Array(chain[:hamiltonian_energy])
        energy_chains = if ndims(e) == 3
            [vec(e[:, 1, c]) for c in axes(e, 3)]
        elseif ndims(e) == 2
            [vec(e[:, c]) for c in axes(e, 2)]
        else
            error("unexpected Hamiltonian-energy shape $(size(e))")
        end
        [var(ec) > 0 ? mean(diff(ec).^2) / var(ec) : missing for ec in energy_chains]
    else
        missing
    end
    return (; max_rhat=maxr.value, max_rhat_label=maxr.label, min_bulk_ess=bulk.value, min_bulk_ess_label=bulk.label,
        min_tail_ess=tail.value, min_tail_ess_label=tail.label, divergences=div, max_tree_depth=isempty(depth) ? missing : maximum(depth),
        depth_cap_hits=caps, depth_cap_fraction=capfrac, acceptance=accept, finite_lp=finite_lp, bfmi=bfmi,
        diagnostics_available=(rhat=!ismissing(maxr.value), bulk=!ismissing(bulk.value), tail=!ismissing(tail.value), divergences=!ismissing(div), lp=!ismissing(finite_lp)))
end
hard_smoke_pass(d) = d.diagnostics_available.rhat && d.diagnostics_available.bulk && d.diagnostics_available.tail && d.diagnostics_available.divergences && d.diagnostics_available.lp && d.max_rhat <= 1.05 && d.min_bulk_ess >= 100 && d.min_tail_ess >= 100 && d.divergences == 0 && d.finite_lp
sampler_metadata(c) = (; samples=c.n_samples, warmup=c.n_warmup, chains=c.n_chains, accept_rate=c.accept_rate, max_depth=c.max_depth, init=string(typeof(c.initialisation)))

end
