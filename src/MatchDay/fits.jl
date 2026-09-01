# src/MatchDay/fits.jl
#
# The seam between the mid-week training run and Saturday's price.
#
# `mcmc_experiments` is the register of what has been fitted; `betdb` is the register of what is
# happening. This file is the only place MatchDay touches the first one, and it deliberately does
# no sampling: a match-day price conditions on a chain that already exists, and a run that has to
# fit something before it can quote has already missed the window.

export CanonicalFit, canonical_fit, fit_registry, matchday_fit_report

"""
    CanonicalFit

A `Training.Fit` plus the provenance needed to justify pricing from it.

`run_name` and `experiment` identify the row in `mcmc_experiments.runs`; `converged` and
`failed_gates` are `Evaluation.convergence_verdict`, read once here rather than re-derived per
slate. `n_folds` is what `select_split` will index into.
"""
struct CanonicalFit{F}
    fit::F
    run_name::String
    experiment::String
    n_folds::Int
    converged::Bool
    failed_gates::Vector{String}
    loaded_at::DateTime
end

Base.getproperty(c::CanonicalFit, s::Symbol) =
    s === :config ? getfield(c, :fit).config :
    s === :training_results ? getfield(c, :fit).training_results :
    getfield(c, s)

Base.propertynames(::CanonicalFit) =
    (fieldnames(CanonicalFit)..., :config, :training_results)

"""
    canonical_fit(storage, key; require_converged = false) -> CanonicalFit

Load one trained run and audit it before it can price anything.

`storage` is a `Training.PostgresStorage` (`mcmc_experiments`) or a directory path; `key` is the
run name, its integer id, or a `UUID`. The convergence verdict is read at load time and carried,
because the gate belongs at the point the posterior enters the system rather than at the point
money is sized -- by then the caller has a sheet and an incentive.

`require_converged` defaults to **`false`** here and `true` in `Portfolio.build_books_reported`,
and the asymmetry is intentional: this function is also how a diagnostic runner inspects a bad
fit. The refusal that matters is the one in front of the bankroll.

A `CanonicalFit` is accepted anywhere a `Fit` is, because it forwards `.config` and
`.training_results` -- the only two properties `matchday_latents` and `Portfolio` read off it.
"""
function canonical_fit(storage, key; require_converged::Bool = false)
    fit = Training.load_fit(storage, key)
    passed, gates, detail = Evaluation.convergence_verdict(fit)
    if require_converged && !passed
        error("canonical_fit: run '$(Training.fit_name(fit))' failed convergence gates " *
              "$(join(gates, ", ")). An unconverged chain produces a posterior that is too " *
              "NARROW, so every p_model - p_market edge looks larger and Kelly stake is " *
              "monotone in that edge. " * join(detail, " ") *
              " Pass `require_converged = false` to load it for inspection.")
    end
    exp_name = hasproperty(storage, :experiment_name) ? String(storage.experiment_name) : ""
    return CanonicalFit(fit, Training.fit_name(fit), exp_name,
                        length(getfield(fit, :folds)), passed, String.(gates), now())
end

"""
    canonical_fit(experiment_name::AbstractString, key; kwargs...) -> CanonicalFit

Convenience over `mcmc_experiments`, resolving the DSN from `BF_EXPERIMENTS_DB_URL` and falling
back to libpq's `~/.pgpass`.
"""
canonical_fit(experiment_name::AbstractString, key; kwargs...) =
    canonical_fit(Training.PostgresStorage(String(experiment_name)), key; kwargs...)

"""
    fit_registry(storage) -> DataFrame

What is available to price from: run name, status, fold count and target date range.

The last column is the one that decides the answer. A fit whose most recent fold targets a date
AFTER the fixtures being priced has already seen them, and `select_split` will step back from it
-- loudly, but only if someone reads the warning. Reading this table first is cheaper.
"""
function fit_registry(storage)
    rows = Training.list_fits(storage; quiet = true)
    isempty(rows) && return DataFrame(name = String[], folds = Int[], converged = Bool[])
    return DataFrame(rows)
end

"""
    matchday_fit_report(cf::CanonicalFit; io = stdout)

Print the four facts that decide whether this fit can price today, and say what to do about each.
"""
function matchday_fit_report(cf::CanonicalFit; io::IO = stdout)
    println(io, "CanonicalFit '", cf.run_name, "'",
            isempty(cf.experiment) ? "" : " (" * cf.experiment * ")")
    println(io, "  model      : ", typeof(cf.config.model))
    println(io, "  folds      : ", cf.n_folds)
    println(io, "  converged  : ", cf.converged,
            cf.converged ? "" : "   <- failed: " * join(cf.failed_gates, ", "))
    println(io, "  loaded     : ", cf.loaded_at)
    cf.converged || println(io,
        "  REFUSE unless you mean it: an unconverged posterior is too narrow, so every edge " *
        "reads larger than the evidence supports and stake size is monotone in the edge.")
    return nothing
end
