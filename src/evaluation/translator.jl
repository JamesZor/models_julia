# src/evaluation/translator.jl

export to_dataframe_row

# Base Case: It's just a number. Wrap it in a NamedTuple.
function unroll(prefix::String, val::Real)
    return NamedTuple{(Symbol(prefix),)}((val,))
end

# `missing` is a leaf too. `MIQStats`' fields are `Union{Missing, Float64}` and become
# `missing` whenever a selection group has fewer than two winners or two losers — which
# any store that does not quote Over/Under 1.5 or 3.5 guarantees for those four fields.
# Without this method `to_dataframe_row(exp, MIQResult(...))` raises `MethodError` inside
# `evaluate_experiments`' `try`, which drops the model's ENTIRE row with only a `@warn`.
# Additive: nothing that worked before behaves differently.
function unroll(prefix::String, ::Missing)
    return NamedTuple{(Symbol(prefix),)}((missing,))
end

# Recursive Case: It's a nested component (like DistributionStats). Dive inside!
function unroll(prefix::String, comp::AbstractMetricComponent)
    keys = propertynames(comp)
    # Recursively unroll each field
    unrolled_tuples = [unroll("$(prefix)_$(k)", getproperty(comp, k)) for k in keys]
    return merge(unrolled_tuples...)
end

"""
    _metric_selection_suffix(metric)

Suffix appended to a metric's column prefix so that PER-SELECTION metrics
(e.g. `LogLoss(:over_25)`, `GLMEdge(:under_25)`) produce DISTINCT columns and do
not collide/overwrite each other inside a single `evaluate_experiments` call.
Aggregate metrics (empty/absent `selections`) get `""` → unchanged column names,
preserving backward compatibility with every existing runner.
"""
function _metric_selection_suffix(metric::AbstractScoringRule)
    if hasproperty(metric, :selections) && !isempty(metric.selections)
        return "_" * join(String.(metric.selections), "_")
    end
    return ""
end

"""
    to_dataframe_row(exp::ExperimentResults, metric::AbstractScoringRule, result::AbstractEvaluationResult)

Flattens any nested AbstractEvaluationResult into a single, flat NamedTuple
that can be easily pushed into a DataFrame. The `metric` is used to append a
per-selection suffix so filtered metrics stay distinct.
"""
function to_dataframe_row(exp::ExperimentResults, metric::AbstractScoringRule, result::AbstractEvaluationResult)
    keys = propertynames(result)

    model_name = Experiments.get_model_name(exp)

    metric_name = get_metric_method_name(result) * _metric_selection_suffix(metric)

    # Start the unrolling process at the top level
    unrolled_tuples = [unroll("$(metric_name)_$(k)", getproperty(result, k)) for k in keys]

    # Merge all the flattened tuples together
    flat_data = merge(unrolled_tuples...)

    # Attach the model name to the front
    return merge((model = model_name,), flat_data)
end

# Backward-compatible 2-arg form (aggregate naming, no selection suffix).
function to_dataframe_row(exp::ExperimentResults, result::AbstractEvaluationResult)
    keys = propertynames(result)
    model_name = Experiments.get_model_name(exp)
    metric_name = get_metric_method_name(result)
    unrolled_tuples = [unroll("$(metric_name)_$(k)", getproperty(result, k)) for k in keys]
    flat_data = merge(unrolled_tuples...)
    return merge((model = model_name,), flat_data)
end
