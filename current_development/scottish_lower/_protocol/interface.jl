# Adapter hooks have no defaults: every model explicitly owns its translation.
for hook in (:sl_model, :sl_model_name, :sl_required_features, :sl_assert_model_contract, :sl_build_turing_model, :sl_params_from_varinfo, :sl_equation_data, :sl_equation_logjoint, :sl_sampled_sites, :sl_parameter_row, :sl_synthetic_n_teams, :sl_synthetic_draws, :sl_synthetic_fixtures, :sl_reference_extract, :sl_extract_parameters, :sl_extract_params, :sl_compute_score_matrix, :sl_reference_grid, :sl_marginal_cdf_bounds, :sl_marginal_logpdf, :sl_capabilities, :sl_referee_eval)
 @eval ($hook)(adapter::AbstractSLModelAdapter, args...; kwargs...) = _sl_missing(adapter, $(QuoteNode(hook)))
end
"Validate explicit grouped-site and expanded-column posterior schema."
function sl_posterior_schema(adapter)
 caps=sl_capabilities(adapter); hasproperty(caps,:posterior_schema) || error("sl_capabilities must declare posterior_schema")
 s=caps.posterior_schema
 all(hasproperty(s,k) for k in (:varinfo_sites,:chain_columns,:parameter_count)) || error("posterior_schema needs varinfo_sites, chain_columns, parameter_count")
 s
end
function sl_adapter_check(adapter::AbstractSLModelAdapter, stage::Symbol, args...)
 checks=sl_referee_eval(adapter,stage,args...); checks isa AbstractVector || error("sl_referee_eval must return Vector")
 all(x->hasproperty(x,:name)&&hasproperty(x,:pass)&&hasproperty(x,:detail),checks) || error("invalid referee result")
 checks
end
