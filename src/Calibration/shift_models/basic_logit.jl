# src/Calibration/shift_models/basic_logit.jl
#
# DEPRECATED. The selection-level logit shift, superseded by `GenerativeRateCalibrator`.
#
# WHY IT IS WRONG, PRECISELY. This model fits one GLM offset per SELECTION and applies it
# to that selection's scalar probability. Applied to `over_25` and `under_25` with the two
# offsets those two GLMs happen to learn,
#
#     logistic(logit(p_over) + c_over) + logistic(logit(p_under) + c_under)  !=  1
#
# and the 1X2 triple drifts off 1 the same way. There is no scoreline distribution behind
# the shifted board at all, so "Over 2.5", "BTTS yes" and "Home" become three unrelated
# claims that a Kelly allocator is then invited to hold simultaneously. Coherence is not
# merely unchecked here; it is unrepresentable.
#
# It is RETAINED, not deleted, because a legacy script that stops running is a worse
# outcome than one that prints a line. The warning fires once per session per model type.

const _L2_DEPRECATION_WARNED = Ref(false)

"""
    _warn_selection_level_deprecated(what)

Warn once per session that a selection-level shift model is being used.

`maxlog = 1` on the `@warn` itself would be per-call-site; this is per-session across all
of them, which is what "once" means to someone reading a log.
"""
function _warn_selection_level_deprecated(what::AbstractString)
    _L2_DEPRECATION_WARNED[] && return nothing
    _L2_DEPRECATION_WARNED[] = true
    @warn(
        "$what is deprecated; use GenerativeRateCalibrator for coherent derivative " *
        "pricing. A selection-level logit shift moves each market's probability " *
        "independently, so P(over 2.5) + P(under 2.5) no longer sums to 1 and the " *
        "shifted board is not a scoreline distribution. GenerativeRateCalibrator shifts " *
        "the generative intensity instead, so every derivative price is read off one " *
        "score tensor and cannot disagree. See docs/architecture/rfc_layer2_calibration_v2.md.")
    return nothing
end

"""
    BasicLogitShift()

**DEPRECATED — use [`GenerativeRateCalibrator`](@ref).**

A single logit offset per selection, fitted by GLM against the realised outcome. See the
header of this file for the coherence failure that motivates the replacement.
"""
struct BasicLogitShift <: AbstractLayerTwoModel
    # No hyperparameters needed for a pure shift
    function BasicLogitShift()
        _warn_selection_level_deprecated("BasicLogitShift")
        return new()
    end
end

struct FittedLogitShift
    c_shift::Float64
    model::StatsModels.TableRegressionModel 
    prob_col::Symbol # NEW: The model remembers what column it was trained on!
end

function fit_calibrator(model::BasicLogitShift, data::DataFrame, config::CalibrationConfig)
    dropmissing!(data, :is_winner)
    eps = 1e-6
    
    # Extract the dynamic column specified in the config
    prob_data = data[!, config.prob_col]
    
    df_fit = DataFrame(
        actual = Float64.(data.is_winner),
        logit_prob = logit.(clamp.(prob_data, eps, 1.0 - eps))
    )
    
    glm_model = glm(@formula(actual ~ 1), df_fit, Binomial(), LogitLink(), offset=df_fit.logit_prob)
    c_shift = coef(glm_model)[1]
    
    # Return the fitted model AND the column name it expects
    return FittedLogitShift(c_shift, glm_model, config.prob_col)
end

"""
    apply_calibration(fitted_model::FittedLogitShift, new_data::DataFrame)
"""
function apply_calibration(fitted_model::FittedLogitShift, new_data::DataFrame)
    eps = 1e-6
    c = fitted_model.c_shift 
    
    # Dynamically grab the correct column based on what the model was trained on
    prob_data = new_data[!, fitted_model.prob_col]
    
    # 1. Shift the scalar probabilities
    shifted_scalars = logistic.(logit.(clamp.(prob_data, eps, 1.0 - eps)) .+ c)
    
    # 2. Shift the MCMC distributions
    shifted_dists = map(new_data.distribution) do dist
        logistic.(logit.(clamp.(dist, eps, 1.0 - eps)) .+ c)
    end
    
    return shifted_scalars, shifted_dists
end
