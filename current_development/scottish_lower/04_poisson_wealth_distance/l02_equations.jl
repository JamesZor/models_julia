# TP04 independent log-space Poisson referee.
using SpecialFunctions
function tp04_equation_data(model, fs)
    d = fs.data
    n = length(d[:flat_home_ids])
    (; home=Vector{Int}(d[:flat_home_ids]), away=Vector{Int}(d[:flat_away_ids]),
       yh=Vector{Int}(d[:flat_home_goals]), ya=Vector{Int}(d[:flat_away_goals]),
       weights=_slfp_weight(Float64.(d[:dates]), model.dynamics_config.days_half_life),
       wealth=Float64.(get(d, :flat_delta_wealth, zeros(Float64,n))),
       distance=Float64.(get(d, :flat_distance, zeros(Float64,n))))
end
tp04_logjoint(model, params, data) = slfp_logjoint(model, params, data)
