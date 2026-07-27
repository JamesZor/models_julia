#=
r01b_hier_lead.jl — do teams react DIFFERENTLY to leading/trailing? (user hypothesis)

Non-centred team-specific slopes on `leading` / `trailing`:
  logλ += (γ_ld + δ_lead[team])·leading + (γ_tr + δ_trail[team])·trailing,
  δ_·[team] = z·σ,  σ ~ trunc-Normal(0, 0.3).
Self-regularizing: if teams do NOT differ, σ collapses toward 0 (the hierarchical-σ
null pattern). Interpretation guide:
  - σ_ld ≈ 0 & γ_ld still > 0  → supports the FRAILTY-artifact reading of r01's flip,
    not team character.
  - σ_ld credibly > 0           → team heterogeneity is real; Stage B held-out race
    decides whether it helps prediction (Ireland precedent: team hierarchies HURT OOS
    at 253 matches; we have 715).

Stage A: full-data fit + σ diagnostics + team ladder.
Stage B (auto, only if σ_ld q05 > 0.03): 75/25 held-out race base vs hier,
  plug-in posterior-mean params, per-row Poisson loglik, match-clustered SE.
=#

using Serialization, Statistics, DataFrames, Random

const BF = BayesianFootball
include(joinpath(@__DIR__, "l01_nhpp_scottish.jl"))

OUT = joinpath(@__DIR__, "out")
mseqs  = deserialize(joinpath(OUT, "r01_mseqs.jls"))
config_base = NHPPXConfig()
config_hier = NHPPXConfig(hier_lead = true, hier_trail = true)
slices = build_slices(mseqs; Δt = config_base.Δt, Tend = config_base.Tend)
tidx_full, tnames = team_indexer(slices)

# ---------------------------------------------------------------------------
# Stage A: full fit
# ---------------------------------------------------------------------------

chain_h = Samplers.run_sampler(make_nhppx_model(slices, config_hier),
    Samplers.NUTSConfig(n_samples = 600, n_chains = 4, n_warmup = 300,
                        max_depth = 8, show_progress = false))

_post(v) = (mean = mean(v), q05 = quantile(v, 0.05), q95 = quantile(v, 0.95))
σ_ld_post  = _post(_cv(chain_h, :σ_ld))
σ_trl_post = _post(_cv(chain_h, :σ_trl))
γ_ld_post  = _post(_cv(chain_h, :γ_ld))
γ_tr_post  = _post(_cv(chain_h, :γ_tr))

nT = length(tnames)
δ_lead = vec(mean(_cm(chain_h, :z_ld, nT) .* _cv(chain_h, :σ_ld); dims = 1))
lead_slices = combine(groupby(subset(slices, :leading => ByRow(==(1.0))), :team), nrow => :n)
team_ladder = leftjoin(DataFrame(team = tnames, δ_lead = δ_lead), lead_slices, on = :team)
sort!(team_ladder, :δ_lead)

# ---------------------------------------------------------------------------
# Stage B: held-out race (only if lead heterogeneity is credibly non-null)
# ---------------------------------------------------------------------------

"Plug-in posterior-mean per-row Poisson loglik on a slice frame."
function rowloglik(chain, c::NHPPXConfig, df, tidx)
    m(v) = mean(v)
    logλ = m(_cv(chain, :α)) .+ df.log_pg .+ m(_cv(chain, :β)) .* df.z .+
           m(_cv(chain, :γ_tr)) .* df.trailing .+ m(_cv(chain, :γ_ld)) .* df.leading .+
           m(_cv(chain, :γ_man)) .* df.man_adv
    if _has(chain, "z_time")
        δt = vec(mean(_cm(chain, :z_time, Int(cld(c.Tend, c.Δt))) .* _cv(chain, :σ_time); dims = 1))
        logλ = logλ .+ δt[df.time_idx]
    end
    if _has(chain, "z_ld")
        δl = vec(mean(_cm(chain, :z_ld, length(tnames)) .* _cv(chain, :σ_ld); dims = 1))
        logλ = logλ .+ δl[tidx] .* df.leading
    end
    if _has(chain, "z_trl")
        δr = vec(mean(_cm(chain, :z_trl, length(tnames)) .* _cv(chain, :σ_trl); dims = 1))
        logλ = logλ .+ δr[tidx] .* df.trailing
    end
    μ = exp.(clamp.(logλ .+ df.off, -20.0, 20.0)) .+ 1e-6
    logpdf.(Poisson.(μ), df.y)
end

race = nothing
if σ_ld_post.q05 > 0.03
    rng = Xoshiro(11)
    mids = shuffle(rng, unique(slices.match_id))
    test_set = Set(mids[1:round(Int, 0.25 * length(mids))])
    is_te = [m in test_set for m in slices.match_id]
    trn = slices[.!is_te, :]; tst = slices[is_te, :]
    tidx_te = tidx_full[is_te]
    scfg = Samplers.NUTSConfig(n_samples = 400, n_chains = 4, n_warmup = 250,
                               max_depth = 8, show_progress = false)
    ch_b = Samplers.run_sampler(make_nhppx_model(trn, config_base), scfg)
    ch_hh = Samplers.run_sampler(make_nhppx_model(trn, config_hier), scfg)
    # NOTE: hier fit's team indices come from trn's own indexer; align by refit lookup
    d = rowloglik(ch_hh, config_hier, tst, tidx_te) .- rowloglik(ch_b, config_base, tst, tidx_te)
    per_match = combine(groupby(DataFrame(mid = tst.match_id, d = d), :mid), :d => sum => :d)
    race = (mean_row = mean(d), match_mean = mean(per_match.d),
            match_se = std(per_match.d) / sqrt(nrow(per_match)),
            t = mean(per_match.d) / (std(per_match.d) / sqrt(nrow(per_match))))
end

serialize(joinpath(OUT, "r01b_chain_hier.jls"), chain_h)
R01B = (σ_ld = σ_ld_post, σ_trl = σ_trl_post, γ_ld = γ_ld_post, γ_tr = γ_tr_post,
        ladder = team_ladder, race = race,
        rhat_max = maximum(DataFrame(MCMCChains.summarystats(chain_h)).rhat))
@info "r01b done" σ_ld_post σ_trl_post
