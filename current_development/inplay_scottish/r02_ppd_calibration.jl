#=
r02_ppd_calibration.jl — WP3 runner: validate the composed in-play PPD on 56/57.

Gates (plan §WP3):
  §1 t=0 CONSISTENCY: composed prices at kickoff (0-0, no reds) must track the pregame
     model's own prices — quantifies the NHPP net normalisation exp(α)·∫shape dt / 90.
  §2 OUTCOME CALIBRATION per checkpoint: composed OU/1X2 probs at t0 ∈ {0, 30, 60, 80}
     with the ACTUAL score/reds at t0, scored vs realized outcomes (Brier / logloss,
     grouped by checkpoint).
  §3 (r02b, needs clock map) Betfair identifiable-bin comparison on 56 24/25.

Requires: r01 artifacts (out/r01_chain.jls, out/r01_mseqs.jls). Homelab-sized.
=#

using Serialization, Statistics, DataFrames, Random

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(@__DIR__, "l01_nhpp_scottish.jl"))
include(joinpath(@__DIR__, "l02_ppd_compose.jl"))

OUT = joinpath(@__DIR__, "out")
chain = deserialize(joinpath(OUT, "r01_chain.jls"))
mseqs = deserialize(joinpath(OUT, "r01_mseqs.jls"))
config = NHPPXConfig()

const LATENTS_JLS = joinpath(dirname(@__DIR__), "..", "data", "scottish_decay_grid",
                             "latents_hl365_hs2.jls")
latents_df = deserialize(abspath(LATENTS_JLS))
lat = Dict(r.match_id => (collect(r.λ_h), collect(r.λ_a)) for r in eachrow(latents_df))

# thin the eval loop: PPD per (match × checkpoint) is the expensive object
rng = Xoshiro(7)
eval_ms = mseqs[randperm(rng, length(mseqs))[1:min(300, length(mseqs))]]

# ---------------------------------------------------------------------------
# §1 t=0 consistency vs the pregame model itself
# ---------------------------------------------------------------------------

K0_h, K0_a = intensity_kernels(chain, config; gh = 0, ga = 0, t_now = 0.0)
# net normalisation: E[Λ | pg=1] should be ≈ 1 × (goals actually delivered per unit pg)
t0_gate = (K_h_mean = mean(K0_h), K_a_mean = mean(K0_a))

t0_rows = DataFrame(mid = Int[], sel = Symbol[], composed = Float64[], pregame = Float64[])
for ms in eval_ms[1:min(100, length(eval_ms))]
    λh, λa = lat[ms.mid]
    ppd = inplay_ppd(chain, config, λh, λa; gh = 0, ga = 0, t_now = 0.0, rng = Xoshiro(ms.mid))
    # pregame reference: same double-Poisson functional on the raw pregame draws
    Spre = compose_score_matrix(λh, λa, ones(1), ones(1); gh = 0, ga = 0,
                                n_pairs = 2000, rng = Xoshiro(ms.mid))
    for m in default_markets(), (sel, v) in Pred.compute_market_probs(Spre, m)
        push!(t0_rows, (ms.mid, sel, mean(ppd[sel]), mean(v)))
    end
end
t0_summary = combine(groupby(t0_rows, :sel),
    [:composed, :pregame] => ((c, p) -> mean(c .- p)) => :mean_diff,
    [:composed, :pregame] => ((c, p) -> maximum(abs.(c .- p))) => :max_abs_diff)

# ---------------------------------------------------------------------------
# §2 outcome calibration at checkpoints
# ---------------------------------------------------------------------------

function state_at(ms, t0)
    (gh = count(g ->  g.home && g.t < t0, ms.goals),
     ga = count(g -> !g.home && g.t < t0, ms.goals),
     rh = count(c ->  c.home && c.t < t0, ms.reds),
     ra = count(c -> !c.home && c.t < t0, ms.reds))
end

cal_rows = DataFrame(mid = Int[], t0 = Float64[], sel = Symbol[],
                     p = Float64[], y = Int[])
for ms in eval_ms, t0 in (0.0, 30.0, 60.0, 80.0)
    st = state_at(ms, t0)
    λh, λa = lat[ms.mid]
    ppd = inplay_ppd(chain, config, λh, λa; gh = st.gh, ga = st.ga,
                     reds_h = st.rh, reds_a = st.ra, t_now = t0,
                     n_pairs = 1000, rng = Xoshiro(ms.mid + Int(t0)))
    fh = count(g -> g.home, ms.goals); fa = count(g -> !g.home, ms.goals)
    tot = fh + fa
    truth = Dict(
        :home => Int(fh > fa), :draw => Int(fh == fa), :away => Int(fh < fa),
        :btts_yes => Int(fh > 0 && fa > 0), :btts_no => Int(!(fh > 0 && fa > 0)))
    for k in 0:5
        truth[Symbol("over_$(k)5")] = Int(tot > k); truth[Symbol("under_$(k)5")] = Int(tot < k + 1)
    end
    for (sel, v) in ppd
        haskey(truth, sel) || continue
        push!(cal_rows, (ms.mid, t0, sel, mean(v), truth[sel]))
    end
end

fam(sel) = startswith(String(sel), "over") || startswith(String(sel), "under") ? :ou :
           (sel in (:home, :draw, :away) ? :x12 : :btts)
cal_rows.family = fam.(cal_rows.sel)
calibration = combine(groupby(cal_rows, [:t0, :family]),
    nrow => :n,
    [:p, :y] => ((p, y) -> mean((p .- y) .^ 2)) => :brier,
    [:p, :y] => ((p, y) -> -mean(y .* log.(clamp.(p, 1e-9, 1)) .+
                                 (1 .- y) .* log.(clamp.(1 .- p, 1e-9, 1)))) => :logloss,
    [:p, :y] => ((p, y) -> mean(p .- y)) => :bias)

serialize(joinpath(OUT, "r02_cal_rows.jls"), cal_rows)
R02 = (t0_gate = t0_gate, t0_summary = t0_summary, calibration = calibration)
@info "r02 done" t0_gate
