# r08_replay_console.jl
#
# ===================================================================
# WHAT THIS IS
# ===================================================================
# A backtest replay of a HISTORICAL match day, driven through the live MatchDay pipeline with
# `as_of` supplied by a scrubber instead of by `now()`. It answers one question the live console
# cannot:
#
#     "What would this model have said, at this minute, against the book that actually existed
#      then -- and what would it have won?"
#
# It runs the same stages in the same order as `r07_serve_console.jl`:
#
#     fixtures -> identity -> lineups -> BOOK -> features -> inference -> gate -> stake_sheet
#
# and replaces only the sources that read a clock or a network. The gates, the instrument rule,
# the rounding, the market set and the portfolio policy are the live ones, unchanged, because a
# replay that relaxed any of them would prove nothing about a Saturday.
#
# ===================================================================
# WHAT THIS IS NOT
# ===================================================================
# * NOT a training run. Every posterior comes from `MD.canonical_fit`, i.e. a completed run in
#   `mcmc_experiments`. Nothing here samples.
# * NOT the live console. It binds **8086**; 8085 is `r07_serve_console.jl` and stays up.
# * NOT allowed near the live ledger. Every write goes to `paper_replay`; `assert_replay_schema`
#   refuses `paper_runbook` at eleven call sites rather than trusting a default argument.
#
# ===================================================================
# FILTRATION CONTRACT
# ===================================================================
# Three leaks are possible in a replay and all three are closed structurally rather than by care:
#
# 1. THE BOOK. `PreloadedBook` holds each runner's ladder sorted by `ts` and reads it with
#    `searchsortedlast(stamps, as_of)`. A tick from after the replayed instant is unreachable.
# 2. THE XI. `PreloadedLineups` filters `scraped_at <= as_of` and has NO historical fallback
#    behind it, so before the scrape lands a player model prices with no lineup and contributes
#    exactly zero.
# 3. THE PLAYER RATINGS. `:player_lineup_ratings_map` is built by the feature extractor over
#    EVERY match in the store, so for a finished fixture it already holds the XI that took the
#    field. `PointInTimeLineupRatings` overwrites it per tick from the visible teamsheet. Without
#    that materialiser the hybrid model would price a T-60m decision off the teamsheet.
#
# The fold is chosen by `MD.select_split`, which identifies it POSITIVELY -- the fold whose next
# observed round is this card -- and steps back from any fold whose target window contains the
# fixtures being priced.
#
# ===================================================================
# USAGE
# ===================================================================
#   julia --project -t 8 current_development/match_day_inference/r08_replay_console.jl
#   open http://localhost:8086          (LAN: http://192.168.1.88:8086)
#
# Environment: `BF_DB_URL` (betdb) and `BF_EXPERIMENTS_DB_URL` or `~/.pgpass` (mcmc_experiments).
# Override the match day without editing this file:
#   R08_DAY=2026-08-15 julia --project -t 8 current_development/match_day_inference/r08_replay_console.jl

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning, LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using DataFrames, Dates
import LibPQ

include(joinpath(@__DIR__, "replay_state.jl"))
include(joinpath(@__DIR__, "replay_server.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
# The default day is 2026-08-08: it is the only Saturday in the archive that carries BOTH a
# 1-minute order book (12:00-16:03, i.e. T-120 to T+123 around a 14:00 kick-off) and a scraped
# provisional XI for 9 of its 10 fixtures. The other three replayable Saturdays -- 08-01, 08-15,
# 08-22 -- have the book but no lineup scrape, so the player pillar contributes zero throughout
# and the XI-drop transition is not observable on them. All four are selectable in the console.
const R08_DAY       = Date(get(ENV, "R08_DAY", "2026-08-08"))
const R08_TIDS      = [56, 57]                 # Scottish League One and League Two
const R08_BANKROLL  = 2_400.0
const R08_ACCOUNT   = "replay_scottish"
const R08_SCHEMA    = REPLAY_SCHEMA            # "paper_replay" -- never paper_runbook
const R08_HOST      = "0.0.0.0"
const R08_PORT      = REPLAY_PORT              # 8086 -- never 8085
const R08_MODEL     = get(ENV, "R08_MODEL", "m00")

println("\n" * "="^78)
println("  MatchDay REPLAY console")
println("  match day : ", R08_DAY)
println("  schema    : ", R08_SCHEMA, "   (live paper_runbook is untouched)")
println("  port      : ", R08_PORT, "        (live console on 8085 is untouched)")
println("="^78 * "\n")

# %%
# ===================================================================
# 3. Data snapshot
# ===================================================================
@info "loading ScottishLower DataStore (uses .cache/ if warm)"
ds = DD.load_datastore_cached(DD.ScottishLower())

conn = MD.paper_connection()

@info "replayable match days"
show(available_matchdays(conn; tournament_ids = R08_TIDS); allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 4. The match day, read once into memory
# ===================================================================
# Identity, the whole day's order book, every lineup scrape, and the final scores. After this
# the scrubber touches no network at all, which is what makes 60x (one simulated minute per wall
# second) a usable speed rather than a queue of round trips.
card = load_replay_card(conn, R08_DAY; tournament_ids = R08_TIDS)

@info "card loaded" day = card.day n_fixtures = length(card.fixtures) kickoff = card.kickoff
@info "identity" resolved = count(v -> v isa MD.Resolved, values(card.identities))
@info "order book" from = card.book_span[1] to = card.book_span[2] n_fixtures_with_book =
    length(card.book.ladders)
@info "lineup scrapes" n = length(card.lineup_drop) drops_T_minus =
    sort([Int(round(Dates.value(card.kickoff - t) / 60_000)) for t in values(card.lineup_drop)])
@info "full-time scores" n = length(card.results)

# %%
# ===================================================================
# 5. Portfolio policy
# ===================================================================
# Identical to `r07_serve_console.jl`. `SlateDrawdown` solves ONE `k` for the whole settlement
# window and `FixedCap` bounds total simultaneous exposure, so the stake vector is only valid as
# a vector -- which is why execution is one transaction and not one per leg.
system = PF.PortfolioSystem(
    PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb()),
    PF.PolicySpec(risk = PF.SlateDrawdown(20.0), cap = PF.FixedCap(0.25),
                  trust = PF.FlatTrust(1.0)))

# %%
# ===================================================================
# 6. Replay state, ledger, and the first model
# ===================================================================
state = ReplayState(ds, conn, card; system = system, bankroll = R08_BANKROLL,
                    account_id = R08_ACCOUNT, schema = R08_SCHEMA, active = R08_MODEL)

account = ensure_replay_account!(state)
@info "replay account" account.account_id balance = account.balance reserved = account.reserved

# Only the default model is loaded now. The others load on first selection, which costs one
# `Features.create_features` each (~10 s for the team-level pillars, ~60 s for the hybrid player
# pillar) and is then cached for the life of the process.
@info "loading canonical fit" model = R08_MODEL
slot = load_slot!(active_slot(state), ds, card)
if slot.status === :ready
    MD.matchday_fit_report(slot.fit)
    @info "fold selection" fold = slot.fold_idx covered = length(slot.covered) refused =
        length(slot.refused)
    isempty(slot.refused) || for (mid, why) in slot.refused
        @warn "fixture not covered by this fold" match_id = mid reason = why
    end
else
    @error "model failed to load" model = R08_MODEL error = slot.error
end

# %%
# ===================================================================
# 7. Price the opening instant
# ===================================================================
seek!(state, T_START)
if state.slate !== nothing
    @info "priced at T-60m" legs = MD.n_legs(state.slate) fixtures =
        MD.n_fixtures(state.slate) total_risk = round(state.slate.total_risk, digits = 2)
end
isempty(state.tick_note) || @warn "tick" note = state.tick_note

# %%
# ===================================================================
# 8. Serve
# ===================================================================
server = ReplayServer(state)
serve_replay(server; host = R08_HOST, port = R08_PORT)

println("\n" * "="^78)
println("  MatchDay REPLAY Console is LIVE")
println("  Local URL : http://localhost:", R08_PORT)
println("  LAN URL   : http://192.168.1.88:", R08_PORT)
println("  Schema    : ", R08_SCHEMA, "   (live 8085 console and paper_runbook untouched)")
println()
println("  VCR      space play/pause · ← → step 1m · x XI drop · e exec · k kickoff · f settle")
println("  API      POST /api/replay/{play,pause,speed,step,jump,seek,set_model,set_matchday,")
println("                             execute,settle,reset}")
println("  Press Ctrl+C to stop the server.")
println("="^78 * "\n")

try
    wait()
catch error
    error isa InterruptException || rethrow()
finally
    stop_replay!(server)
    close(conn)
    println("\nReplay console stopped. The live console on 8085 was never touched.")
end
