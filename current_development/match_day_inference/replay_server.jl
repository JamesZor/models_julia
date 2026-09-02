# current_development/match_day_inference/replay_server.jl
#
# The replay console's read model and HTTP/WebSocket surface.
#
# WHY THIS IS NOT `MD.ConsoleState`. The live console has four routes and one intent, because at
# T-12 on a Saturday there is exactly one decision to make and every extra control is a way to
# make it wrong. A replay is the opposite instrument: it exists to be scrubbed, re-modelled and
# re-run, so it carries eleven routes and a transport. Bolting those onto `console/server.jl`
# would put a `POST /api/replay/seek` on the process that can commit real paper money, and the
# only reason that is safe TODAY is that nothing in this file is loaded by it.
#
#   8085  src/MatchDay/console/server.jl  -> paper_runbook   -- live, untouched
#   8086  this file                       -> paper_replay    -- replay, this process
#
# WHY THE PAYLOAD EXTENDS RATHER THAN REPLACES. `slate_snapshot` is the live console's read model
# and the card/leg shape it emits is the thing the operator has learned to read: two bars on one
# 0-1 scale, the overhang IS the edge. The replay adds a `replay` block and a `settlement` block
# and changes NOTHING inside `cards`, so a card means the same thing on both consoles. If it did
# not, a habit formed on the replay would misread the live one.

import HTTP
import JSON3

# ===================================================================
# 1. The read model
# ===================================================================

"""
    replay_payload(st) -> NamedTuple

The whole console state, ready for `JSON3.write`: one object, one frame, one `x-for`.

Four blocks, answering four questions in the order an operator asks them:

    replay      -- where in the match day are we, and with which model?
    account     -- can we afford anything?
    batch       -- is THIS VECTOR safe to commit?
    cards       -- which legs, and how good are they?
    settlement  -- and what actually happened?

`batch` and `cards` come from `MD.slate_snapshot` unchanged. `stale` in the `replay` block is the
one field a replay needs that a live console does not: post-kickoff the gates legitimately refuse
the card, the grid freezes at the last priced minute, and a frozen grid that did not say so would
be indistinguishable from a live one.
"""
function replay_payload(st::ReplayState)
    return lock(st.lock) do
        slate = st.slate
        account = _account_or_placeholder(st)
        base = slate === nothing ?
            (batch = _empty_batch(st), cards = NamedTuple[], blocked = NamedTuple[]) :
            (batch = MD.batch_payload(slate, _batch_status(st, slate)),
             cards = MD.card_payload(slate),
             blocked = [(match_id = c.fixture.m_id,
                         fixture = c.fixture.home * " v " * c.fixture.away,
                         kickoff = string(c.fixture.kickoff),
                         reasons = [string(k) * ": " * v for (k, v) in c.readiness.reasons])
                        for c in slate.blocked if c.readiness isa MD.Blocked])

        return (
            at = string(Dates.now()),
            replay = _replay_block(st),
            account = MD.account_payload(account),
            batch = base.batch,
            cards = base.cards,
            blocked = base.blocked,
            settlement = st.settlement,
        )
    end
end

function _replay_block(st::ReplayState)
    card, clock = st.card, st.clock
    slot = active_slot(st)
    return (
        day            = string(card.day),
        kickoff        = string(card.kickoff),
        t              = clock.t,
        as_of          = string(as_of_at(card, clock.t)),
        playing        = clock.playing,
        speed          = clock.speed,
        speeds         = collect(SPEEDS),
        t_start        = T_START,
        t_end          = T_END,
        markers        = (lineups = T_LINEUP, exec = T_EXEC, kickoff = T_KICKOFF),
        priced_t       = st.slate_t,
        stale          = st.slate !== nothing && st.slate_t != clock.t,
        # After the whistle the console is comparing a PRE-GAME posterior with an IN-PLAY book,
        # and the book has seen goals the posterior has not. On a 1-0 the edges reach four
        # figures and they are not signals -- they are the measurement of how far a pre-game
        # model is from an in-play one. The flag exists so the page can say so, because a
        # +1179% card that carries no warning will eventually be believed by someone.
        in_play        = clock.t >= T_KICKOFF,
        note           = st.tick_note,
        error          = st.tick_error,
        reprice_ms     = round(1000 * st.reprice_seconds, digits = 1),
        tick_seq       = st.tick_seq,
        schema         = st.schema,
        account_id     = st.account_id,
        n_executed     = length(st.executed),
        active_model   = slot.key,
        models         = [(key = m.key, label = m.label, run_name = m.run_name,
                           experiment = m.experiment, status = String(m.status),
                           error = m.error, fold_idx = m.fold_idx,
                           n_covered = length(m.covered), n_refused = length(m.refused),
                           refused = [(match_id = p.first, reason = p.second)
                                      for p in m.refused],
                           load_seconds = round(m.load_seconds, digits = 1),
                           n_latent_states = length(m.latents),
                           active = m.key == slot.key)
                          for m in st.models],
        fixtures       = [(match_id = f.m_id, fixture = f.home * " v " * f.away,
                           lineup_drop_min = haskey(card.lineup_drop, f.m_id) ?
                               Int(round(Dates.value(f.kickoff -
                                   card.lineup_drop[f.m_id]) / 60_000)) : nothing)
                          for f in card.fixtures],
        book_from      = card.book_span[1] === nothing ? nothing : string(card.book_span[1]),
        book_to        = card.book_span[2] === nothing ? nothing : string(card.book_span[2]),
        n_lineups      = length(card.lineup_drop),
        n_fixtures     = length(card.fixtures),
    )
end

"""
Batch status for the header, looked up by `(account, window, as_of)` -- the ledger's own
idempotency key -- rather than by the id of the last slate this session executed.

The distinction matters and is not pedantic. Every tick prices a NEW `PricedSlate` with a fresh
`slate_id`, so "the last thing we executed" and "the thing on screen" are different objects the
moment the operator steps a minute after executing. Keying on `as_of` means the header says
EXECUTED at the minute that was executed and PRICED at every other, which is the question the
operator is actually asking of it.
"""
function _batch_status(st::ReplayState, slate::MD.PricedSlate)
    isempty(st.executed) && return MD.PRICED
    try
        df = DataFrame(LibPQ.execute(st.conn,
            """SELECT batch_status FROM $(st.schema).paper_slates
               WHERE account_id = \$1 AND slate_window = \$2 AND as_of = \$3;""",
            (st.account_id, slate.window, slate.as_of)))
        nrow(df) == 1 || return MD.PRICED
        return MD._parse_batch(String(first(df).batch_status))
    catch
        # The schema may not be migrated yet on the very first snapshot. An unknown status is
        # PRICED, which is the state that permits an Execute -- and `execute_slate_batch!` is the
        # thing that decides, not this.
        return MD.PRICED
    end
end

function _account_or_placeholder(st::ReplayState)
    try
        return MD.account_row(st.conn, st.account_id; schema = st.schema)
    catch
        return MD.PaperAccount(account_id = st.account_id, opening_balance = st.bankroll,
                               balance = st.bankroll, max_slate_exposure = 0.25)
    end
end

_empty_batch(st::ReplayState) = (
    slate_id = "", window = string(st.card.day),
    as_of = string(as_of_at(st.card, st.clock.t)), status = "PRICED",
    bankroll = round(st.bankroll, digits = 2), total_risk = 0.0, slate_exposure = 0.0,
    exposure_cap = _policy_cap(st.system), exposure_pct = 0.0,
    cap_pct = round(100 * _policy_cap(st.system), digits = 2), k_risk = 0.0,
    risk_lambda = _policy_lambda(st.system), capped = false, n_fixtures = 0, n_legs = 0,
    n_blocked = 0, fold_idx = 0, warning = "", n_low_confidence = 0)

# ===================================================================
# 2. The server
# ===================================================================

const REPLAY_HTML_PATH = joinpath(@__DIR__, "replay_console.html")

"The single page, read from disk each call so an edit is visible on refresh."
replay_html() = read(REPLAY_HTML_PATH, String)

"""
    ReplayServer

The transport. Holds the sockets and the push task; every decision lives in `ReplayState`.

`interval` is 0.5 s rather than the live console's 1.0. The underlying book is still 1-minute
data, so this is not more information -- but at 60x the clock advances once a second and a 1 Hz
push aliases with it, making the scrubber appear to stutter and skip minutes that were in fact
priced.
"""
mutable struct ReplayServer
    state::ReplayState
    clients::Vector{Any}
    lock::ReentrantLock
    interval::Float64
    server::Any
    pusher::Any
    running::Bool
end

ReplayServer(state::ReplayState; interval::Real = 0.5) =
    ReplayServer(state, Any[], ReentrantLock(), Float64(interval), nothing, nothing, false)

_json(x; status::Int = 200) =
    HTTP.Response(status, ["Content-Type" => "application/json; charset=utf-8"],
                  body = JSON3.write(x))

const REPLAY_ROUTES = [
    "GET  /", "GET  /api/snapshot", "GET  /api/health", "GET  /api/replay/matchdays",
    "POST /api/replay/play", "POST /api/replay/pause", "POST /api/replay/speed",
    "POST /api/replay/step", "POST /api/replay/jump", "POST /api/replay/seek",
    "POST /api/replay/set_model", "POST /api/replay/set_matchday",
    "POST /api/replay/execute", "POST /api/replay/settle", "POST /api/replay/reset",
]

"""
    route_replay(srv, req) -> HTTP.Response

The whole API. Every route that changes anything is a POST, and every one of them returns the
same shape: `(ok, note)` or `(ok = false, error)`.

The browser is not in the trust path here either. The page POSTs an INTENT -- "advance a minute",
"switch to m12", "execute" -- and this process validates it and performs the same transaction a
script would. `POST /api/replay/execute` reaches `execute_slate_batch!`, which takes the account
lock and refuses an over-cap vector whatever the page believes.
"""
function route_replay(srv::ReplayServer, req::HTTP.Request)
    uri = HTTP.URI(req.target)
    target, method = uri.path, req.method
    st = srv.state

    if method == "GET" && (target == "/" || target == "/index.html")
        return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"],
                             body = replay_html())
    elseif method == "GET" && target == "/api/snapshot"
        return _json(replay_payload(st))
    elseif method == "GET" && target == "/api/health"
        n = lock(srv.lock) do; length(srv.clients); end
        return _json((ok = true, clients = n, interval = srv.interval,
                      port = REPLAY_PORT, schema = st.schema, day = string(st.card.day),
                      t = st.clock.t, at = string(Dates.now())))
    elseif method == "GET" && target == "/api/replay/matchdays"
        return _json(_intent(() -> (ok = true,
            matchdays = [NamedTuple(r) for r in eachrow(available_matchdays(st.conn))])))
    end

    args = _body_args(req, uri)

    if method == "POST" && target == "/api/replay/play"
        return _json(_intent(() -> (play!(st); (ok = true, note = "playing at $(st.clock.speed)x",
                                                t = st.clock.t))))
    elseif method == "POST" && target == "/api/replay/pause"
        return _json(_intent(() -> (pause!(st); (ok = true, note = "paused", t = st.clock.t))))
    elseif method == "POST" && target == "/api/replay/speed"
        return _json(_intent(() -> begin
            s = set_speed!(st, _num(args, "speed", 60.0))
            (ok = true, note = "speed $(s)x", speed = s)
        end))
    elseif method == "POST" && target == "/api/replay/step"
        return _json(_intent(() -> begin
            t = step!(st, Int(round(_num(args, "minutes", 1.0))))
            (ok = true, note = "T$(_signed(t))m", t = t)
        end))
    elseif method == "POST" && target == "/api/replay/jump"
        return _json(_intent(() -> begin
            t = jump!(st, _str(args, "target", "exec"))
            (ok = true, note = "jumped to T$(_signed(t))m", t = t)
        end))
    elseif method == "POST" && target == "/api/replay/seek"
        return _json(_intent(() -> begin
            t = seek!(st, Int(round(_num(args, "t", Float64(T_START)))))
            (ok = true, note = "T$(_signed(t))m", t = t)
        end))
    elseif method == "POST" && target == "/api/replay/set_model"
        return _json(_intent(() -> begin
            slot = set_model!(st, _str(args, "model", st.active))
            slot.status === :ready ?
                (ok = true, note = "active model $(slot.key) (fold $(slot.fold_idx), " *
                                   "$(length(slot.covered)) fixtures covered)",
                 model = slot.key) :
                (ok = false, error = "model $(slot.key) failed to load: $(slot.error)")
        end))
    elseif method == "POST" && target == "/api/replay/set_matchday"
        return _json(_intent(() -> begin
            day = Date(_str(args, "day", string(st.card.day)))
            card = set_matchday!(st, day)
            (ok = true, note = "loaded $(day): $(length(card.fixtures)) fixtures, " *
                               "$(length(card.lineup_drop)) with a scraped XI",
             day = string(day))
        end))
    elseif method == "POST" && target == "/api/replay/execute"
        return _json(_intent(() ->
            execute!(st; allow_in_play = _num(args, "allow_in_play", 0.0) != 0.0 ||
                                         _str(args, "allow_in_play", "") == "true")))
    elseif method == "POST" && target == "/api/replay/settle"
        return _json(_intent(() -> settle!(st)))
    elseif method == "POST" && target == "/api/replay/reset"
        return _json(_intent(() -> (reset_replay_ledger!(st);
                                    (ok = true, note = "ledger reset to £$(st.bankroll)"))))
    end

    return _json((ok = false, error = "no route $method $target", routes = REPLAY_ROUTES);
                 status = 404)
end

"""
Run one intent, turning any exception into a reported refusal.

A throw here closes the connection and leaves the operator with a spinner and no reason. The
exception text is returned instead: this console is on a LAN and its operator is the person who
would read the stack trace anyway.
"""
function _intent(f::Function)
    try
        out = f()
        return out isa NamedTuple ? out : (ok = true, result = string(out))
    catch e
        return (ok = false, error = sprint(showerror, e))
    end
end

"Merge a JSON body with the query string, so every control is also reachable with `curl -X POST`."
function _body_args(req::HTTP.Request, uri::HTTP.URI)
    args = Dict{String,Any}()
    for (k, v) in HTTP.queryparams(uri)
        args[k] = v
    end
    body = String(req.body)
    isempty(body) && return args
    try
        for (k, v) in pairs(JSON3.read(body))
            args[String(k)] = v
        end
    catch
        # A malformed body is a client bug, not a server one. The query string still applies and
        # the defaults below cover the rest, so the control still does something explainable.
    end
    return args
end

function _num(args::Dict{String,Any}, key::AbstractString, default::Float64)
    v = get(args, String(key), nothing)
    v === nothing && return default
    v isa Number && return Float64(v)
    p = tryparse(Float64, string(v))
    return p === nothing ? default : p
end

function _str(args::Dict{String,Any}, key::AbstractString, default::AbstractString)
    v = get(args, String(key), nothing)
    return v === nothing ? String(default) : String(string(v))
end

# ===================================================================
# 3. Push, and lifecycle
# ===================================================================

"Send the current payload to every live client, dropping the ones that have gone."
function push_replay!(srv::ReplayServer)
    payload = try
        JSON3.write(replay_payload(srv.state))
    catch e
        JSON3.write((error = sprint(showerror, e),))
    end
    dead = Any[]
    lock(srv.lock) do
        for ws in srv.clients
            try
                HTTP.WebSockets.send(ws, payload)
            catch
                push!(dead, ws)
            end
        end
        isempty(dead) || filter!(w -> !(w in dead), srv.clients)
    end
    return length(dead)
end

function _ws_loop(srv::ReplayServer, ws)
    lock(srv.lock) do; push!(srv.clients, ws); end
    try
        HTTP.WebSockets.send(ws, JSON3.write(replay_payload(srv.state)))
        for _ in ws
        end
    catch
        # a client closing mid-read is normal, not an error worth logging every reconnect
    finally
        lock(srv.lock) do; filter!(w -> w !== ws, srv.clients); end
    end
end

function _stream_handler(srv::ReplayServer)
    return function (http::HTTP.Streams.Stream)
        if HTTP.WebSockets.isupgrade(http.message)
            HTTP.WebSockets.upgrade(ws -> _ws_loop(srv, ws), http)
            return nothing
        end
        req = http.message
        req.body = read(http)
        resp = try
            route_replay(srv, req)
        catch e
            _json((ok = false, error = sprint(showerror, e)); status = 500)
        end
        HTTP.setstatus(http, resp.status)
        for (k, v) in resp.headers
            HTTP.setheader(http, k => v)
        end
        HTTP.startwrite(http)
        write(http, resp.body)
        return nothing
    end
end

"""
    serve_replay(srv; host, port) -> ReplayServer

Start the replay console. Non-blocking: returns as soon as the socket is listening.

Refuses to bind 8085. That port is the live console and a second process on it would either fail
noisily or -- worse, if the live one had just died -- succeed, and put a replay's Execute button
where an operator expects the real one.
"""
function serve_replay(srv::ReplayServer; host = "0.0.0.0", port::Integer = REPLAY_PORT,
                      push::Bool = true, verbose = -1)
    port == 8085 && error(
        "serve_replay: 8085 is the LIVE operator console (paper_runbook). The replay console " *
        "binds $REPLAY_PORT so the two can run side by side and never be confused for each other.")
    srv.running && error("serve_replay: already serving. Call stop_replay! first.")
    srv.state.running = true
    srv.server = HTTP.serve!(_stream_handler(srv), host, port; stream = true, verbose = verbose)
    srv.running = true
    if push
        srv.pusher = Threads.@spawn begin
            while srv.running
                try
                    push_replay!(srv)
                catch
                end
                sleep(srv.interval)
            end
        end
    end
    return srv
end

"Stop the console, halt the clock, and release the port. Idempotent."
function stop_replay!(srv::ReplayServer)
    srv.running = false
    srv.state.running = false
    try pause!(srv.state) catch end
    if srv.server !== nothing
        try close(srv.server) catch end
        srv.server = nothing
    end
    lock(srv.lock) do
        for ws in srv.clients
            try close(ws) catch end
        end
        empty!(srv.clients)
    end
    srv.pusher = nothing
    return srv
end
