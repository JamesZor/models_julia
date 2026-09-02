# src/MatchDay/console/server.jl
#
# The operator console: HTTP.jl + WebSockets serving one static page.
#
# WHY A PAGE AND NOT A TUI. A Textual TUI already exists for the COLLECTOR
# (`betdb/tui/screens/live/*`) and it is better at what it does than a rewrite would be -- the
# `betdb_matchday_status` verdict ladder in particular. What it cannot do is the trading job: 21
# cards visible AT ONCE with a comparable model-vs-market read on each, because the operator has
# to judge the vector as a vector before committing it in one action, and a 50-row terminal
# scrolls exactly when everything needs to be on screen together.
#
# WHY IT STAYS SMALL. No build step, no bundler, no component library, no client-side router, no
# state manager. The server sends one JSON object; the client has one array and one sort
# comparator. HTMX is deliberately NOT used: it swaps server-rendered fragments, which is right
# for forms and wrong for a 1 Hz stream of a hundred numbers -- it would re-render DOM subtrees
# every tick and lose focus, scroll and selection. Alpine mutates a JS array in place and lets the
# browser diff the text nodes.
#
# WHY THE BROWSER IS NOT IN THE TRUST PATH. The page never writes to a venue or to the account. It
# POSTs an INTENT; the Julia process validates it and performs the same transaction
# `execute_slate_batch!` performs from a script. A compromised or merely confused tab can ask for
# something; it cannot do it.

export ConsoleState, serve_console, stop_console!, console_html, push_snapshot!

const CONSOLE_HTML_PATH = joinpath(@__DIR__, "console.html")

"""
    ConsoleState

Everything the server needs, and nothing it does not.

* `snapshot` -- a zero-argument function returning the payload. A function rather than a stored
  object so the page can never show a snapshot the process has moved past, and so the tests can
  drive it without a database.
* `on_execute`, `on_kill` -- intent handlers. Both default to a refusal that says what is
  missing, because a console whose Execute button silently does nothing is worse than one that
  has no button.
* `clients` -- live WebSockets, guarded by `lock`. Guarded rather than lock-free because the push
  task and the accept task both touch it, and a `Vector` is not thread-safe.
* `interval` -- seconds between pushes. 1.0: the book underneath is 1-minute data, so anything
  faster is animation rather than information.
"""
mutable struct ConsoleState
    snapshot::Function
    on_execute::Function
    on_kill::Function
    clients::Vector{Any}
    lock::ReentrantLock
    interval::Float64
    server::Any
    pusher::Any
    running::Bool
end

_no_executor(_args...) = (ok = false,
    error = "no executor is wired into this console. Construct ConsoleState with " *
            "`on_execute = () -> execute_slate_batch!(conn, account_id, slate_id)` -- the " *
            "browser must never reach the ledger directly.")

ConsoleState(snapshot::Function; on_execute::Function = _no_executor,
             on_kill::Function = _no_executor, interval::Real = 1.0) =
    ConsoleState(snapshot, on_execute, on_kill, Any[], ReentrantLock(),
                 Float64(interval), nothing, nothing, false)

"The single page, read from disk each call so an edit is visible on refresh."
console_html() = read(CONSOLE_HTML_PATH, String)

# ===================================================================
# Routing
# ===================================================================

_json(x; status::Int = 200) =
    HTTP.Response(status, ["Content-Type" => "application/json; charset=utf-8"],
                  body = JSON3.write(x))

"""
    route_request(state, req) -> HTTP.Response

The whole API. Four endpoints, and every one that changes anything is a POST.

* `GET  /`              the page
* `GET  /api/snapshot`  the payload, for a client with no WebSocket
* `GET  /api/health`    liveness plus the client count, for a supervisor
* `POST /api/execute`   commit the slate -- one intent, the atom of §the reservation
* `POST /api/kill`      abort the slate

An unknown path is a 404 with the route list, not an empty body: an operator debugging a console
at T-12 should not have to read the source to find out what it serves.
"""
function route_request(state::ConsoleState, req::HTTP.Request)
    target = HTTP.URI(req.target).path
    method = req.method

    if method == "GET" && (target == "/" || target == "/index.html")
        return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"],
                             body = console_html())
    elseif method == "GET" && target == "/api/snapshot"
        return _json(state.snapshot())
    elseif method == "GET" && target == "/api/health"
        n = lock(state.lock) do; length(state.clients); end
        return _json((ok = true, clients = n, interval = state.interval, at = string(now())))
    elseif method == "POST" && target == "/api/execute"
        return _json(_intent(state.on_execute, req))
    elseif method == "POST" && target == "/api/kill"
        return _json(_intent(state.on_kill, req))
    end
    return _json((ok = false, error = "no route $method $target",
                  routes = ["GET /", "GET /api/snapshot", "GET /api/health",
                            "POST /api/execute", "POST /api/kill"]); status = 404)
end

"""
Run one intent, turning any exception into a reported refusal.

A throw here would close the connection and leave the operator with a spinner and no reason,
which at T-12 is indistinguishable from the process having died. The exception text is returned
instead -- this console is on a LAN and its operator is the person who would read the stack trace
anyway.
"""
function _intent(f::Function, req::HTTP.Request)
    body = String(req.body)
    args = isempty(body) ? Dict{String,Any}() :
           try Dict{String,Any}(JSON3.read(body)) catch; Dict{String,Any}() end
    try
        out = isempty(args) ? f() : f(args)
        return out isa NamedTuple ? out : (ok = true, result = string(out))
    catch e
        return (ok = false, error = sprint(showerror, e))
    end
end

# ===================================================================
# WebSocket push
# ===================================================================

"Send the current snapshot to every live client, dropping the ones that have gone."
function push_snapshot!(state::ConsoleState)
    payload = try
        JSON3.write(state.snapshot())
    catch e
        JSON3.write((error = sprint(showerror, e),))
    end
    dead = Any[]
    lock(state.lock) do
        for ws in state.clients
            try
                HTTP.WebSockets.send(ws, payload)
            catch
                push!(dead, ws)
            end
        end
        isempty(dead) || filter!(w -> !(w in dead), state.clients)
    end
    return length(dead)
end

function _ws_loop(state::ConsoleState, ws)
    lock(state.lock) do; push!(state.clients, ws); end
    try
        HTTP.WebSockets.send(ws, JSON3.write(state.snapshot()))   # first frame, immediately
        for _ in ws                                               # keep open; ignore client text
        end
    catch
        # a client closing mid-read is normal, not an error worth logging every reconnect
    finally
        lock(state.lock) do; filter!(w -> w !== ws, state.clients); end
    end
end

function _stream_handler(state::ConsoleState)
    return function (http::HTTP.Streams.Stream)
        if HTTP.WebSockets.isupgrade(http.message)
            HTTP.WebSockets.upgrade(ws -> _ws_loop(state, ws), http)
            return nothing
        end
        req = http.message
        req.body = read(http)
        resp = try
            route_request(state, req)
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

# ===================================================================
# Lifecycle
# ===================================================================

"""
    serve_console(state; host = "127.0.0.1", port = 8080, push = true) -> ConsoleState

Start the console. Non-blocking: returns as soon as the socket is listening.

Binds to **loopback** by default. This page can commit a slate; it should be reached over an SSH
tunnel or a tailnet, and a default of `0.0.0.0` would put the Execute button on whatever network
the machine happens to be on.
"""
function serve_console(state::ConsoleState; host = "127.0.0.1", port::Integer = 8080,
                       push::Bool = true, verbose = -1)
    state.running && error("serve_console: this ConsoleState is already serving. " *
                           "Call stop_console! first.")
    state.server = HTTP.serve!(_stream_handler(state), host, port;
                               stream = true, verbose = verbose)
    state.running = true
    if push
        state.pusher = Threads.@spawn begin
            while state.running
                try
                    push_snapshot!(state)
                catch
                end
                sleep(state.interval)
            end
        end
    end
    return state
end

"Stop the console and release the port. Idempotent."
function stop_console!(state::ConsoleState)
    state.running = false
    if state.server !== nothing
        try close(state.server) catch end
        state.server = nothing
    end
    lock(state.lock) do
        for ws in state.clients
            try close(ws) catch end
        end
        empty!(state.clients)
    end
    state.pusher = nothing
    return state
end
