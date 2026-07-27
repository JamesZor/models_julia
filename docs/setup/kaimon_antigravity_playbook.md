# Kaimon Server Protocol for Antigravity Agents

> [!IMPORTANT]
> **READ THIS BEFORE ATTEMPTING TO USE KAIMON.**
> If you are an AI Agent reading this document, you are operating in an environment where the Antigravity CLI has a known bug: it fails to inject Kaimon's native MCP tools into your system declarations. You **MUST** use `curl` via the `run_command` tool to interact with Kaimon. 

This playbook outlines the exact JSON-RPC protocols you must follow to prevent spawning "zombie" processes, avoid flooding the user's Terminal UI (TUI) with ghost clients, and successfully execute Julia code.

---

## 1. Authentication & The Base Payload

All Kaimon interactions must be sent as a `POST` request to `http://localhost:2828/mcp`. 
You **must** include the API key as a Bearer token: `Authorization: Bearer kaimon_4c628f9d82e37ba0559281b9d85c9b63465869e7`.

Your bash commands should always follow this structure:
```bash
curl -s -X POST \
  -H "Authorization: Bearer kaimon_4c628f9d82e37ba0559281b9d85c9b63465869e7" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "<TOOL_NAME>", "arguments": {<ARGS>}}}' \
  http://localhost:2828/mcp
```

---

## 2. Session Management (CRITICAL)

> [!WARNING]
> Do **NOT** repeatedly call `start_session`. Do **NOT** call `ex` without a `ses` parameter. Failing to manage sessions correctly will crash the user's server by booting up dozens of concurrent Julia processes that will lock up the CPU and RAM.

### Step A: Start or Identify a Session
Before running any code, you must ensure a Julia REPL session is alive for the target project.
Tool: `start_session`
Arguments: `{"project_path": "/root/BayesianFootball"}` *(Note: The user must have whitelisted this absolute server path in the Kaimon TUI first!)*

If successful, Kaimon will return something like: `"Session started. Session key: ec76949f"`.
**You must extract and remember this 8-character session key (e.g., `ec76949f`).**

### Step B: Using the Session Key
Every subsequent code evaluation **MUST** include `"ses": "ec76949f"` in the arguments. If you omit it, the command will fail or attempt to spawn a new process.

---

## 3. Evaluating Code (`ex`)

Use the `ex` tool to inject Julia code into the persistent REPL session.

**Arguments:**
- `ses`: The 8-character session key.
- `e`: The exact Julia string to evaluate.
- `q`: Set to `false` if you want the result returned to you. (Defaults to `true` / silent).

> [!TIP]
> Kaimon completely strips `println()` and `display()` calls from the returned payload to prevent console spam. If you need output, ensure the **final expression** of your Julia code block is the variable or string you want returned.
> *Example:* `df = DataFrame(A=1:5); summary = describe(df); summary`

---

## 4. Handling Long-Running Background Jobs

If your Julia code triggers heavy precompilation or takes longer than 30 seconds, Kaimon will automatically promote it to a background job and return an `eval_id` (e.g., `"eval_id":"63cbc80d"`).

### How to Poll for Completion
You must use the `check_eval` tool to poll the status.
**Tool:** `check_eval`
**Arguments:** `{"eval_id": "63cbc80d"}` *(Note: `check_eval` does NOT take the `ses` parameter!)*

> [!CAUTION]
> **Avoid Ghost Client Spam:** Every time you run `curl`, Kaimon logs a new stateless MCP client connection on the user's dashboard. If you poll rapidly (e.g., every 1 second), you will flood their screen with hundreds of ghost agents. 
> You MUST write a bash script that sleeps for at least **10-15 seconds** between `check_eval` curl requests to minimize dashboard spam.

### Example Polling Script
```bash
EVAL_ID="63cbc80d"
for i in {1..20}; do
    sleep 15
    RESP=$(curl -s -X POST -H "Authorization: ..." -H "Content-Type: application/json" -d "{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"tools/call\", \"params\": {\"name\": \"check_eval\", \"arguments\": {\"eval_id\": \"$EVAL_ID\"}}}" http://localhost:2828/mcp)
    
    if [[ "$RESP" != *"running"* ]]; then
        echo "$RESP"
        exit 0
    fi
done
```

---

## 5. Cleaning up Disasters (`manage_repl`)

If you suspect you have broken the state or spawned zombie processes, you can either gracefully shut them down or ask the user to run `pkill -9 julia`.
To gracefully kill a session natively:
**Tool:** `manage_repl`
**Arguments:** `{"command": "shutdown", "session": "<SESSION_KEY>"}`
