# Comprehensive Guide: Setting up Kaimon.jl as a Remote MCP Server in Claude Code

This guide provides step-by-step instructions for configuring **Kaimon.jl** as a remote Model Context Protocol (MCP) server for Claude Code. It assumes you are running Claude Code on an Arch Linux laptop and connecting to a headless Linux server over a secure Tailscale network.

## 0. Prerequisites
- **Tailscale:** Installed, running, and connected on both your laptop and remote server. (Verify with `ping <server-tailscale-ip>`).
- **Julia 1.12+:** Kaimon requires at least Julia 1.12. Update via `juliaup` on your server if necessary (`juliaup add 1.12 && juliaup default 1.12`).

## 1. Installing and Running Kaimon.jl on the Server
Kaimon provides a rich Terminal UI (TUI) and is designed to run persistently in the background.

1. **SSH into your server:**
   ```bash
   ssh user@<server-tailscale-ip>
   ```
2. **Start a Tmux Session:** This ensures the Kaimon server remains alive when you disconnect.
   ```bash
   tmux new -s kaimon_server
   ```
3. **Install Kaimon globally:**
   Open the Julia REPL and install Kaimon as a Julia app:
   ```julia
   julia> ]
   pkg> app add Kaimon
   ```
   *(Ensure `~/.julia/bin` is in your `PATH` via your `~/.bashrc` or `~/.zshrc`)*.
4. **Launch Kaimon:**
   ```bash
   kaimon
   ```
5. **Complete the Setup Wizard:**
   - **Security Mode:** Select **`gentle`** or **`standard`**. Because you are on a secure Tailscale network, this grants the AI enough autonomy to work efficiently without pausing for manual permission on every command.
   - **API Key:** Let the wizard generate an API key. **Copy this key**; you will need it for Claude Code.
   - **Port:** Leave it as the default `2828`.

## 2. Network Configuration (Tailscale)
Since you are on Arch Linux, ensure your server's firewall allows incoming traffic on port 2828 strictly from the Tailscale interface:
```bash
sudo ufw allow in on tailscale0 to any port 2828
```
*(You can now safely detach from your Tmux session using `Ctrl+b` then `d`)*.

## 3. Configuring Claude Code on Your Laptop
To connect Claude Code to your remote Kaimon server via Server-Sent Events (SSE), you need to pass your API key as an HTTP Authorization header. 

First, grab your server's Tailscale IP by running `tailscale status` on your laptop.

### Option A: Using the Claude Code CLI (Recommended)
You can add the Kaimon server directly using the `claude mcp add` command. For SSE remote endpoints, Claude Code uses the `http` transport type:

```bash
claude mcp add --transport sse kaimon-remote \
  "http://<SERVER_TAILSCALE_IP>:2828/mcp" \
  --header "Authorization: Bearer <YOUR_COPIED_API_KEY>"
```
*Note: Make sure to place the `--header` argument after the URL to ensure the CLI parses it correctly.*

### Option B: Manual JSON Configuration
If you prefer editing the Claude Code configuration file directly (e.g., your `.mcp.json` or global Claude configuration), structure the headers in the server configuration block like this:

```json
{
  "mcpServers": {
    "kaimon-remote": {
      "type": "http",
      "url": "http://<SERVER_TAILSCALE_IP>:2828/sse",
      "headers": {
        "Authorization": "Bearer <YOUR_COPIED_API_KEY>"
      }
    }
  }
}
```

## 4. Agent Protocols & Workflows (CRITICAL)
When Claude Code interacts with Kaimon, it must adhere to strict session management protocols to prevent spawning "zombie" processes or crashing the server. Kaimon provides ~30 advanced tools natively, but they must be used carefully.

### A. Session Management
Do **NOT** repeatedly start new sessions.
1. **Start a Session:** Call the `start_session` tool with the target project path (e.g., `{"project_path": "/root/BayesianFootball"}`). 
   - *Note: This path must be whitelisted in the Kaimon TUI first.*
2. **Save the Key:** Extract the 8-character session key (e.g., `ec76949f`) returned by the server.
3. **Use the Key:** Pass this key as the `"ses"` parameter in **all** subsequent code evaluations.

### B. Evaluating Code (`ex`)
Use the `ex` tool to run Julia code in the persistent REPL:
- `"ses"`: The 8-character session key.
- `"e"`: The Julia code to evaluate. 
> **Important:** Make sure the final expression of the code block is the variable you want returned. Kaimon automatically strips `println()` and `display()` output to prevent TUI spam.

### C. Background Jobs & Polling
If an evaluation takes longer than 30 seconds (e.g., heavy precompiling), Kaimon promotes it to a background job and returns an `eval_id` (e.g., `"63cbc80d"`).
- Use the `check_eval` tool to poll the status (using the `eval_id` without the `ses` parameter). 
- **Rate Limit Polling:** Sleep for at least **10-15 seconds** between polling attempts. Rapid polling floods the Kaimon TUI with ghost client connections.

### D. Extending with "The Gate"
Kaimon automatically generates JSON schemas for custom Julia functions using reflection. 
If you write a domain-specific function in Neovim locally and execute it in your remote REPL, you can expose it directly to Claude Code by registering it:
```julia
Kaimon.register_tool(my_custom_analysis_function)
```
Claude Code will instantly see this new tool in its MCP context, bridging your local development environment directly to the AI's native capabilities.
