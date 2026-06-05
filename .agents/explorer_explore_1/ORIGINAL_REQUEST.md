## 2026-06-05T15:38:41Z
You are explorer_explore_1, a Codebase and Data Explorer.
Your working directory is /home/james/bet_project/BayesianFootball/.agents/explorer_explore_1/.
Please investigate:
1. Where matches, statistics, and SofaScore momentum data (especially `momentum_vector`) are stored in the database or how they are loaded/structured in memory (e.g., in `DataStore` or `Incidents`). Find any database table or Julia struct representing them.
2. Examine the file `src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl` (or similar path with correct case) and explain its structure and how lambda values / latent team parameters (like home attack, away defense) are defined and calculated.
3. Document your findings in /home/james/bet_project/BayesianFootball/.agents/explorer_explore_1/analysis.md and write a handoff.md in your working directory.
When done, send a message back to the Project Orchestrator (conversation ID 429c198b-bf9f-4617-ab4a-a7c770a4b4c1) detailing your findings and references.
