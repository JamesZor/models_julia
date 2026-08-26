# T002 kickoff prompt

Paste the block below into a fresh Claude Code session started in
`/home/james/bet_project/BayesianFootball`.

---

Work ticket **T002** to completion.

Read these three files first, in order:

1. `docs/tickets/T002-scalar-taped-likelihood.md` — the canonical brief: evidence, root
   cause with `file:line`, blast radius, three fix options, acceptance criteria, scope guard.
2. `tickets/t002/README.md` — working notes: measured baseline, confirmed code path,
   attribution table, and the open decisions that are yours to make.
3. `tickets/README.md` — the ticket workflow this repo uses.

**The problem in one line:** gradient evaluation is 1.15 ms because the ReverseDiff tape
holds 35,421 instructions (24.6 per observation) for a 51-parameter model — the maths is
fine, the tape is not.

**Two causes that must be fixed together.** `view(...)` on a `TrackedArray` yields
`SubArray{TrackedReal}` and forces every downstream broadcast onto the scalar path
(`src/models/pregame/engines/team_level/time_decay/goals.jl:54-57`). Switching to
`getindex` collapses the tape from 15,120 instructions to 5 for the identical value — and
then the gradient throws `InexactError` from `src/MyDistributions/negative_binomial.jl:79`,
where `Int(k)` cannot accept a ForwardDiff dual. The slow path is currently masking a crash
on the fast one.

Note that `view` being slower than `getindex` is the **opposite** of Rule 4 in
`docs/turing_ad_performance_guide.md`. The engines follow the guide. If you fix the engines
without fixing the guide, the next engine reintroduces this.

**Workflow.** Branch `fix/t002-scalar-taped-likelihood` off `feat/scottish-lower-protocol`.
Mark T002 `in progress` in both `docs/tickets/T002-scalar-taped-likelihood.md` and
`docs/tickets/README.md`. Run `tickets/t002/reproduce.jl` to confirm the baseline **before**
changing production code, and again after.

Execution happens on the user's server through the kaimon MCP REPL, not locally — the local
checkout has no data cache. Local edits reach the server only via `git push` then pull in
`/root/BayesianFootball`. That checkout has unrelated untracked research files; never clean
them. Start the Julia session with `julia --project -t 16`, then `pinthreads(:cores)` and
`BLAS.set_num_threads(1)`.

**The invariant.** Section 5 of the reproducer runs `tp_gate_equation_parity`, which compares
DynamicPPL's log density against an independent implementation written from the model
documentation rather than from the engine. It currently passes at max |Δ| = **0.000e+00**.
It must still be exactly 0 when you are done. This is a pure AD-performance ticket: same
posterior, same numbers, fewer tape nodes. If the density moves at all, you have changed the
model and gone out of scope.

**Done means** every acceptance criterion in the canonical brief passes — tape size
independent of row count, fold-20 gradient median under 1 ms, no `InexactError` on the
vectorised path, parity still exactly 0, Rule 4 corrected in the guide, and all 403 package
tests green. Put the durable resolution in the canonical ticket, not only in the workspace,
then mark it `done`.

Report honestly: if an acceptance criterion cannot be met, say which and why rather than
relaxing it. If you find a *separate* defect while working, raise it as T003 rather than
folding it in.
