# Ticket workspaces

`docs/tickets/` is the canonical issue register. `tickets/<id>/` is disposable-but-tracked
working context for an active ticket: reproduction scripts, measurements, decisions, and
handoff notes that would make a fresh agent productive without reconstructing the investigation.

## Workflow

1. Branch from the branch containing the ticket as `fix/<id>-<slug>`.
2. Mark the canonical ticket and `docs/tickets/README.md` as `in progress`.
3. Create `tickets/<id>/README.md` and a deterministic reproducer where practical.
4. Record baseline evidence before changing production code.
5. Implement the narrowest fix allowed by the ticket's scope guard and add regression tests.
6. Run focused tests, the reproducer/measurement matrix, and then the full test suite.
7. Put the durable resolution and blast-radius results in the canonical ticket; do not leave
   essential conclusions only in the workspace.
8. Mark the ticket `done` only when its acceptance criteria pass, then merge the ticket branch.

Remote Kaimon sessions have their own checkout. Commit and push local changes, then fetch and
switch/pull `/root/BayesianFootball` before remote execution. Never clean unrelated untracked
files from that checkout.
