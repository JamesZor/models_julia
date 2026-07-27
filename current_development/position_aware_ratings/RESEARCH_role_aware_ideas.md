# Can we improve position-aware ratings with the data we have? — research synthesis

> Deep-research batch `batch-20260627-130505-1a266db9` (5 Gemini-Pro workers), orchestrated +
> verified 2026-06-27. Full worker reports under `~/.antigravity-jobs/batch-20260627-130505-1a266db9/`.
> Reframe that triggered this: **the SofaScore rating is the wrong yardstick — the market is.**

## Executive summary

1. **The Gate-3 null is explained and expected, not a data defect.** SofaScore/WhoScored/FotMob
   ratings are computed *conditional on the position played* (role-specific event weights + baselines),
   so an out-of-position player who racks up normal counting stats for the assigned role gets a normal
   rating. The penalty (lost off-ball discipline, positional awareness) is invisible to an event-count
   rating. A within-player `rating ~ off_modal` regression returns ~0 **by construction**.
2. **To expose a role effect you must switch to a role-neutral *outcome* target** (xG, xA, shot/chance
   creation, defensive actions, touches in box) — and judge the whole thing **against the market**, not
   against the rating. This is your existing per-line LogLoss/GLMEdge frame (r12), correctly applied.
3. **Our data is richer than generic SofaScore aggregate.** We already pull per-player `expected_goals`,
   `expected_assists`, and the JSON stat block (`bigChanceCreated`, `touchesInOppBox`, shots, etc. — see
   the `datastore-feature-screen` finding). So several "role-neutral" targets the research called
   un-reconstructable *are* available to us at the coverage levels noted there.
4. **One genuinely new, repo-aligned method stands out: prior-informed Bayesian RAPM** — regress team
   match-level **xG-differential** on on-pitch player indicators with Ridge/Bayesian shrinkage, using
   each player's historical rating + xG/xA as the **prior mean**. It directly yields a player-value
   estimate that can deviate from the market, and the Bayesian shrinkage is exactly how this repo already
   thinks. (Plain summing of player ratings — what the live pipeline does — is the weak baseline.)
5. **Sobering caveat (manage expectations):** lineup/availability edges largely **do not survive the
   closing line** — public lineup news is priced by kickoff. If position-aware info helps anywhere, the
   research points at **derivative/totals markets** (formation → Over/Under, corners) over 1X2, which
   dovetails with your split-market-pillar work. Don't expect to beat 1X2 closing lines with this.

---

## Findings by theme

### A. Why the rating can't see out-of-position play (confirms our null)
Composite ratings weight actions by the **assigned match position** — WhoScored uses position-specific
thresholds for "above/beyond baseline" actions; SofaScore documents "exceptions depending on player
position" (saves weighted for keepers, goals for attackers, etc.). They track *statistical events* and
miss off-ball tactical discipline — precisely the thing degraded out of position. So the rating is a
**role-conditioned** quality measure, not absolute. ⇒ our Gate-3 ≈ 0 is the predicted result.
*Sources: whoscored.com/Explanations, sofascore.com/news/sofascore-statistical-ratings-explained — two
independent first-party methodology pages. Verified.*

### B. Role-neutral targets that *would* move out of position
xG, xA, and especially **expected threat (xT)** judge the location/outcome of actions objectively, so a
player lacking the spatial instincts of a new role under-produces against that role's benchmark.
**Progressive passes/carries and shot-creating actions** are objective volume benchmarks (a CB shifted
to FB shows a sharp progressive-action drop). On the defensive side, **"fault-based" metrics** (penalising
structural errors / failure to cover space) expose positional failure far better than praise-based counts
like total tackles.
- *Reconstructable for us:* xG, xA, tackles, interceptions, passes, touches — **plus** (richer than the
  worker assumed) `bigChanceCreated`, `touchesInOppBox`, shots, from our lineups JSON stats.
- *Not reconstructable* without coordinate/event data: true **xT**, progressive passes/carries,
  shot-creating-action chains, precise zone entries.
- *Verification note:* the sport-science "effect size" claims and the arXiv IDs `2604.05678` /
  `2606.01234` could not be corroborated and look fabricated — treat the **principle** as sound but the
  **magnitudes as unverified.**

### C. Testing a signal against the market (your existing frame, validated)
Best practice matches what the repo already does, with two refinements:
- **Devig with Shin's method**, not multiplicative — Shin accounts for favourite-longshot bias and is the
  cited best practice. (Worth checking what `Markets`/`summarize_betfair_market` currently uses.)
- **CLV / beating the closing line** over a large sample is the gold-standard proof of edge, above
  short-run P&L — consistent with your `staking-research-conclusions` and r12.
- **Information-beyond-market test:** logit of outcome on `[devigged_market_prob, your_signal]`; a
  significant signal coefficient = novel information. (This is literally GLMEdge.)
- **Walk-forward only**, never random k-fold (look-ahead). Minor leagues are *less efficient* (thin
  liquidity) **but carry higher vig**, so the edge must clear a wider margin.
*Sources: methodology is standard; cited URLs were low-quality affiliate sites, but the techniques are
textbook and already in use here. Verified by practice, not by those URLs.*

### D. Which valuation frameworks are actually feasible on our data
| Framework | Needs | Feasible for us? |
|---|---|---|
| VAEP, xT, EPV | event-stream (SPADL) / tracking coords | **No** — no coordinate data |
| Player role clustering (K-means/GMM) | nominal position + per-90 xG/xA + rating | **Partial/yes** — defines empirical roles to *condition* on |
| RAPM (vanilla) | on/off lineups + outcome | feasible but breaks on soccer collinearity (same XI plays 90′ together) + sparse goals |
| **Prior-informed Bayesian RAPM** | lineups + **team xG-diff target** + player rating/xG **as priors** | **Yes, recommended** — shrinkage fixes small-sample/collinearity |
*Source for the RAPM recommendation: squared2020 prior-informed RAPM + American Soccer Analysis
goals-added (the collinearity/low-event problems are well documented). Method verified; predictive edge
not.*

### E. Do lineup signals beat the market?
- Key-player absence shifts win prob materially (worker says 5–15%; **magnitude unverified**, direction
  consistent); **goalkeeper and playmaker/"link" absences** are the strongest signals.
- **Formation changes move derivative markets** (Totals, Corners) more predictably than 1X2.
- Lineup/availability features **improve log-loss over history-only baselines** (pre-market).
- **But the edge lives in a 60–75-min pre-kickoff lag window and does not survive the closing line.**
  Once public lineup news is in, closing lines are efficient. *Consistent with market-efficiency
  consensus and your own finding that L2/L3 layers didn't add betting return.*

---

## Recommendation: if we do a Phase 2, do THIS (ranked)

1. **Re-run the existing position-aware MVP plan, but with a role-neutral target and a market verdict.**
   Cheapest test of the original idea done right: build the position-aware feature (the off-modal δ on
   xG/chance-creation, not on rating) and judge it with per-line LogLoss + GLMEdge vs de-vigged odds,
   **focusing on totals/BTTS/derivative lines** (where E says the effect should live), not 1X2.
2. **Prior-informed Bayesian RAPM as a new team-strength signal.** Target = team match xG-differential;
   design matrix = on-pitch player indicators; prior mean per player = historical rating + xG/xA; Ridge /
   Turing shrinkage for collinearity. Then GLMEdge the resulting strength estimate against the market.
   This is the most novel, most repo-native idea and doesn't depend on the position-Δ at all.
3. **Role clustering to replace G/D/M/F.** Our 4-bucket taxonomy may be too coarse (the Gate-2 entropy was
   real but blunt). K-means on (position, per-90 xG/xA, rating) gives empirical roles to condition on —
   feeds either (1) or (2).

**Expectation setting:** B and E together say the realistic upside is a small structural-model
improvement on **totals/derivative markets**, not a 1X2 closing-line beat. Given the
`staking-research-conclusions` memory (edge came from market curation + contrarian tilt, not extra model
layers), gate any Phase-2 build hard on a GLMEdge that the single rating doesn't already open.

## Unverified / discard
- All specific **effect-size magnitudes** (out-of-position performance drop %, 5–15% win-prob shift).
- arXiv IDs `2604.05678`, `2606.01234` (xDT/DxT) and most cited affiliate URLs — not corroborated.
- FotMob's exact algorithm (closed-source; only directional claims).
- The claim that role-conditioned ratings + xG "could identify mispriced squad cohesion" — flagged by the
  worker itself as speculative and **not** shown to beat closing-line log-loss.
