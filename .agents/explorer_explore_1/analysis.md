# Bayesian Football Investigation Report: Data Structures & Pregame Model Analysis

This document details the findings from the investigation of matches, statistics, and SofaScore momentum data storage/structures, as well as a comprehensive review of the `DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel` structure and mathematical equations.

---

## Part 1: Database Storage and Memory Structures

The data pipeline for the `BayesianFootball` project is located in `src/Data/`. It is structured around the `DataStore` object, which is loaded from PostgreSQL (database name `sofascrape_db`) and cached in memory using Julia serialization (`.jls` files).

### 1. Matches Data
* **Database Storage**: Matches are stored in the database table `matches` and joined with the `seasons` table.
  * **Query**: `SELECT m.tournament_id, m.season_id, s.year AS season, m.match_id, m.raw_data -> 'tournament' ->> 'slug' AS tournament_slug, m.home_team, m.away_team, m.home_score, m.away_score, m.home_score_ht, m.away_score_ht, m.winner_code, m.start_timestamp, m.round, (m.raw_data ->> 'hasXg')::boolean AS has_xg, (m.raw_data ->> 'hasEventPlayerStatistics')::boolean AS has_stats FROM matches m JOIN seasons s ON m.season_id = s.season_id WHERE m.status_type = 'finished' AND m.tournament_id = ANY($1)` (from `src/Data/fetchers/sql/matches.jl`).
* **Memory Structure**: Matches are stored as a `DataFrame` in the `.matches` field of the `DataStore` struct (defined in `src/Data/types.jl`).
  * **Processing**: During processing (in `process_data`), the raw timestamp is decomposed into `match_hour`, `match_month`, `match_dayofweek`, and `match_date`. In addition, match week (`match_week`), biweek (`match_biweek`), and month index are appended.
  * **Schema**: Enforced via `MATCHES_SCHEMA` which maps column names to types (e.g., `:match_id => Int32`, `:home_team => InlineStrings.String31`, etc.).

### 2. Statistics Data
* **Database Storage**: Statistics are stored in the database table `match_statistics`, filtered by matching matches' tournament IDs.
  * **Query**: `SELECT DISTINCT m.match_id, m.tournament_id, m.season_id, s.period, s.stat_key, s.home_value, s.away_value FROM match_statistics s JOIN matches m ON s.match_id = m.match_id WHERE m.tournament_id = ANY($1)` (from `src/Data/fetchers/sql/statistics.jl`).
* **Memory Structure**: Stored as a `DataFrame` in the `.statistics` field of the `DataStore` struct.
  * **Processing**: In Julia-side processing, the long format stats table from PostgreSQL is pivoted (unstacked) by `stat_key` into a wide table containing both home and away values (`$(stat_key)_home` and `$(stat_key)_away`). These wide tables are then inner-joined on `[:match_id, :tournament_id, :season_id, :period]`.
  * **Schema**: Identifiers like `match_id` are cast to `Int32` and `period` to `InlineStrings.String31`, whereas all other pivoted stat keys are mapped to `Union{Missing, Float64}`.

### 3. Incidents Data
* **Database Storage**: Incidents are stored in the database table `match_incidents`.
  * **Query**: `SELECT i.id, i.match_id, i.incident_type, i.time, i.is_home, i.added_time, i.data -> 'player' ->> 'slug' AS player_name, i.data -> 'playerIn' ->> 'slug' AS player_in_name, i.data -> 'playerOut' ->> 'slug' AS player_out_name, i.data -> 'assist1' ->> 'slug' AS assist1_name, i.data -> 'assist2' ->> 'name' AS assist2_name, i.data ->> 'incidentClass' AS incident_class, i.data ->> 'reason' AS reason, (i.data ->> 'injury')::boolean AS is_injury, (i.data ->> 'rescinded')::boolean AS rescinded, i.data ->> 'text' AS period_text, (i.data ->> 'timeSeconds')::numeric AS time_seconds FROM match_incidents i JOIN matches m ON i.match_id = m.match_id WHERE m.tournament_id = ANY($1)` (from `src/Data/fetchers/sql/incidents.jl`).
* **Memory Structure**: Stored as a `DataFrame` in the `.incidents` field of the `DataStore` struct and typed via `INCIDENTS_SCHEMA`.

### 4. SofaScore Momentum Data (`momentum_vector`)
* **Database Storage**: Momentum data is stored in the database table `match_graph`.
  * **Query**: `SELECT mg.match_id, mg.points FROM match_graph as mg INNER join matches as mm on mg.match_id = mm.match_id WHERE mm.tournament_id = ANY($1)` (from `eda/match_graphes/00_fetch_data.jl`).
  * **Structure of `points`**: A JSON string containing an array of objects representing game momentum at specific match times, e.g. `[{"minute":1,"value":10}, {"minute":2.5,"value":-5}, ...]`.
* **Memory Structure**: Currently, momentum data is **not** loaded into the default `DataStore` or pipeline. It is only fetched and parsed in the exploratory script `eda/match_graphes/00_fetch_data.jl`.
  * **In-Memory parsing**:
    1. `parse_match_graph_to_dict` reads the `points` JSON string and creates a dictionary of type `Dict{Float64, Int}` mapping fractional/integer minutes to momentum values.
    2. `dict_to_momentum_vector` takes this dictionary and constructs an indexed `Vector{Int}`.
       - Vector length is determined by `ceil(Int, maximum_minute)`.
       - Every element is initialized to `0` (neutral momentum).
       - Values are assigned to indices by rounding minutes (`round(Int, min_val)`). Collision overwrites are allowed (e.g. `45.5` and `46.0` both map to index `46`), and if rounding pushes the index out of bounds, the vector is expanded.

---

## Part 2: Pregame Engine Model Structure and Mathematical Equations

The file `src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl` defines `DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel`. It leverages player positional ratings (Goalkeeper `G`, Defender `D`, Midfielder `M`, Forward `F`) to predict goals and expected goals (xG), calibrating parameters against actual match goals, match xG, and market odds.

### 1. General Structure of the File
1. **Model Configuration (`DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel`)**: A Julia struct containing configuration parameters and priors for the model components (Interception, Dynamics, Dispersion, Home Advantage, Kappa, Player Ratings Feature, and Market Features).
2. **Turing Engine (`build_double_poisson_xg_market_player_engine`)**: The core Turing submodel function that implements the mathematical model, mapping inputs (ratings, goals, xG, market expectations) to a joint likelihood.
3. **Builder (`build_turing_model`)**: Extracts features from a `FeatureSet` and computes match time-decay weights:
   $$w_{\text{match}} = 0.5^{\frac{\Delta t}{t_{\text{half-life}}}}$$
   where $\Delta t$ represents days since the match and $t_{\text{half-life}}$ is the dynamics half-life (e.g., 60 or 180 days).
4. **Extractor (`extract_parameters`)**: Computes posterior rates ($\lambda_h, \lambda_a$) and true xG values for each match using MCMC chain samples.

### 2. Player Ratings Feature Processing
Positional player ratings are extracted chronologically in `src/features/extractors/player_extractors.jl`. For each match:
- Player ratings are time-weighted by the minutes they played:
  $$\text{weighted\_rating} = \text{pre\_match\_rating} \times \frac{\text{clamp}(\text{minutes\_played}, 0, 90)}{90}$$
- The weighted ratings are aggregated by team side and position (resulting in 8 values per match: home/away for Goalkeeper `G`, Defender `D`, Midfielder `M`, and Forward `F`).
- To improve convergence, the outfield positions (Defender, Midfielder, Forward) are combined into a single outfield rating, while the Goalkeeper remains separate.

### 3. Mathematical Equations and Parameter Definitions

#### A. Centered Ratings
Ratings are centered around a global base rating $r_{\text{base}}$:
- **Goalkeeper Centered Rating ($G_c$)**:
  $$G_c = G_{\text{rating}} - r_{\text{base}}$$
- **Outfield Centered Rating ($O_c$)**:
  $$O_c = (D_{\text{rating}} + M_{\text{rating}} + F_{\text{rating}}) - 10 \times r_{\text{base}}$$
  *(Where $10 \times r_{\text{base}}$ adjusts for the 10 outfield players).*

#### B. Latent Team Strengths
Using global attacking and defending weights ($w_{\text{G\_att}}, w_{\text{Outfield\_att}}, w_{\text{G\_def}}, w_{\text{Outfield\_def}}$) sampled from priors:
- **Home Attack strength ($att_h$)**:
  $$att_h = w_{\text{G\_att}} \times G_{c, h} + w_{\text{Outfield\_att}} \times O_{c, h}$$
- **Home Defense strength ($def_h$)**:
  $$def_h = w_{\text{G\_def}} \times G_{c, h} + w_{\text{Outfield\_def}} \times O_{c, h}$$
- **Away Attack strength ($att_a$)**:
  $$att_a = w_{\text{G\_att}} \times G_{c, a} + w_{\text{Outfield\_att}} \times O_{c, a}$$
- **Away Defense strength ($def_a$)**:
  $$def_a = w_{\text{G\_def}} \times G_{c, a} + w_{\text{Outfield\_def}} \times O_{c, a}$$

#### C. Log Latent Goal Expectancy
Let $\mu_m$ be the time-decay interception term computed from season base intercept ($\mu_{\text{base}, s}$) and monthly delta ($\delta_{\text{month}, m}$):
$$\mu_m = \mu_{\text{base}, s} + \delta_{\text{month}, m}$$
Let $\gamma_h$ be the home advantage parameter for the home team.
The log expected goals (before team-level actual-goal scaling) are defined as:
$$\log \lambda'_h = \text{clamp}(\mu_m + \gamma_h + att_h + def_a, -20.0, 20.0)$$
$$\log \lambda'_a = \text{clamp}(\mu_m + att_a + def_h, -20.0, 20.0)$$

#### D. Poisson Goal Rates (Lambdas)
Actual goals are modelled using a Poisson distribution. The Poisson rate parameters ($\lambda_h, \lambda_a$) scale the latent goal expectancies by team-specific conversion factors $\kappa$ (kappa):
$$\lambda_h = \kappa_h \times \exp(\log \lambda'_h) + 10^{-6}$$
$$\lambda_a = \kappa_a \times \exp(\log \lambda'_a) + 10^{-6}$$

These $\lambda$ values represent the final expected goals for actual matches:
$$\text{Home Goals} \sim \text{Poisson}(\lambda_h)$$
$$\text{Away Goals} \sim \text{Poisson}(\lambda_a)$$

---

## Part 3: The Three Pillars of Likelihood Co-Training

The model parameters are trained jointly by adding three log-likelihood pillars to the Turing accumulator (`Turing.@addlogprob!`):

### Pillar A: Expected Goals (xG)
Matches with xG data are calibrated using a Gamma distribution. The shape parameter $\nu_{\text{xg}}$ controls dispersion, and the rate parameters correspond to $\exp(\log \lambda'_h)$ and $\exp(\log \lambda'_a)$ respectively:
$$xG_h \sim \text{Gamma}\left(\nu_{\text{xg}}, \frac{\exp(\log \lambda'_h)}{\nu_{\text{xg}}}\right)$$
$$xG_a \sim \text{Gamma}\left(\nu_{\text{xg}}, \frac{\exp(\log \lambda'_a)}{\nu_{\text{xg}}}\right)$$
The xG likelihood contribution is weighted by match time-decay weights and masked for games where xG was not tracked.

### Pillar B: Actual Goals
Actual goals are modeled directly using the scaled Poisson distributions:
$$\text{Home Goals} \sim \text{Poisson}(\lambda_h)$$
$$\text{Away Goals} \sim \text{Poisson}(\lambda_a)$$
This contribution is weighted by the match time-decay weights.

### Pillar C: Market Expectation Co-Training (Bookmaker Odds)
Bookmaker expectations are integrated by comparing the model's total expected goals (including the goal conversion factor $\kappa$) against the market's log expectations ($\log \lambda_{\text{market}, h}$ and $\log \lambda_{\text{market}, a}$) using a Normal likelihood:
$$\log \lambda_{\text{market}, h} \sim \text{Normal}\left(\log \lambda'_h + \log \kappa_h, \sigma_{\text{market}}\right)$$
$$\log \lambda_{\text{market}, a} \sim \text{Normal}\left(\log \lambda'_a + \log \kappa_a, \sigma_{\text{market}}\right)$$
This contribution is masked by availability, weighted by match decay weights, and multiplied by the global config parameter `market_weight`.
