# Opponent-Adjusted Efficiency (Game-Level)

This app currently stores historical results at the **game level** (not drive-by-drive) in SQLite.
To correct for schedule strength in a simple, explainable way, we compute **opponent-adjusted points-per-game (PPG)** and map it into the simulator's generic offense/defense ratings.

## What it measures

Raw PPG can be misleading:

- Scoring 35 points against an elite defense is not the same as scoring 35 against a weak defense.
- Likewise, allowing 24 points against an elite offense is more impressive than allowing 24 against a weak offense.

Opponent-adjusted efficiency answers:

> How many points does a team score/allow **relative to what its opponents typically allow/score**?

## Adjustments

For each team *t* over a season range:

- `team_ppg_for(t)` = average points scored per game
- `team_ppg_against(t)` = average points allowed per game

Then compute schedule-strength averages:

- `opp_avg_ppg_allowed(t)` = average of each opponent's `team_ppg_against`
- `opp_avg_ppg_scored(t)` = average of each opponent's `team_ppg_for`

Finally compute:

- `off_adj_ppg(t) = team_ppg_for(t) - opp_avg_ppg_allowed(t)`
- `def_adj_ppg(t) = opp_avg_ppg_scored(t) - team_ppg_against(t)`

Interpretation:

- Positive `off_adj_ppg` means the team scores **more than expected** given the defenses faced.
- Positive `def_adj_ppg` means the team allows **less than expected** given the offenses faced.

## How ratings are seeded

These adjustments are mapped into the simulator's generic ratings:

- `Team.off_rush = Team.off_pass = off_adj_ppg * scale`
- `Team.def_rush = Team.def_pass = def_adj_ppg * scale`

Where `scale` is a tuning constant (default `10.0`) to put ratings into a useful numeric range.

## Exact SQL

The endpoint `POST /ratings/seed-opp-adjusted` runs the following SQL against the local SQLite DB.

```sql
WITH games_clean AS (
  SELECT
    season,
    home_id,
    away_id,
    home_pts,
    away_pts,
    season_type
  FROM games
  WHERE
    season BETWEEN :start_season AND :end_season
    AND home_pts IS NOT NULL
    AND away_pts IS NOT NULL
    AND (
      :include_postseason = 1
      OR season_type IS NULL
      OR season_type = 'regular'
    )
),
team_games AS (
  SELECT
    season,
    home_id AS team_id,
    away_id AS opp_id,
    home_pts AS pf,
    away_pts AS pa
  FROM games_clean
  UNION ALL
  SELECT
    season,
    away_id AS team_id,
    home_id AS opp_id,
    away_pts AS pf,
    home_pts AS pa
  FROM games_clean
),
team_stats AS (
  SELECT
    team_id,
    COUNT(*) AS games_played,
    AVG(pf * 1.0) AS team_ppg_for,
    AVG(pa * 1.0) AS team_ppg_against
  FROM team_games
  GROUP BY team_id
),
opp_avgs AS (
  SELECT
    tg.team_id,
    AVG(opp.team_ppg_against) AS opp_avg_ppg_allowed,
    AVG(opp.team_ppg_for) AS opp_avg_ppg_scored
  FROM team_games tg
  JOIN team_stats opp
    ON opp.team_id = tg.opp_id
  GROUP BY tg.team_id
)
SELECT
  ts.team_id AS team_id,
  ts.games_played AS games_played,
  ts.team_ppg_for AS team_ppg_for,
  ts.team_ppg_against AS team_ppg_against,
  oa.opp_avg_ppg_allowed AS opp_avg_ppg_allowed,
  oa.opp_avg_ppg_scored AS opp_avg_ppg_scored,
  (ts.team_ppg_for - oa.opp_avg_ppg_allowed) AS off_adj_ppg,
  (oa.opp_avg_ppg_scored - ts.team_ppg_against) AS def_adj_ppg
FROM team_stats ts
JOIN opp_avgs oa
  ON oa.team_id = ts.team_id
WHERE ts.games_played >= :min_team_games
ORDER BY ts.team_id;
```

Parameters:

- `:start_season`, `:end_season` — the season range to compute over
- `:min_team_games` — minimum completed games per team (default 6)
- `:include_postseason` — 1 to include postseason, 0 for regular only

## Endpoint

Use:

- `POST /ratings/seed-opp-adjusted?start_season=2019&end_season=2024&scale=10&min_team_games=6`

Then run simulations normally.
