# Pace metrics + SQL (SQLite)

This app represents pace as a **0–100-ish rating** on `Rating.pace` where:

- **Higher pace** ⇒ drives consume **less time** ⇒ **more drives** occur before the game clock hits 0.
- **60** is “neutral”.

To compute pace from historical data we ingest **drive summaries** from CFBD `/drives` into `drives`, then compute team-season pace metrics and map them into the simulator’s `pace` rating.

## Required tables/columns

### `games`
This project stores games in `games` with (at minimum):
- `game_id` (PK)
- `season`
- `home_id`, `away_id`
- `home_pts`, `away_pts`

### `drives`
This project stores drives in `drives` with (at minimum):
- `game_id`, `season`
- `offense_id`, `defense_id`
- `start_period`, `end_period`
- `start_minutes`, `start_seconds` (clock remaining in period)
- `end_minutes`, `end_seconds` (clock remaining in period)

> OT periods are excluded from timing calculations (period 1–4 only).

## Pace metrics

For each team-season:

- `drives_for_per_game`
- `drives_against_per_game`
- `sec_per_drive_for`
- `sec_per_drive_against`
- `sec_per_drive` = average of offense + defense seconds per drive

### Mapping into simulator `pace`

We compute a league-average seconds-per-drive **by season**, then map:

```
pace_rating = 60 * (league_sec_per_drive / team_sec_per_drive)
```

and clamp to `[30, 90]`.

So a team that uses **less time per drive** than the league gets a **higher** pace rating.

## Exact SQL used

This is the exact SQL used by `app/pace_pipeline.py` (`PACE_SQL`).

```sql
WITH team_games AS (
  SELECT season, game_id, home_id AS team_id FROM games
   WHERE season BETWEEN :start_season AND :end_season
     AND home_pts IS NOT NULL AND away_pts IS NOT NULL
  UNION ALL
  SELECT season, game_id, away_id AS team_id FROM games
   WHERE season BETWEEN :start_season AND :end_season
     AND home_pts IS NOT NULL AND away_pts IS NOT NULL
),

drive_durations AS (
  SELECT
    d.game_id,
    d.season,
    d.offense_id AS team_id,
    d.defense_id AS opp_id,
    CASE
      WHEN d.start_period BETWEEN 1 AND 4
       AND d.end_period BETWEEN 1 AND 4
       AND d.start_minutes IS NOT NULL AND d.start_seconds IS NOT NULL
       AND d.end_minutes   IS NOT NULL AND d.end_seconds   IS NOT NULL
      THEN
        (
          ((d.end_period - 1) * 900 + (900 - (d.end_minutes * 60 + d.end_seconds)))
          -
          ((d.start_period - 1) * 900 + (900 - (d.start_minutes * 60 + d.start_seconds)))
        )
      ELSE NULL
    END AS drive_sec
  FROM drives d
  WHERE d.season BETWEEN :start_season AND :end_season
),

agg AS (
  SELECT
    tg.season,
    tg.team_id,
    COUNT(DISTINCT tg.game_id) AS games_played,

    SUM(CASE WHEN dd.team_id = tg.team_id THEN 1 ELSE 0 END) AS drives_for,
    SUM(CASE WHEN dd.opp_id  = tg.team_id THEN 1 ELSE 0 END) AS drives_against,

    1.0 * SUM(CASE WHEN dd.team_id = tg.team_id THEN 1 ELSE 0 END)
      / COUNT(DISTINCT tg.game_id) AS drives_for_per_game,
    1.0 * SUM(CASE WHEN dd.opp_id = tg.team_id THEN 1 ELSE 0 END)
      / COUNT(DISTINCT tg.game_id) AS drives_against_per_game,

    1.0 * SUM(CASE WHEN dd.team_id = tg.team_id THEN CASE WHEN dd.drive_sec < 0 THEN 0 ELSE dd.drive_sec END ELSE 0 END)
      / NULLIF(SUM(CASE WHEN dd.team_id = tg.team_id THEN 1 ELSE 0 END), 0) AS sec_per_drive_for,

    1.0 * SUM(CASE WHEN dd.opp_id = tg.team_id THEN CASE WHEN dd.drive_sec < 0 THEN 0 ELSE dd.drive_sec END ELSE 0 END)
      / NULLIF(SUM(CASE WHEN dd.opp_id = tg.team_id THEN 1 ELSE 0 END), 0) AS sec_per_drive_against

  FROM team_games tg
  LEFT JOIN drive_durations dd
    ON dd.game_id = tg.game_id AND dd.season = tg.season
  GROUP BY tg.season, tg.team_id
)

SELECT
  season,
  team_id,
  games_played,
  drives_for,
  drives_against,
  drives_for_per_game,
  drives_against_per_game,
  sec_per_drive_for,
  sec_per_drive_against,
  (0.5 * (sec_per_drive_for + sec_per_drive_against)) AS sec_per_drive
FROM agg
WHERE games_played >= :min_games
  AND drives_for >= :min_drives
  AND drives_against >= :min_drives;
```

Parameters:
- `:start_season`, `:end_season`
- `:min_games` (default 6)
- `:min_drives` (default 300)

## Operational flow

1) Ingest drives:
- `POST /cron/ingest-drives?season=2024`

2) Compute pace ratings:
- `POST /cron/compute-pace?start_season=2019&end_season=2024`

3) Run your simulation endpoints.

Pace is automatically picked up because `compute-pace` writes a `Rating` row with `source='pace'` for each team.
