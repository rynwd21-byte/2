"""Opponent-adjusted efficiency seeding (game-level).

The local SQLite DB stores game-level results (not drive-level outcomes).
To keep the first version of opponent adjustment simple and explainable, we
compute points-per-game (PPG) for each team and adjust it by opponent strength.

For a given season range:

1) Build a per-team table of:
   - team_ppg_for      = avg(points_for)
   - team_ppg_against  = avg(points_against)
   - games_played

2) For each team, compute opponent averages across its schedule:
   - opp_avg_ppg_allowed = avg(opponent_team_ppg_against)
   - opp_avg_ppg_scored  = avg(opponent_team_ppg_for)

3) Define adjustments:
   - off_adj_ppg = team_ppg_for - opp_avg_ppg_allowed
   - def_adj_ppg = opp_avg_ppg_scored - team_ppg_against

These adjustments are then mapped to the simulator's generic team ratings:
   team.off_rush = team.off_pass = off_adj_ppg * scale
   team.def_rush = team.def_pass = def_adj_ppg * scale

Notes:
- This does *not* require a regression model and is robust as a first step.
- Use `min_team_games` to avoid unstable teams with tiny samples.
- Default behavior excludes postseason unless `include_postseason=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class OppAdjustedRow:
    team_id: int
    games_played: int
    team_ppg_for: float
    team_ppg_against: float
    opp_avg_ppg_allowed: float
    opp_avg_ppg_scored: float
    off_adj_ppg: float
    def_adj_ppg: float


SQL_OPP_ADJUSTED = """
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
"""


def compute_opponent_adjusted_ppg(
    sess: Session,
    *,
    start_season: int,
    end_season: int,
    min_team_games: int = 6,
    include_postseason: bool = False,
) -> list[OppAdjustedRow]:
    """Return opponent-adjusted PPG rows for all teams in the DB."""
    rows = sess.execute(
        text(SQL_OPP_ADJUSTED),
        {
            "start_season": int(start_season),
            "end_season": int(end_season),
            "min_team_games": int(min_team_games),
            "include_postseason": 1 if include_postseason else 0,
        },
    ).mappings().all()

    out: list[OppAdjustedRow] = []
    for r in rows:
        out.append(
            OppAdjustedRow(
                team_id=int(r["team_id"]),
                games_played=int(r["games_played"]),
                team_ppg_for=float(r["team_ppg_for"] or 0.0),
                team_ppg_against=float(r["team_ppg_against"] or 0.0),
                opp_avg_ppg_allowed=float(r["opp_avg_ppg_allowed"] or 0.0),
                opp_avg_ppg_scored=float(r["opp_avg_ppg_scored"] or 0.0),
                off_adj_ppg=float(r["off_adj_ppg"] or 0.0),
                def_adj_ppg=float(r["def_adj_ppg"] or 0.0),
            )
        )
    return out


def to_debug_dict(row: OppAdjustedRow) -> dict[str, Any]:
    return {
        "team_id": row.team_id,
        "games_played": row.games_played,
        "team_ppg_for": row.team_ppg_for,
        "team_ppg_against": row.team_ppg_against,
        "opp_avg_ppg_allowed": row.opp_avg_ppg_allowed,
        "opp_avg_ppg_scored": row.opp_avg_ppg_scored,
        "off_adj_ppg": row.off_adj_ppg,
        "def_adj_ppg": row.def_adj_ppg,
    }
