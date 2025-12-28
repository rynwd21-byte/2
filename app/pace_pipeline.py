from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text, select

from .cfbd import get as cfbd_get
from .db import SessionLocal
from .models import Drive, Game, Team, Rating, TeamSeasonPace
from .data_pipeline import ensure_tables, get_team_by_name, now_iso, _clear_combined_cache


def _parse_time(obj: Any) -> Tuple[Optional[int], Optional[int]]:
    """Parse CFBD time objects.

    CFBD drive responses typically contain:
      start_time: { minutes: int, seconds: int }
      end_time:   { minutes: int, seconds: int }
    """
    if not obj or not isinstance(obj, dict):
        return None, None
    m = obj.get("minutes")
    s = obj.get("seconds")
    try:
        return (int(m) if m is not None else None), (int(s) if s is not None else None)
    except Exception:
        return None, None


async def ingest_cfbd_drives(season: int, season_type: str = "regular") -> Dict[str, Any]:
    """Ingest drive summaries from CFBD into the local SQLite DB."""
    ensure_tables()

    data = await cfbd_get(
        "/drives",
        params={"year": season, "seasonType": season_type},
    )

    inserted = 0
    updated = 0
    skipped = 0
    errors: List[str] = []

    with SessionLocal() as sess:
        for d in data:
            try:
                drive_uid = d.get("id")
                game_id = d.get("game_id")
                if drive_uid is None or game_id is None:
                    skipped += 1
                    continue

                offense_name = d.get("offense")
                defense_name = d.get("defense")
                if not offense_name or not defense_name:
                    skipped += 1
                    continue

                # Ensure teams exist; map names -> ids.
                off_team = get_team_by_name(sess, offense_name)
                def_team = get_team_by_name(sess, defense_name)

                sm, ss = _parse_time(d.get("start_time"))
                em, es = _parse_time(d.get("end_time"))

                existing = sess.execute(
                    select(Drive).where(Drive.drive_uid == int(drive_uid))
                ).scalar_one_or_none()

                payload = dict(
                    drive_uid=int(drive_uid),
                    game_id=int(game_id),
                    season=int(season),
                    week=(int(d.get("week")) if d.get("week") is not None else None),
                    season_type=season_type,
                    offense_id=off_team.team_id,
                    defense_id=def_team.team_id,
                    offense_name=str(offense_name),
                    defense_name=str(defense_name),
                    drive_number=(int(d.get("drive_number")) if d.get("drive_number") is not None else None),
                    drive_result=d.get("drive_result"),
                    scoring=(1 if d.get("scoring") else 0),
                    start_period=(int(d.get("start_period")) if d.get("start_period") is not None else None),
                    end_period=(int(d.get("end_period")) if d.get("end_period") is not None else None),
                    start_minutes=sm,
                    start_seconds=ss,
                    end_minutes=em,
                    end_seconds=es,
                    plays=(int(d.get("plays")) if d.get("plays") is not None else None),
                    yards=(int(d.get("yards")) if d.get("yards") is not None else None),
                )

                if existing:
                    for k, v in payload.items():
                        setattr(existing, k, v)
                    updated += 1
                else:
                    sess.add(Drive(**payload))
                    inserted += 1
            except Exception as exc:
                errors.append(str(exc)[:200])

        sess.commit()

    _clear_combined_cache()
    return {
        "season": season,
        "season_type": season_type,
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "errors": errors[:5],
    }


PACE_SQL = """
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
        -- convert period+clock (mm:ss remaining) to elapsed seconds from start
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
"""


def compute_and_store_pace(
    start_season: int,
    end_season: int,
    min_games: int = 6,
    min_drives: int = 300,
) -> Dict[str, Any]:
    """Compute pace metrics from drives and store into TeamSeasonPace + Rating(pace)."""
    ensure_tables()
    with SessionLocal() as sess:
        rows = sess.execute(
            text(PACE_SQL),
            {
                "start_season": int(start_season),
                "end_season": int(end_season),
                "min_games": int(min_games),
                "min_drives": int(min_drives),
            },
        ).mappings().all()

        if not rows:
            return {
                "start_season": start_season,
                "end_season": end_season,
                "rows": 0,
                "note": "No pace rows produced. Ensure drives are ingested for these seasons.",
            }

        # League average seconds per drive by season
        by_season: Dict[int, List[float]] = {}
        for r in rows:
            by_season.setdefault(int(r["season"]), []).append(float(r["sec_per_drive"]))
        league_sec = {s: (sum(v) / len(v)) for s, v in by_season.items()}

        # Clear existing materialized pace rows for this season range
        sess.execute(
            text("DELETE FROM team_season_pace WHERE season BETWEEN :a AND :b"),
            {"a": int(start_season), "b": int(end_season)},
        )
        # Clear existing pace ratings (optional, but keeps combined values clean)
        sess.execute(
            text("DELETE FROM ratings WHERE source = 'pace'"),
        )

        now = now_iso()
        out_rows = 0
        for r in rows:
            season = int(r["season"])
            team_id = int(r["team_id"])
            sec_per_drive = float(r["sec_per_drive"]) if r["sec_per_drive"] is not None else None
            if not sec_per_drive or sec_per_drive <= 1:
                continue
            lg = float(league_sec.get(season) or 150.0)
            # Map seconds-per-drive into the simulator's 0-100-ish pace scale.
            pace_rating = 60.0 * (lg / sec_per_drive)
            pace_rating = max(30.0, min(90.0, pace_rating))

            sess.add(
                TeamSeasonPace(
                    season=season,
                    team_id=team_id,
                    games_played=int(r["games_played"]),
                    drives_for=int(r["drives_for"]),
                    drives_against=int(r["drives_against"]),
                    drives_for_per_game=float(r["drives_for_per_game"]),
                    drives_against_per_game=float(r["drives_against_per_game"]),
                    sec_per_drive_for=float(r["sec_per_drive_for"]),
                    sec_per_drive_against=float(r["sec_per_drive_against"]),
                    sec_per_drive=sec_per_drive,
                    pace_rating=pace_rating,
                    updated_at=now,
                )
            )
            # Store a single consensus pace rating source that the simulator can pick up.
            sess.add(
                Rating(
                    team_id=team_id,
                    source="pace",
                    pace=pace_rating,
                    updated_at=now,
                )
            )
            out_rows += 1

        sess.commit()

    _clear_combined_cache()
    return {
        "start_season": start_season,
        "end_season": end_season,
        "rows": out_rows,
        "min_games": min_games,
        "min_drives": min_drives,
    }
