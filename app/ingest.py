from __future__ import annotations
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from .db import SessionLocal, engine, Base
from .models import Team, Game
from .cfbd import get as cfbd_get

def init_db():
    Base.metadata.create_all(bind=engine)

def upsert_team(sess: Session, name: str, conference: Optional[str] = None) -> Team:
    t = sess.execute(select(Team).where(Team.name == name)).scalar_one_or_none()
    if not t:
        t = Team(name=name, conference=conference or None); sess.add(t); sess.flush()
    else:
        if conference and t.conference != conference: t.conference = conference
    return t

async def fetch_and_store_teams() -> int:
    init_db()
    data = await cfbd_get("/teams/fbs")
    count = 0
    with SessionLocal() as sess:
        for item in data:
            name = item.get("school"); conf = item.get("conference")
            if not name: continue
            upsert_team(sess, name=name, conference=conf); count += 1
        sess.commit()
    return count

async def fetch_and_store_games(season: int, team: Optional[str] = None, week: Optional[int] = None) -> int:
    init_db()

    def _score(g: dict, field_base: str):
        """Return the score for home/away from a CFBD game payload.
        CFBD has historically used `home_points`/`away_points` but this may
        change over time. We defensively look for a few common variants
        and return the first one that is actually present (including 0).
        """
        for key in (f"{field_base}_points", f"{field_base}Points",
                   f"{field_base}_score", f"{field_base}Score"):
            if key in g and g[key] is not None:
                return g[key]
        return None

    # Restrict to FBS to avoid CFBD 404s for mixed-division queries
    params = {"year": season, "division": "fbs"}
    if week is not None:
        params["week"] = week
    if team:
        params["team"] = team
    data = await cfbd_get("/games", params=params)
    count = 0
    with SessionLocal() as sess:
        for g in data:
            home_name = g.get("home_team") or g.get("homeTeam") or g.get("home") or g.get("home_school")
            away_name = g.get("away_team") or g.get("awayTeam") or g.get("away") or g.get("away_school")
            if not home_name or not away_name:
                # Skip rows without proper names to avoid NOT NULL violations
                continue
            home = upsert_team(sess, home_name)
            away = upsert_team(sess, away_name)
            season_v = g.get("season")
            week_v = g.get("week")
            season_type = g.get("seasonType") or g.get("season_type")
            home_pts = _score(g, "home")
            away_pts = _score(g, "away")
            date_v = g.get("start_date") or g.get("start_time_tbd") or g.get("startTime")
            neutral = 1 if g.get("neutral_site") else 0
            exists = sess.execute(select(Game).where(
                Game.season == season_v, Game.week == week_v,
                Game.home_id == home.team_id, Game.away_id == away.team_id
            )).scalar_one_or_none()
            if exists:
                changed = False
                if home_pts is not None and exists.home_pts != home_pts:
                    exists.home_pts = home_pts
                    changed = True
                if away_pts is not None and exists.away_pts != away_pts:
                    exists.away_pts = away_pts
                    changed = True
                if date_v and exists.date != date_v:
                    exists.date = date_v
                    changed = True
                if season_type and exists.season_type != season_type:
                    exists.season_type = season_type
                    changed = True
                if exists.neutral != neutral:
                    exists.neutral = neutral
                    changed = True
                if changed:
                    sess.add(exists)
            else:
                game = Game(
                    season=season_v,
                    week=week_v,
                    date=str(date_v) if date_v else None,
                    neutral=neutral,
                    home_id=home.team_id,
                    away_id=away.team_id,
                    home_pts=home_pts,
                    away_pts=away_pts,
                    season_type=season_type,
                )
                sess.add(game)
            count += 1
        sess.commit()
    return count

async def fetch_and_store_market_lines(season: int) -> int:
    """Fetch closing/consensus Vegas lines from CFBD and attach them
    to existing Game rows for the given season.

    Returns the number of games that were updated.
    """
    init_db()
    params: dict[str, object] = {
        "year": season,
        "division": "fbs",
    }
    data = await cfbd_get("/lines", params=params)
    if not data:
        return 0

    updated = 0
    with SessionLocal() as sess:
        for g in data:
            home_name = g.get("homeTeam") or g.get("home_team")
            away_name = g.get("awayTeam") or g.get("away_team")
            week_v = g.get("week")
            if not home_name or not away_name or week_v is None:
                continue

            # Choose a single line record: prefer consensus, otherwise first with both spread and total.
            chosen = None
            for line in g.get("lines") or []:
                spread = line.get("spread")
                total = line.get("overUnder")
                if spread is None or total is None:
                    continue
                if chosen is None or line.get("provider") == "consensus":
                    chosen = line
            if not chosen:
                continue

            spread = float(chosen.get("spread"))
            total = float(chosen.get("overUnder"))

            # Look up teams by name
            home_team = sess.execute(select(Team).where(Team.name == home_name)).scalar_one_or_none()
            away_team = sess.execute(select(Team).where(Team.name == away_name)).scalar_one_or_none()
            if not home_team or not away_team:
                continue

            game = sess.execute(
                select(Game).where(
                    Game.season == season,
                    Game.week == week_v,
                    Game.home_id == home_team.team_id,
                    Game.away_id == away_team.team_id,
                )
            ).scalar_one_or_none()
            if not game:
                # We only want to attach lines to games we already know about.
                continue

            game.closing_spread = spread
            game.closing_total = total
            sess.add(game)
            updated += 1

        sess.commit()

    return updated
