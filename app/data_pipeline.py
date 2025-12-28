
from __future__ import annotations
import datetime as _dt
from typing import Optional, Dict, Any, List
from sqlalchemy import select
from .db import SessionLocal, Base, engine
from .models import Team, Rating
from .cfbd import get as cfbd_get

_combined_cache: Dict[str, Dict[str, Any]] = {}

def _clear_combined_cache() -> None:
    """Clear the in-process cache of combined team ratings."""
    _combined_cache.clear()


def now_iso() -> str:
    return _dt.datetime.utcnow().isoformat()

def ensure_tables():
    Base.metadata.create_all(bind=engine)

def get_team_by_name(sess, name: str) -> Team:
    t = sess.execute(select(Team).where(Team.name == name)).scalar_one_or_none()
    if not t:
        t = Team(name=name); sess.add(t); sess.flush()
    return t

async def ingest_cfbd_ppa(season: int) -> int:
    """Pull team-level PPA splits (offense + defense) and store as ratings."""
    ensure_tables()
    count=0
    data_off = await cfbd_get("/ppa/teams/offense", params={"year": season})
    data_def = await cfbd_get("/ppa/teams/defense", params={"year": season})
    # Index by team
    off_ix = {d.get("team") or d.get("team_name"): d for d in data_off}
    def_ix = {d.get("team") or d.get("team_name"): d for d in data_def}
    with SessionLocal() as sess:
        for name, o in off_ix.items():
            if not name: continue
            d = def_ix.get(name, {})
            t = get_team_by_name(sess, name)
            def val(dct, path): 
                cur=dct
                for k in path:
                    if cur is None: return None
                    cur = cur.get(k)
                return cur
            # PPA units ~ points per play. We'll scale to a 0..100-ish rating for UI.
            def scale(x):
                if x is None: return None
                return 50 + 300 * float(x)  # heuristic scaling
            r = Rating(team_id=t.team_id, source="cfbd_ppa", 
                       off_rush=scale(val(o, ["rushing", "ppa"])),
                       off_pass=scale(val(o, ["passing", "ppa"])),
                       def_rush=scale(-(val(d, ["rushing", "ppa"]) or 0.0)),  # negative is better defense -> invert
                       def_pass=scale(-(val(d, ["passing", "ppa"]) or 0.0)),
                       st=None, pace=None, updated_at=now_iso())
            sess.add(r); count+=1
        sess.commit()
    return count

async def ingest_cfbd_lines(season: int, week: Optional[int]=None) -> int:
    ensure_tables()
    params={"year": season}
    if week is not None: params["week"]=week
    data = await cfbd_get("/lines", params=params)
    count=0
    with SessionLocal() as sess:
        for g in data:
            home = g.get("homeTeam") or g.get("home_team")
            away = g.get("awayTeam") or g.get("away_team")
            lines = g.get("lines") or []
            # choose a recent/consensus line
            best=None
            for L in lines:
                spread = L.get("spread")
                total = L.get("overUnder") or L.get("total")
                if spread is None and total is None: continue
                best = (spread, total)  # last seen
            if not best: continue
            spread, total = best
            for name, sgn in [(home, +1), (away, -1)]:
                if not name: continue
                t = get_team_by_name(sess, name)
                r = Rating(team_id=t.team_id, source="market", spread=(spread*sgn if spread is not None else None), total=total, updated_at=now_iso())
                sess.add(r); count+=1
        sess.commit()
    return count

def combine_ratings_for_team(sess, team: Team) -> Dict[str, Any]:
    """Combine all Rating rows for a team into a single consensus view.

    - Simple mean across sources for each numeric field.
    - Applies sensible fallbacks so core ratings are never None.
    """ 
    rows = sess.execute(select(Rating).where(Rating.team_id == team.team_id)).scalars().all()
    if not rows:
        # Fall back to the team's own columns with reasonable defaults.
        return dict(
            team_id=team.team_id,
            name=team.name,
            off_rush=(team.off_rush or 50.0),
            off_pass=(team.off_pass or 50.0),
            def_rush=(team.def_rush or 50.0),
            def_pass=(team.def_pass or 50.0),
            st=(team.st or 0.0),
            pace=60.0,
            spread=None,
            total=None,
        )

    agg = dict(off_rush=[], off_pass=[], def_rush=[], def_pass=[], st=[], pace=[], spread=[], total=[])
    for r in rows:
        for k in agg.keys():
            v = getattr(r, k, None)
            if v is not None:
                agg[k].append(float(v))

    out: Dict[str, Any] = {}
    for k, arr in agg.items():
        out[k] = (sum(arr) / len(arr)) if arr else None

    def _fallback(val, team_val, default):
        if val is not None:
            return val
        if team_val not in (None, 0.0):
            return float(team_val)
        return default

    out["team_id"] = team.team_id
    out["team_id"] = team.team_id
    out["name"] = team.name
    out["off_rush"] = _fallback(out.get("off_rush"), team.off_rush, 50.0)
    out["off_pass"] = _fallback(out.get("off_pass"), team.off_pass, 50.0)
    out["def_rush"] = _fallback(out.get("def_rush"), team.def_rush, 50.0)
    out["def_pass"] = _fallback(out.get("def_pass"), team.def_pass, 50.0)
    out["st"] = _fallback(out.get("st"), team.st, 0.0)
    # Pace is not stored on Team, so default to a neutral value when missing.
    if out.get("pace") is None:
        out["pace"] = 60.0

    return out

def combined_by_name(name: str) -> Dict[str, Any] | None:
    """Lookup combined ratings for a team name with a small in-process cache.""" 
    ensure_tables()
    key = (name or "").strip()
    if not key:
        return None

    cached = _combined_cache.get(key)
    if cached is not None:
        return cached

    with SessionLocal() as sess:
        t = sess.execute(select(Team).where(Team.name == key)).scalar_one_or_none()
        if not t:
            return None
        combined = combine_ratings_for_team(sess, t)
        _combined_cache[key] = combined
        return combined
