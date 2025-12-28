# Full API assembled below
import hashlib
import os, random, datetime as _dt
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from statistics import mean, pstdev
from sqlalchemy import select, func
from .sim_engine import Simulator, TeamState, GameState
from .db import SessionLocal, Base, engine
from .models import Team, Game, Rating, ModelEval
from .ingest import fetch_and_store_teams, fetch_and_store_games, fetch_and_store_market_lines, init_db
from .cfbd import get as cfbd_get
from .model_params import get_params, set_param
from .calibrate import run as calibrate_run, league_ppg, calibrate_coef_scale
from .pace_pipeline import ingest_cfbd_drives, compute_and_store_pace
from .data_pipeline import combine_ratings_for_team


def _now_iso() -> str:
    return _dt.datetime.utcnow().isoformat()

app = FastAPI(title="CFB Drive Sim API")

# Ensure DB schema exists in fresh serverless deployments
@app.on_event("startup")
def _startup_create_tables():
    from .db import Base, engine
    from . import models  # noqa: F401  (register tables)
    Base.metadata.create_all(bind=engine)

sim = Simulator()


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _completed_seasons(min_games: int | None = None) -> list[int]:
    """Seasons in the local DB with enough completed games.

    Used to keep simulations/calibration data-driven (no hardcoded season
    lists) while avoiding accidental inclusion of partial seasons.

    A game counts as "completed" when both home_pts and away_pts are present.
    The threshold can be overridden via SIM_MIN_GAMES_PER_SEASON.
    """
    min_games = int(min_games if min_games is not None else _int_env("SIM_MIN_GAMES_PER_SEASON", 500))
    with SessionLocal() as sess:
        rows = (
            sess.execute(
                select(Game.season, func.count(Game.game_id))
                .where(Game.home_pts.is_not(None), Game.away_pts.is_not(None))
                .group_by(Game.season)
                .order_by(Game.season)
            )
            .all()
        )

    seasons: list[int] = []
    for season, n in rows:
        try:
            if int(n or 0) >= min_games:
                seasons.append(int(season))
        except Exception:
            continue
    return seasons


def _backtest_thresholds() -> dict:
    """Decision thresholds for PASS/EVEN/FAIL.

    Tuned to be conservative (avoid shipping on noise). Can be overridden via env.
    """

    def _f(name: str, default: float) -> float:
        v = os.getenv(name)
        try:
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    return {
        "spread_improve": _f("BT_SPREAD_IMPROVE", 0.20),
        "total_improve": _f("BT_TOTAL_IMPROVE", 0.30),
        "brier_improve": _f("BT_BRIER_IMPROVE", 0.005),
        "spread_regress": _f("BT_SPREAD_REGRESS", 0.20),
        "total_regress": _f("BT_TOTAL_REGRESS", 0.30),
        "brier_regress": _f("BT_BRIER_REGRESS", 0.005),
    }




def _q(sorted_vals: list[float], p: float) -> float | None:
    """Quantile helper for a pre-sorted list.

    Uses linear interpolation and returns None for empty lists.
    """
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 1:
        return float(sorted_vals[-1])

    n = len(sorted_vals)
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


def _run_backtest(
    test_season: int,
    n_games: int = 200,
    n_sims: int = 200,
    seed: int = 1337,
    require_lines: bool = True,
):
    """Evaluate current model vs market on historical games.

    This does not mutate ratings. It uses whatever ratings are currently
    present in the DB (combined across sources) and runs a Monte Carlo
    simulation for each sampled game.
    """
    from sqlalchemy.orm import aliased
    import math

    home_alias = aliased(Team)
    away_alias = aliased(Team)

    with SessionLocal() as sess:
        stmt = (
            select(Game, home_alias, away_alias)
            .join(home_alias, Game.home_id == home_alias.team_id)
            .join(away_alias, Game.away_id == away_alias.team_id)
            .where(
                Game.season == int(test_season),
                Game.home_pts.is_not(None),
                Game.away_pts.is_not(None),
            )
        )
        if require_lines:
            # Backtest metrics (spread MAE/total MAE) require BOTH a closing spread and total.
            # If we only require totals, spread_mae can become None and baseline storage can fail.
            stmt = stmt.where(
                Game.closing_total.is_not(None),
                Game.closing_spread.is_not(None),
            )
        rows = sess.execute(stmt).all()

        # Filter to games that have a usable spread+total when require_lines.
        games = []
        for g, ht, at in rows:
            if require_lines and (g.closing_total is None or g.closing_spread is None):
                continue
            games.append((g, ht, at))

        if not games:
            raise HTTPException(status_code=400, detail="No completed games found for backtest (or missing lines).")

        rng = random.Random(int(seed))
        if n_games and n_games < len(games):
            # Deterministic sample
            idx = list(range(len(games)))
            rng.shuffle(idx)
            games = [games[i] for i in idx[: int(n_games)]]

        spread_abs_errs: list[float] = []
        total_abs_errs: list[float] = []
        briers: list[float] = []
        per_game: list[dict] = []

        for g, ht, at in games:
            home_r = combine_ratings_for_team(sess, ht)
            away_r = combine_ratings_for_team(sess, at)

            hs = TeamState(
                name=ht.name,
                off_rush=float(home_r.get("off_rush", 50.0)),
                off_pass=float(home_r.get("off_pass", 50.0)),
                def_rush=float(home_r.get("def_rush", 50.0)),
                def_pass=float(home_r.get("def_pass", 50.0)),
                st=float(home_r.get("st", 0.0)),
                pace=float(home_r.get("pace", 60.0)),
            )
            as_ = TeamState(
                name=at.name,
                off_rush=float(away_r.get("off_rush", 50.0)),
                off_pass=float(away_r.get("off_pass", 50.0)),
                def_rush=float(away_r.get("def_rush", 50.0)),
                def_pass=float(away_r.get("def_pass", 50.0)),
                st=float(away_r.get("st", 0.0)),
                pace=float(away_r.get("pace", 60.0)),
            )

            # Monte Carlo for this game. Deterministic per game for stable comparisons.
            base_seed = int(hashlib.md5(f"{g.game_id}|{test_season}".encode("utf-8")).hexdigest()[:8], 16)
            rr = random.Random(base_seed)
            hscores: list[int] = []
            ascores: list[int] = []
            hw = 0
            for _ in range(int(n_sims)):
                s = rr.randrange(0, 10_000_000)
                out = sim.sim_game(GameState(home=hs, away=as_), seed=s)
                hscores.append(int(out.score_home))
                ascores.append(int(out.score_away))
                if out.score_home > out.score_away:
                    hw += 1

            mean_home = sum(hscores) / len(hscores)
            mean_away = sum(ascores) / len(ascores)
            model_spread_home = float(mean_home - mean_away)
            model_total = float(mean_home + mean_away)
            p_home_win = float(hw / len(hscores))

            # Market conventions: DB stores closing_spread as (home - away), negative if home favored.
            market_spread_home = float(-g.closing_spread) if g.closing_spread is not None else None
            market_total = float(g.closing_total) if g.closing_total is not None else None

            if market_spread_home is not None:
                spread_abs_errs.append(abs(model_spread_home - market_spread_home))
            if market_total is not None:
                total_abs_errs.append(abs(model_total - market_total))

            outcome = 1.0 if (g.home_pts or 0) > (g.away_pts or 0) else 0.0
            briers.append((p_home_win - outcome) ** 2)


            # Per-game diagnostics for Phase 1 transparency
            spread_samples = [float(h - a) for h, a in zip(hscores, ascores)]
            total_samples = [float(h + a) for h, a in zip(hscores, ascores)]
            spread_samples.sort()
            total_samples.sort()

            spread_p05 = _q(spread_samples, 0.05)
            spread_p95 = _q(spread_samples, 0.95)
            total_p05 = _q(total_samples, 0.05)
            total_p95 = _q(total_samples, 0.95)

            final_home = int(g.home_pts or 0)
            final_away = int(g.away_pts or 0)
            final_margin = float(final_home - final_away)
            final_total = float(final_home + final_away)

            # Market can be None if missing; gate keeping is handled by require_lines.
            spread_err = abs(model_spread_home - market_spread_home) if market_spread_home is not None else None
            total_err = abs(model_total - market_total) if market_total is not None else None

            per_game.append({
                "game_id": int(getattr(g, "id", 0) or 0),
                "home": ht.name,
                "away": at.name,
                "model_spread_home": float(model_spread_home),
                "market_spread_home": float(market_spread_home) if market_spread_home is not None else None,
                "spread_abs_error": float(spread_err) if spread_err is not None else None,
                "model_total": float(model_total),
                "market_total": float(market_total) if market_total is not None else None,
                "total_abs_error": float(total_err) if total_err is not None else None,
                "home_win_prob": float(p_home_win),
                "home_won": 1 if final_home > final_away else 0,
                "final_home": final_home,
                "final_away": final_away,
                "final_margin": float(final_margin),
                "final_total": float(final_total),
                "spread_p05": float(spread_p05),
                "spread_p95": float(spread_p95),
                "total_p05": float(total_p05),
                "total_p95": float(total_p95),
            })




        # ---- Aggregate metrics ----
        spread_errors = [r["spread_abs_error"] for r in per_game if r.get("spread_abs_error") is not None]
        total_errors = [r["total_abs_error"] for r in per_game if r.get("total_abs_error") is not None]
        brier_terms = [
            (float(r.get("home_win_prob", 0.5)) - float(r.get("home_won", 0))) ** 2
            for r in per_game
            if r.get("home_win_prob") is not None
        ]

        spread_mae = float(mean(spread_errors)) if spread_errors else None
        total_mae = float(mean(total_errors)) if total_errors else None
        brier = float(mean(brier_terms)) if brier_terms else None

        # ---- Phase 1 diagnostics ----
        worst_spread_misses = sorted(
            [r for r in per_game if r.get("spread_abs_error") is not None],
            key=lambda r: r["spread_abs_error"],
            reverse=True,
        )[:10]
        worst_total_misses = sorted(
            [r for r in per_game if r.get("total_abs_error") is not None],
            key=lambda r: r["total_abs_error"],
            reverse=True,
        )[:10]

        
        best_spread_hits = sorted(
            [r for r in per_game if r.get("spread_abs_error") is not None],
            key=lambda r: r["spread_abs_error"],
        )[:10]
        best_total_hits = sorted(
            [r for r in per_game if r.get("total_abs_error") is not None],
            key=lambda r: r["total_abs_error"],
        )[:10]

        def _fmt_game_row(r: dict, kind: str) -> dict:
            # kind: "spread" or "total"
            game = f'{r.get("away","")} @ {r.get("home","")}'.strip()
            if kind == "spread":
                model = r.get("model_spread_home")
                market = r.get("market_spread_home")
                abs_err = r.get("spread_abs_error")
            else:
                model = r.get("model_total")
                market = r.get("market_total")
                abs_err = r.get("total_abs_error")
            final = r.get("final_margin") if kind == "spread" else r.get("final_total")
            return {
                "game": game,
                "model": None if model is None else float(model),
                "market": None if market is None else float(market),
                "abs_err": None if abs_err is None else float(abs_err),
                "final": None if final is None else float(final),
                "game_id": r.get("game_id", 0),
                "home": r.get("home"),
                "away": r.get("away"),
            }

        worst_spread_misses_fmt = [_fmt_game_row(r, "spread") for r in worst_spread_misses]
        worst_total_misses_fmt = [_fmt_game_row(r, "total") for r in worst_total_misses]
        best_spread_hits_fmt = [_fmt_game_row(r, "spread") for r in best_spread_hits]
        best_total_hits_fmt = [_fmt_game_row(r, "total") for r in best_total_hits]
# Bucketed MAE
        buckets = {
            "spread_small": [],
            "spread_large": [],
            "total_low": [],
            "total_high": [],
        }
        for r in per_game:
            ms = r.get("market_spread_home")
            mt = r.get("market_total")
            se = r.get("spread_abs_error")
            te = r.get("total_abs_error")
            if ms is not None and se is not None:
                if abs(float(ms)) <= 3.0:
                    buckets["spread_small"].append(float(se))
                else:
                    buckets["spread_large"].append(float(se))
            if mt is not None and te is not None:
                if float(mt) < 45.0:
                    buckets["total_low"].append(float(te))
                elif float(mt) > 60.0:
                    buckets["total_high"].append(float(te))

        bucket_mae = {
            "spread": {
                "small": float(mean(buckets["spread_small"])) if buckets["spread_small"] else None,
                "large": float(mean(buckets["spread_large"])) if buckets["spread_large"] else None,
            },
            "total": {
                "low": float(mean(buckets["total_low"])) if buckets["total_low"] else None,
                "high": float(mean(buckets["total_high"])) if buckets["total_high"] else None,
            },
        }

        # Win prob calibration by decile
        calibration = []
        for lo in range(0, 100, 10):
            hi = lo + 10
            bucket = [
                r
                for r in per_game
                if r.get("home_win_prob") is not None and (lo / 100.0) <= float(r["home_win_prob"]) < (hi / 100.0)
            ]
            if bucket:
                calibration.append(
                    {
                        "bucket": f"{lo}-{hi}%",
                        "games": int(len(bucket)),
                        "avg_pred": float(mean([float(r["home_win_prob"]) for r in bucket])),
                        "actual_win_rate": float(mean([float(r["home_won"]) for r in bucket])),
                    }
                )

        # Coverage of the model's 90% interval (p05..p95) vs actual
        spread_inside = sum(
            1
            for r in per_game
            if r.get("spread_p05") is not None
            and r.get("spread_p95") is not None
            and float(r["spread_p05"]) <= float(r["final_margin"]) <= float(r["spread_p95"])
        )
        total_inside = sum(
            1
            for r in per_game
            if r.get("total_p05") is not None
            and r.get("total_p95") is not None
            and float(r["total_p05"]) <= float(r["final_total"]) <= float(r["total_p95"])
        )
        n_cov = max(1, len(per_game))
        coverage = {
            "spread_p90": float(spread_inside / n_cov),
            "total_p90": float(total_inside / n_cov),
        }

        return {
            "ok": True,
            "test_season": int(test_season),
            "n_games": int(len(per_game)),
            "n_sims": int(n_sims),
            "spread_mae": spread_mae,
            "total_mae": total_mae,
            "brier": brier,
            "worst_spread_misses": worst_spread_misses_fmt,
            "worst_total_misses": worst_total_misses_fmt,
            "best_spread_hits": best_spread_hits_fmt,
            "best_total_hits": best_total_hits_fmt,
            "bucket_mae": bucket_mae,
            "win_prob_calibration": calibration,
            "coverage": coverage,
            "per_game": per_game,

            "diagnostics": {
                "worst_spread_misses": worst_spread_misses,
                "worst_total_misses": worst_total_misses,
                "best_spread_hits": best_spread_hits,
                "best_total_hits": best_total_hits,

                # UI-friendly compact rows (Game / Model / Market / Abs err / Final)
                "worst_spread_misses_fmt": worst_spread_misses_fmt,
                "worst_total_misses_fmt": worst_total_misses_fmt,
                "best_spread_hits_fmt": best_spread_hits_fmt,
                "best_total_hits_fmt": best_total_hits_fmt,

                "bucket_mae": bucket_mae,
                "win_prob_calibration": calibration,
                "coverage": coverage,
                "per_game": per_game,
            },
        }
try:
    from fastapi.middleware.cors import CORSMiddleware

    # NOTE: When allow_credentials=True, you cannot use allow_origins=["*"].
    # Vercel preview/prod URLs change frequently, so we allow any *.vercel.app
    # origin via regex, plus local dev.
    _cors_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_origin_regex=r"^https://.*\.vercel\.app$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

@app.get("/")
def root():
    return {"ok": True, "service": "CFB Drive Sim API", "docs": "/docs",
            "endpoints": ["/simulate-game","/simulate-series","/simulate-by-name","/simulate-series-by-name",
                          "/teams/search","/ingest/*","/ratings/seed","/cfbd/*","/model/*","/cron/*"]}

class TeamIn(BaseModel):
    name: str; off_rush: float; off_pass: float; def_rush: float; def_pass: float; st: float = 0.0
class MatchupIn(BaseModel):
    home: TeamIn; away: TeamIn; seed: int | None = None

@app.post("/simulate-game")
def simulate_game(m: MatchupIn):
    gs = GameState(home=TeamState(**m.home.model_dump()), away=TeamState(**m.away.model_dump()))
    out = sim.sim_game(gs, seed=m.seed)
    return {"home": out.home.name, "away": out.away.name, "score_home": out.score_home, "score_away": out.score_away, "ot_periods": out.ot_periods}

class SeriesIn(BaseModel):
    home: TeamIn; away: TeamIn; n: int = 1000; seed: int | None = None; include_samples: bool = False

@app.post("/simulate-series")

def simulate_series(req: SeriesIn):
    rng = random.Random(req.seed)
    hs, as_, ot, hw = [], [], 0, 0
    for _ in range(req.n):
        s = rng.randrange(0, 10_000_000)
        out = sim.sim_game(
            GameState(
                home=TeamState(**req.home.model_dump()),
                away=TeamState(**req.away.model_dump()),
            ),
            seed=s,
        )
        hs.append(out.score_home)
        as_.append(out.score_away)
        if out.ot_periods > 0:
            ot += 1
        if out.score_home > out.score_away:
            hw += 1

    def q(arr, p):
        arr = sorted(arr)
        if not arr:
            return 0.0
        k = max(0, min(len(arr) - 1, int(round((p / 100.0) * (len(arr) - 1)))))
        return arr[k]

    mean_home = sum(hs) / len(hs) if hs else 0.0
    mean_away = sum(as_) / len(as_) if as_ else 0.0

    resp = {
        "samples": req.n,
        "home_win_pct": (hw / req.n) if req.n else 0.0,
        "away_win_pct": 1 - ((hw / req.n) if req.n else 0.0),
        "ot_rate": (ot / req.n) if req.n else 0.0,
        "mean_score_home": mean_home,
        "mean_score_away": mean_away,
        "stdev_score_home": (
            (sum((x - mean_home) ** 2 for x in hs) / len(hs)) ** 0.5 if hs else 0.0
        ),
        "stdev_score_away": (
            (sum((x - mean_away) ** 2 for x in as_) / len(as_)) ** 0.5 if as_ else 0.0
        ),
        "quantiles": {
            "home": {"p05": q(hs, 5), "p50": q(hs, 50), "p95": q(hs, 95)},
            "away": {"p05": q(as_, 5), "p50": q(as_, 50), "p95": q(as_, 95)},
        },
    }
    if req.include_samples and req.n <= 2000:
        resp["samples_detail"] = {"home": hs, "away": as_}
    return resp


from typing import List
@app.get("/teams/search")
def teams_search(q: str = "") -> List[dict]:
    q = q.strip()
    with SessionLocal() as sess:
        stmt = select(Team).order_by(Team.name.asc())
        if q:
            like = f"%{q}%"
            stmt = select(Team).where(func.lower(Team.name).like(func.lower(like))).order_by(Team.name.asc())
        rows = sess.execute(stmt).scalars().all()
        return [{"team_id": t.team_id, "name": t.name, "conference": t.conference} for t in rows]

class NamesIn(BaseModel):
    home_name: str; away_name: str; seed: int | None = None

@app.post("/simulate-by-name")
def simulate_by_name(names: NamesIn):
    with SessionLocal() as sess:
        home = sess.execute(select(Team).where(Team.name == names.home_name)).scalar_one_or_none()
        away = sess.execute(select(Team).where(Team.name == names.away_name)).scalar_one_or_none()
        if not home or not away: raise HTTPException(status_code=404, detail="Team not found in DB. Run /ingest/teams first or check names.")
        hs = TeamState(name=home.name, off_rush=home.off_rush or 0, off_pass=home.off_pass or 0, def_rush=home.def_rush or 0, def_pass=home.def_pass or 0, st=home.st or 0)
        as_ = TeamState(name=away.name, off_rush=away.off_rush or 0, off_pass=away.off_pass or 0, def_rush=away.def_rush or 0, def_pass=away.def_pass or 0, st=away.st or 0)
    out = sim.sim_game(GameState(home=hs, away=as_), seed=names.seed)
    return {"home": out.home.name, "away": out.away.name, "score_home": out.score_home, "score_away": out.score_away, "ot_periods": out.ot_periods}

class SeriesByNameIn(BaseModel):
    home_name: str; away_name: str; n: int = 1000; seed: int | None = None

@app.post("/simulate-series-by-name-db")
def simulate_series_by_name_db(req: SeriesByNameIn):
    with SessionLocal() as sess:
        home = sess.execute(select(Team).where(Team.name == req.home_name)).scalar_one_or_none()
        away = sess.execute(select(Team).where(Team.name == req.away_name)).scalar_one_or_none()
        if not home or not away: raise HTTPException(status_code=404, detail="Team not found in DB. Run /ingest/teams first or check names.")
        hs = TeamState(name=home.name, off_rush=home.off_rush or 0, off_pass=home.off_pass or 0, def_rush=home.def_rush or 0, def_pass=home.def_pass or 0, st=home.st or 0)
        as_ = TeamState(name=away.name, off_rush=away.off_rush or 0, off_pass=away.off_pass or 0, def_rush=away.def_rush or 0, def_pass=away.def_pass or 0, st=away.st or 0)
    import random as R; hw=0; ot=0; H=[]; A=[]
    for _ in range(req.n):
        s = R.randrange(0, 10_000_000)
        out = sim.sim_game(GameState(home=hs, away=as_), seed=s)
        H.append(out.score_home); A.append(out.score_away)
        if out.ot_periods>0: ot+=1
        if out.score_home>out.score_away: hw+=1
    def q(arr,p): arr=sorted(arr); k=max(0,min(len(arr)-1,int(round((p/100.0)*(len(arr)-1))))); return arr[k]
    return {"samples": req.n, "home": req.home_name, "away": req.away_name,
            "home_win_pct": hw/req.n, "away_win_pct": 1-hw/req.n, "ot_rate": ot/req.n,
            "quantiles": {"home":{"p05":q(H,5),"p50":q(H,50),"p95":q(H,95)},"away":{"p05":q(A,5),"p50":q(A,50),"p95":q(A,95)}}}

@app.post("/ingest/teams")
async def ingest_teams():
    try: return {"inserted_or_updated": await fetch_and_store_teams()}
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))

@app.post("/ingest/games")
async def ingest_games(season: int = Query(..., ge=1869, le=2100), team: str | None = None, week: int | None = None):
    try: return {"inserted_or_updated": await fetch_and_store_games(season=season, team=team, week=week)}
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))
@app.get("/games")
def list_games(season: int = Query(..., ge=1869, le=2100)) -> list[dict]:
    """Return games from the local DB for a given season.

    This is used by the UI's dataset summary to count how many games have
    been ingested for each season. It returns a lightweight representation
    of each game with team names attached.
    """
    from sqlalchemy.orm import aliased

    home_alias = aliased(Team)
    away_alias = aliased(Team)
    with SessionLocal() as sess:
        stmt = (
            select(Game, home_alias.name, away_alias.name)
            .join(home_alias, Game.home_id == home_alias.team_id)
            .join(away_alias, Game.away_id == away_alias.team_id)
            .where(Game.season == season)
            .order_by(Game.season.asc(), Game.season_type.asc().nullsfirst(), Game.week.asc(), Game.date.asc())
        )
        rows = sess.execute(stmt).all()
        out: list[dict] = []
        for g, home_name, away_name in rows:
            out.append(
                {
                    "game_id": g.game_id,
                    "season": g.season,
                    "week": g.week,
                    "season_type": g.season_type,
                    "date": g.date,
                    "neutral": bool(g.neutral),
                    "home": home_name,
                    "away": away_name,
                    "home_pts": g.home_pts,
                    "away_pts": g.away_pts,
                    "ot_periods": g.ot_periods,
                    "closing_spread": g.closing_spread,
                    "closing_total": g.closing_total,
                }
            )
        return out


@app.get("/cfbd/teams")
async def cfbd_teams(fbs: bool = True):
    return await cfbd_get("/teams/fbs" if fbs else "/teams")

@app.get("/cfbd/games")
async def cfbd_games(season: int = Query(..., ge=1869, le=2100), week: int | None = None, team: str | None = None):
    params = {"year": season}; 
    if week is not None: params["week"]=week
    if team: params["team"]=team
    return await cfbd_get("/games", params=params)


@app.post("/ratings/market")
def ratings_market(season: int) -> dict:
    """Compute simple market-based power ratings from Vegas lines.

    For each team we average the closing spread (home spread when home,
    negative spread when away) and the closing total across all games
    in the given season that have Vegas data. The results are stored
    in the Rating table with source="market".
    """
    from datetime import datetime

    with SessionLocal() as sess:
        games = sess.execute(
            select(Game).where(
                Game.season == season,
                Game.closing_spread.is_not(None),
                Game.closing_total.is_not(None),
            )
        ).scalars().all()
        if not games:
            return {
                "ok": False,
                "season": season,
                "updated": 0,
                "error": "No games with Vegas lines for that season. Run /cron/ingest-market-lines first.",
            }

        agg: dict[int, dict[str, float]] = {}
        def get_bucket(team_id: int) -> dict:
            if team_id not in agg:
                agg[team_id] = {"spread_sum": 0.0, "total_sum": 0.0, "n": 0.0}
            return agg[team_id]

        for g in games:
            # home team: spread as given
            b_home = get_bucket(g.home_id)
            b_home["spread_sum"] += float(g.closing_spread)
            b_home["total_sum"] += float(g.closing_total)
            b_home["n"] += 1.0

            # away team: invert spread (since stored from home perspective)
            b_away = get_bucket(g.away_id)
            b_away["spread_sum"] -= float(g.closing_spread)
            b_away["total_sum"] += float(g.closing_total)
            b_away["n"] += 1.0

        now = datetime.utcnow().isoformat()
        updated = 0
        for team_id, d in agg.items():
            if not d["n"]:
                continue
            avg_spread = d["spread_sum"] / d["n"]
            avg_total = d["total_sum"] / d["n"]

            r = sess.execute(
                select(Rating).where(
                    Rating.team_id == team_id,
                    Rating.source == "market",
                )
            ).scalar_one_or_none()
            if not r:
                r = Rating(team_id=team_id, source="market")
            r.spread = float(avg_spread)
            r.total = float(avg_total)
            r.updated_at = now
            sess.add(r)
            updated += 1

        sess.commit()
        return {"ok": True, "season": season, "updated": updated}

@app.post("/ratings/seed")
def ratings_seed(season: int, scale: float = 10.0) -> dict:
    with SessionLocal() as sess:
        teams = {t.team_id: {"obj": t, "pf": 0, "pa": 0, "g": 0} for t in sess.execute(select(Team)).scalars().all()}
        games = sess.execute(select(Game).where(Game.season == season)).scalars().all()
        for g in games:
            if g.home_pts is None or g.away_pts is None: continue
            teams[g.home_id]["pf"] += g.home_pts; teams[g.home_id]["pa"] += g.away_pts; teams[g.home_id]["g"] += 1
            teams[g.away_id]["pf"] += g.away_pts; teams[g.away_id]["pa"] += g.home_pts; teams[g.away_id]["g"] += 1
        per_game = [(tid, d["pf"]/d["g"], d["pa"]/d["g"]) for tid,d in teams.items() if d["g"]>0]
        if not per_game: return {"updated": 0, "note": "No completed games found for that season. Run /ingest/games first."}
        avg_pf = sum(p for _,p,_ in per_game)/len(per_game); avg_pa = sum(a for *_,a in per_game)/len(per_game)
        from datetime import datetime; updated=0; now=datetime.utcnow().isoformat()
        for tid, pfg, pag in per_game:
            off_rating = (pfg - avg_pf) * scale
            def_rating = (avg_pa - pag) * scale
            t = teams[tid]["obj"]
            t.off_rush = t.off_pass = off_rating
            t.def_rush = t.def_pass = def_rating
            t.last_updated = now
            sess.add(t); updated += 1
        sess.commit()
        return {"updated": updated, "avg_pf": avg_pf, "avg_pa": avg_pa, "scale": scale}


@app.post("/ratings/seed-opp-adjusted")
def ratings_seed_opp_adjusted(
    start_season: int = Query(..., ge=1869, le=2100),
    end_season: int = Query(..., ge=1869, le=2100),
    scale: float = 10.0,
    min_team_games: int = Query(6, ge=1, le=50, description="Minimum completed games per team in the season range"),
    include_postseason: bool = Query(False, description="Include postseason games when computing opponent adjustments"),
    debug: bool = Query(False, description="Return sample rows of the opponent-adjusted table"),
) -> dict:
    """Seed ratings using opponent-adjusted efficiency (PPG-based).

    This is a first, low-risk opponent-adjustment pass that corrects for
    schedule strength using only the local DB's game-level results.
    See docs/OPPONENT_ADJUSTED_EFFICIENCY.md for the exact SQL.
    """
    if end_season < start_season:
        raise HTTPException(status_code=400, detail="end_season must be >= start_season")

    from .opp_adjust import compute_opponent_adjusted_ppg, to_debug_dict

    with SessionLocal() as sess:
        rows = compute_opponent_adjusted_ppg(
            sess,
            start_season=start_season,
            end_season=end_season,
            min_team_games=min_team_games,
            include_postseason=include_postseason,
        )

        if not rows:
            return {
                "updated": 0,
                "note": "No opponent-adjusted rows computed. Ensure games are ingested and completed in the requested range.",
            }

        from datetime import datetime

        now = datetime.utcnow().isoformat()
        updated = 0
        for r in rows:
            t = sess.get(Team, r.team_id)
            if not t:
                continue
            t.off_rush = t.off_pass = float(r.off_adj_ppg) * float(scale)
            t.def_rush = t.def_pass = float(r.def_adj_ppg) * float(scale)
            t.last_updated = now
            sess.add(t)
            updated += 1

        sess.commit()

        resp: dict[str, object] = {
            "ok": True,
            "seed_mode": "opp_adjusted_ppg",
            "start_season": start_season,
            "end_season": end_season,
            "min_team_games": min_team_games,
            "include_postseason": include_postseason,
            "updated": updated,
            "scale": scale,
        }
        if debug:
            resp["sample_rows"] = [to_debug_dict(x) for x in rows[:10]]
        return resp

@app.get("/model/params")
def model_params(): return get_params()

@app.post("/model/params")
def set_model_param(name: str, value: float):
    set_param(name, float(value)); sim.set_coef_scale(get_params().get("coef_scale", 1.0))
    return {"ok": True, "params": get_params()}

@app.post("/model/calibrate")
def model_calibrate(season: int, samples: int = 2000, seed: int | None = None):
    out = calibrate_run(season=season, samples=samples, seed=seed); sim.set_coef_scale(get_params().get("coef_scale", 1.0))
    return out

def _env_true(name: str, default: bool=True) -> bool:
    v = os.getenv(name); return v.lower() in ("1","true","yes","on") if v is not None else default

@app.on_event("startup")
async def _bootstrap_on_startup():
    if _env_true("AUTO_BOOTSTRAP", False):
        try:
            await fetch_and_store_teams(); year = _dt.datetime.utcnow().year; await fetch_and_store_games(season=year)
        except Exception as e: print("[bootstrap] skipped:", e)

@app.get("/cron/nightly")
async def cron_nightly(request: Request, seasons: str | None = None):
    secret = os.getenv("CRON_SECRET")
    if secret:
        token = request.headers.get("x-cron-secret") or request.query_params.get("token")
        if token != secret: return {"ok": False, "error": "unauthorized"}
    yrs = [int(s.strip()) for s in seasons.split(",")] if seasons else [_dt.datetime.utcnow().year, _dt.datetime.utcnow().year-1]
    await fetch_and_store_teams(); total=0
    for y in yrs: total += await fetch_and_store_games(season=y)
    try:
        last_complete = yrs[-1] if yrs[-1] < _dt.datetime.utcnow().year else yrs[-2]
        _ = model_calibrate(season=last_complete)
    except Exception as e: print("[cron] calibration skipped:", e)
    return {"ok": True, "games_upserted": total}

@app.get("/cron/calibrate")
def cron_calibrate(request: Request, season: int | None = None, samples: int = 2000):
    secret = os.getenv("CRON_SECRET")
    if secret:
        token = request.headers.get("x-cron-secret") or request.query_params.get("token")
        if token != secret: return {"ok": False, "error": "unauthorized"}
    if season is None: season = _dt.datetime.utcnow().year - 1
    out = calibrate_run(season=season, samples=samples); sim.set_coef_scale(get_params().get("coef_scale", 1.0))
    return {"ok": True, "season": season, "result": out}


@app.get("/cron/calibrate-multi")
async def cron_calibrate_multi(
    request: Request,
    seasons: list[int] | None = Query(None, description="Seasons to include. If omitted, uses all completed seasons in the DB."),
):
    """Calibrate the scoring scale using an average PPG across multiple seasons.

    This is used by the UI's "Run multi-season calibration" button.

    Implementation detail:
    - For each requested season we first look for completed games in the local DB.
    - If none are found, we opportunistically ingest that season from CFBD and
      re-check. This makes the endpoint much more robust in serverless
      environments where the DB may start empty.
    """
    secret = os.getenv("CRON_SECRET")
    if secret:
        token = request.headers.get("x-cron-secret") or request.query_params.get("token")
        if token != secret:
            return {"ok": False, "error": "unauthorized"}

    if not seasons:
        seasons = _completed_seasons()
        if not seasons:
            return {"ok": False, "error": "No completed games found in the DB."}

    targets: list[float] = []
    ingested: dict[int, int] = {}

    for s in seasons:
        # First attempt to compute league PPG from whatever is already in the DB.
        t = league_ppg(s)

        if t is None:
            # No completed games in the DB for this season yet. Try ingesting
            # that season on-the-fly, then recompute the PPG.
            try:
                ingested_count = await fetch_and_store_games(season=s)
                ingested[s] = ingested_count
            except Exception as e:  # pragma: no cover - defensive logging only
                # Swallow ingest errors here and fall through; the error will be
                # surfaced below if we still have no targets.
                print(f"[cron/calibrate-multi] ingest failed for season {s}: {e}")

            # Re-check league PPG after ingest attempt.
            t = league_ppg(s)

        if t is not None:
            targets.append(float(t))

    if not targets:
        # Even after ingest attempts we have no completed games.
        return {"ok": False, "error": "No completed games for the requested seasons."}

    target_ppg = float(sum(targets) / len(targets))
    scale = float(calibrate_coef_scale(target_ppg=target_ppg))

    # Ensure the in-process simulator picks up the new scale
    sim.set_coef_scale(get_params().get("coef_scale", scale))

    return {
        "ok": True,
        "seasons": seasons,
        "target_ppg": target_ppg,
        "coef_scale": scale,
        "ingested": ingested,
    }






@app.get("/cron/ingest-market-lines")
async def cron_ingest_market_lines(request: Request, season: int = Query(..., description="Season to ingest Vegas lines for")):
    """
    Ingest closing/consensus Vegas lines from CFBD and attach them to games
    for the given season.

    This is step 1 of the "market data" pipeline: we only store spread/total
    on existing Game rows and do not yet use them in the rating model.
    """
    secret = os.getenv("CRON_SECRET")
    if secret:
        token = request.headers.get("x-cron-secret") or request.query_params.get("token")
        if token != secret:
            return {"ok": False, "error": "unauthorized"}

    try:
        updated = await fetch_and_store_market_lines(season=season)
    except Exception as e:  # pragma: no cover - defensive logging only
        msg = f"{type(e).__name__}: {e}"
        # Keep logs helpful but responses safe/compact
        print(f"[cron/ingest-market-lines] ERROR for {season}: {msg}")
        hint = ""
        if "401" in msg or "403" in msg:
            hint = " (Check CFBD_API_KEY / authorization.)"
        raise HTTPException(status_code=500, detail=f"Failed to ingest market lines: {msg[:500]}{hint}")

    return {"ok": True, "season": season, "games_with_lines": updated}


@app.post("/cron/ingest-drives")
async def cron_ingest_drives(
    request: Request,
    season: int = Query(..., description="Season (year) to ingest drive summaries for"),
    season_type: str = Query("regular", description="Season type for CFBD /drives (regular or postseason)"),
):
    """Ingest drive summaries used for pace normalization."""
    secret = os.getenv("CRON_SECRET")
    if secret:
        token = request.headers.get("x-cron-secret") or request.query_params.get("token")
        if token != secret:
            return {"ok": False, "error": "unauthorized"}

    try:
        out = await ingest_cfbd_drives(season=season, season_type=season_type)
        return {"ok": True, **out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest drives: {type(e).__name__}: {str(e)[:500]}")


@app.post("/cron/compute-pace")
def cron_compute_pace(
    request: Request,
    start_season: int = Query(..., description="First season"),
    end_season: int = Query(..., description="Last season"),
    min_games: int = Query(6, description="Minimum completed games per team-season"),
    min_drives: int = Query(300, description="Minimum drives for/against per team-season"),
):
    """Compute team-season pace ratings from ingested drives.

    Stores results in `team_season_pace` and also writes a consensus `Rating`
    row with source='pace' so simulations automatically pick it up.
    """
    secret = os.getenv("CRON_SECRET")
    if secret:
        token = request.headers.get("x-cron-secret") or request.query_params.get("token")
        if token != secret:
            return {"ok": False, "error": "unauthorized"}

    if end_season < start_season:
        raise HTTPException(status_code=400, detail="end_season must be >= start_season")
    try:
        out = compute_and_store_pace(
            start_season=start_season,
            end_season=end_season,
            min_games=min_games,
            min_drives=min_drives,
        )
        return {"ok": True, **out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute pace: {type(e).__name__}: {str(e)[:500]}")


@app.get("/health")
def health():
    """
    Lightweight health check. Verifies the DB connection is usable and reports
    whether a CFBD API key is configured.
    """
    ok_db = True
    try:
        with SessionLocal() as sess:
            sess.execute(select(Team)).first()
    except Exception:
        ok_db = False
    # Import here (cheap) so we can expose whether a key is set
    try:
        from .cfbd import API_KEY as _CFBD_KEY
        cfbd_ok = bool(_CFBD_KEY)
    except Exception:
        cfbd_ok = False
    return {"ok": True, "service": "cfb-drive-sim", "db": ok_db, "cfbd": cfbd_ok}

@app.get("/version")
def version():
    return {"name": "cfb-drive-sim", "version": "1.2.0", "build": os.getenv("VERCEL_GIT_COMMIT_SHA", "")[:8]}

@app.post("/ingest/season-range")
async def ingest_season_range(start: int = Query(..., ge=1869, le=2100), end: int = Query(..., ge=1869, le=2100)):
    if end < start:
        raise HTTPException(status_code=400, detail="end must be >= start")
    total = 0
    await ingest_teams()
    for y in range(start, end+1):
        r = await ingest_games(season=y)
        total += r.get("inserted_or_updated", 0) if isinstance(r, dict) else 0
    return {"ok": True, "seasons": [start, end], "total_games_upserted": total}

@app.post("/bootstrap")
async def bootstrap(
    request: Request,
    start_season: int = Query(2019, ge=1869, le=2100, description="First season to ingest"),
    end_season: int = Query(2023, ge=1869, le=2100, description="Last season to ingest"),
    include_market_lines: bool = Query(True, description="Also ingest market lines (closing spreads/totals)"),
    include_pace: bool = Query(False, description="Also ingest drives + compute pace ratings"),
    seed_ratings: bool = Query(True, description="Seed team ratings after ingest/calibration"),
    seed_mode: str = Query(
        "simple",
        description="Ratings seeding mode: 'simple' (single-season PPG) or 'opp_adjusted' (opponent-adjusted PPG).",
    ),
    calibrate_multi: bool = Query(True, description="Run multi-season scoring scale calibration"),
    force: bool = Query(False, description="Force re-ingest/recompute even if data already exists"),
):
    """Idempotent one-click setup for accurate simulations.

    This endpoint is designed for serverless deployments where the SQLite DB may
    start empty.

    What it does (in order):
      1) Teams ingest
      2) Games ingest for [start_season..end_season]
      3) (Optional) Market lines ingest
      4) (Optional) Multi-season calibration across requested seasons
      5) (Optional) Seed simple ratings from most recent completed season

    Idempotency: steps are skipped when the DB already contains data, unless
    `force=true`.
    """

    # Make sure tables exist before any queries
    from .data_pipeline import ensure_tables
    ensure_tables()


    # Optional protection: if CRON_SECRET is set, require the same header/query
    # pattern used by cron endpoints.
    secret = os.getenv("CRON_SECRET")
    if secret:
        token = request.headers.get("x-cron-secret") or request.query_params.get("token")
        if token != secret:
            return {"ok": False, "error": "unauthorized"}

    if end_season < start_season:
        raise HTTPException(status_code=400, detail="end_season must be >= start_season")

    summary: dict[str, object] = {
        "ok": True,
        "requested": {
            "start_season": start_season,
            "end_season": end_season,
            "include_market_lines": include_market_lines,
            "include_pace": include_pace,
            "seed_ratings": seed_ratings,
            "seed_mode": seed_mode,
            "calibrate_multi": calibrate_multi,
            "force": force,
        },
        "steps": {},
    }

    # ---------- helpers ----------
    def _has_any_teams() -> bool:
        with SessionLocal() as sess:
            return sess.execute(select(Team).limit(1)).first() is not None

    def _season_has_completed_games(season: int) -> bool:
        with SessionLocal() as sess:
            q = select(Game).where(
                Game.season == season,
                Game.home_pts.isnot(None),
                Game.away_pts.isnot(None),
            ).limit(1)
            return sess.execute(q).first() is not None

    # ---------- 1) teams ----------
    try:
        if force or (not _has_any_teams()):
            await fetch_and_store_teams()
            summary["steps"]["teams"] = {"action": "ingested"}
        else:
            summary["steps"]["teams"] = {"action": "skipped", "reason": "teams already present"}
    except Exception as e:
        summary["steps"]["teams"] = {"action": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=f"teams ingest failed: {e}")

    # ---------- 2) games (range) ----------
    ingested_games: dict[int, int] = {}
    for y in range(start_season, end_season + 1):
        try:
            if force or (not _season_has_completed_games(y)):
                n = await fetch_and_store_games(season=y)
                ingested_games[int(y)] = int(n or 0)
            else:
                ingested_games[int(y)] = 0
        except Exception as e:
            summary["steps"].setdefault("games", {})
            summary["steps"]["games"][str(y)] = {"action": "error", "error": str(e)}
            raise HTTPException(status_code=500, detail=f"games ingest failed for season {y}: {e}")

    summary["steps"]["games"] = {
        "action": "ingested_or_skipped",
        "range": [start_season, end_season],
        "upserted_by_season": ingested_games,
    }

    # ---------- 2b) drives ingest (pace) ----------
    if include_pace:
        per_season: dict[int, object] = {}
        any_error = False
        total_inserted = 0
        for yr in range(start_season, end_season + 1):
            try:
                out = await ingest_cfbd_drives(season=yr, season_type="regular")
                per_season[yr] = {"ok": True, **out}
                total_inserted += int(out.get("inserted") or 0)
            except Exception as e:
                any_error = True
                per_season[yr] = {"ok": False, "error": str(e)[:500]}
        summary["steps"]["drives"] = {
            "action": "ingested" if not any_error else "partial",
            "total_inserted": total_inserted,
            "per_season": per_season,
            "non_fatal": True,
        }
        if any_error:
            summary.setdefault("warnings", []).append("drives_ingest_failed")
            summary["partial"] = True
    else:
        summary["steps"]["drives"] = {"action": "skipped"}

    # ---------- 2c) compute pace ratings (non-fatal) ----------
    if include_pace:
        try:
            pace_out = compute_and_store_pace(
                start_season=start_season,
                end_season=end_season,
                min_games=_int_env("PACE_MIN_GAMES", 6),
                min_drives=_int_env("PACE_MIN_DRIVES", 300),
            )
            summary["steps"]["pace"] = {"action": "computed", **pace_out}
        except Exception as e:
            summary["steps"]["pace"] = {"action": "error", "error": str(e)[:500], "non_fatal": True}
            summary.setdefault("warnings", []).append("pace_compute_failed")
            summary["partial"] = True
    else:
        summary["steps"]["pace"] = {"action": "skipped"}

# ---------- 3) market lines ----------
    if include_market_lines:
        # Market lines are not required for simulations yet (Phase 2),
        # so this block is always non-fatal. We still try to ingest and
        # return detailed per-season diagnostics to make failures easy to fix.
        per_season: dict[int, object] = {}
        total_updated = 0
        any_error = False
        for yr in range(start_season, end_season + 1):
            try:
                updated = await fetch_and_store_market_lines(season=yr)
                per_season[yr] = {"ok": True, "games_with_lines": updated}
                total_updated += int(updated or 0)
            except Exception as e:
                any_error = True
                msg = getattr(e, "detail", None) or str(e)
                hint = ""
                if "401" in msg or "403" in msg:
                    hint = " Check CFBD_API_KEY / authorization."
                per_season[yr] = {"ok": False, "error": f"{type(e).__name__}: {msg[:500]}", "hint": hint}
        summary["steps"]["market_lines"] = {
            "action": "ingested" if not any_error else "partial",
            "total_games_with_lines": total_updated,
            "per_season": per_season,
            "non_fatal": True,
        }
        if any_error:
            summary.setdefault("warnings", []).append("market_lines_failed")
            summary["partial"] = True
    else:
        summary["steps"]["market_lines"] = {"action": "skipped"}
    # ---------- 4) multi-season calibration ----------
    if calibrate_multi:
        try:
            # Prefer seasons that are actually present in the DB with enough
            # completed games, so we don't have to maintain hardcoded lists.
            seasons = [s for s in _completed_seasons() if start_season <= s <= end_season]
            if not seasons:
                seasons = list(range(start_season, end_season + 1))
            out = await cron_calibrate_multi(request, seasons=seasons)
            summary["steps"]["calibration"] = {"action": "calibrated_multi", "result": out}
        except Exception as e:
            summary["steps"]["calibration"] = {"action": "error", "error": str(e)}
            raise HTTPException(status_code=500, detail=f"multi-season calibration failed: {e}")
    else:
        summary["steps"]["calibration"] = {"action": "skipped"}

    # ---------- 5) ratings seed ----------
    if seed_ratings:
        try:
            # Use the most recent season in the requested range with enough
            # completed games (data-driven).
            seasons = [s for s in _completed_seasons() if start_season <= s <= end_season]
            seed_season = max(seasons) if seasons else end_season
            # Only seed if we have completed games.
            if force or _season_has_completed_games(seed_season):
                mode = (seed_mode or "simple").strip().lower()
                if mode in ("opp_adjusted", "opp", "opponent", "opponent_adjusted"):
                    out = ratings_seed_opp_adjusted(
                        start_season=start_season,
                        end_season=end_season,
                        scale=10.0,
                        min_team_games=6,
                        include_postseason=False,
                        debug=False,
                    )
                    summary["steps"]["ratings_seed"] = {"action": "seeded", "mode": "opp_adjusted", "result": out}
                else:
                    out = ratings_seed(season=seed_season)
                    summary["steps"]["ratings_seed"] = {"action": "seeded", "mode": "simple", "season": seed_season, "result": out}
            else:
                summary["steps"]["ratings_seed"] = {
                    "action": "skipped",
                    "season": seed_season,
                    "reason": "no completed games found",
                }
        except Exception as e:
            summary["steps"]["ratings_seed"] = {"action": "error", "error": str(e)}
            raise HTTPException(status_code=500, detail=f"ratings seed failed: {e}")
    else:
        summary["steps"]["ratings_seed"] = {"action": "skipped"}

    return summary



from .data_pipeline import ingest_cfbd_ppa, ingest_cfbd_lines, combined_by_name

@app.post("/update/ratings")
async def update_ratings(season: int, week: int | None = None) -> dict:
    """Update ratings for a season using CFBD PPA + betting lines.

    This endpoint is intentionally defensive: if CFBD returns HTML or an error
    instead of JSON for the PPA endpoint, we surface that as `ppa_error` in the
    response instead of crashing with a 500.
    """
    ppa_rows: int | None = None
    ppa_error: str | None = None

    # PPA ingestion can fail in a few different ways (HTTP error, non-JSON body).
    try:
        ppa_rows = await ingest_cfbd_ppa(season=season)
    except Exception as exc:  # pragma: no cover - defensive logging
        ppa_error = str(exc)

    # Lines ingestion has historically been more stable, but we still let
    # exceptions propagate as 400/500 if something truly unexpected happens.
    lines_rows = await ingest_cfbd_lines(season=season, week=week)

    resp: dict[str, object] = {
        "ppa_rows": int(ppa_rows or 0),
        "lines_rows": int(lines_rows or 0),
    }
    if ppa_error:
        resp["ppa_error"] = ppa_error
    return resp

@app.get("/ratings/combined")
def ratings_combined(team: str) -> dict:
    r = combined_by_name(team)
    if not r: raise HTTPException(status_code=404, detail="Team not found")
    return r

class NamesAutoIn(BaseModel):
    home_name: str; away_name: str; deterministic: bool = True


@app.post("/simulate-by-name-auto")
def simulate_by_name_auto(req: NamesAutoIn):
    """
    Simulate a game using the consensus auto ratings pipeline.

    - Uses combined team ratings from all available Rating sources.
    - If `deterministic` is true, derives a stable seed from the
      (home, away) team names so the same matchup always produces
      the same result.
    """
    home_r = combined_by_name(req.home_name)
    away_r = combined_by_name(req.away_name)
    if not home_r or not away_r:
        raise HTTPException(status_code=404, detail="Team not found in ratings. Run /update/ratings or check names.")

    hs = TeamState(
        name=home_r["name"],
        off_rush=float(home_r.get("off_rush", 50.0)),
        off_pass=float(home_r.get("off_pass", 50.0)),
        def_rush=float(home_r.get("def_rush", 50.0)),
        def_pass=float(away_r.get("def_pass", 50.0)),
        st=float(home_r.get("st", 0.0)),
        pace=float(home_r.get("pace", 60.0)),
    )
    as_ = TeamState(
        name=away_r["name"],
        off_rush=float(away_r.get("off_rush", 50.0)),
        off_pass=float(away_r.get("off_pass", 50.0)),
        def_rush=float(away_r.get("def_rush", 50.0)),
        def_pass=float(home_r.get("def_pass", 50.0)),
        st=float(away_r.get("st", 0.0)),
        pace=float(away_r.get("pace", 60.0)),
    )

    if req.deterministic:
        key = f"{req.home_name}|{req.away_name}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        seed = int(h[:8], 16)
    else:
        seed = None

    out = sim.sim_game(GameState(home=hs, away=as_), seed=seed)
    return {
        "home": out.home.name,
        "away": out.away.name,
        "score_home": out.score_home,
        "score_away": out.score_away,
        "ot_periods": out.ot_periods,
        "market": {
            "spread": (home_r.get("spread") if home_r else None),
            "total": (home_r.get("total") if home_r else None),
        },
    }


class NamesAutoSeriesIn(BaseModel):
    home_name: str
    away_name: str
    n: int = 1000
    deterministic: bool = True
    include_samples: bool = False



def _lookup_closing_line(home_team_id: int, away_team_id: int):
    """Return the most recent closing line for this exact home/away matchup.

    Stored in `games.closing_spread` from the *home perspective* where
    negative implies the home team is favored. For Phase 2 UI we expose a
    normalized `spread_home` where positive implies the home team is favored.

    Returns None when no line is available.
    """
    with SessionLocal() as sess:
        g = sess.execute(
            select(Game)
            .where(
                Game.home_id == int(home_team_id),
                Game.away_id == int(away_team_id),
                Game.closing_total.is_not(None),
            )
            .order_by(Game.season.desc(), func.coalesce(Game.week, 0).desc(), func.coalesce(Game.date, "").desc())
            .limit(1)
        ).scalar_one_or_none()
        if not g:
            return None
        spread_home = None
        if g.closing_spread is not None:
            # DB convention: negative = home favored. API convention: positive = home favored.
            spread_home = float(-g.closing_spread)
        total = float(g.closing_total) if g.closing_total is not None else None
        return {
            "season": int(g.season),
            "week": (int(g.week) if g.week is not None else None),
            "date": g.date,
            "spread_home": spread_home,
            "total": total,
            "source": "closing",
        }


@app.post("/simulate-series-by-name")
def simulate_series_by_name(req: NamesAutoSeriesIn):
    """
    Compatibility wrapper: expose the auto ratings series simulator under
    /simulate-series-by-name so that older UIs continue to work.
    """
    return simulate_series_by_name_auto(req)

@app.post("/simulate-series-by-name-auto")
def simulate_series_by_name_auto(req: NamesAutoSeriesIn):
    """Run many simulations using the consensus auto ratings pipeline."""
    home_r = combined_by_name(req.home_name)
    away_r = combined_by_name(req.away_name)
    if not home_r or not away_r:
        raise HTTPException(status_code=404, detail="Team not found in ratings. Run /update/ratings or check names.")

    hs = TeamState(
        name=home_r["name"],
        off_rush=float(home_r.get("off_rush", 50.0)),
        off_pass=float(home_r.get("off_pass", 50.0)),
        def_rush=float(home_r.get("def_rush", 50.0)),
        def_pass=float(home_r.get("def_pass", 50.0)),
        st=float(home_r.get("st", 0.0)),
        pace=float(home_r.get("pace", 60.0)),
    )
    as_ = TeamState(
        name=away_r["name"],
        off_rush=float(away_r.get("off_rush", 50.0)),
        off_pass=float(away_r.get("off_pass", 50.0)),
        def_rush=float(away_r.get("def_rush", 50.0)),
        def_pass=float(home_r.get("def_pass", 50.0)),
        st=float(away_r.get("st", 0.0)),
        pace=float(away_r.get("pace", 60.0)),
    )

    if req.deterministic:
        key = f"{req.home_name}|{req.away_name}|{req.n}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        base_seed = int(h[:8], 16)
    else:
        base_seed = None

    rng = random.Random(base_seed)
    hscores, ascores, ot, hw = [], [], 0, 0
    for _ in range(req.n):
        s = rng.randrange(0, 10_000_000)
        out = sim.sim_game(GameState(home=hs, away=as_), seed=s)
        hscores.append(out.score_home)
        ascores.append(out.score_away)
        if out.ot_periods > 0:
            ot += 1
        if out.score_home > out.score_away:
            hw += 1

    def q(arr, p):
        arr = sorted(arr)
        if not arr:
            return 0.0
        k = max(0, min(len(arr) - 1, int(round((p / 100.0) * (len(arr) - 1)))))
        return arr[k]

    
    mean_home = sum(hscores) / len(hscores) if hscores else 0.0
    mean_away = sum(ascores) / len(ascores) if ascores else 0.0
    spread_samples = [h - a for h, a in zip(hscores, ascores)]
    total_samples = [h + a for h, a in zip(hscores, ascores)]

    home_stats = {
        "name": hs.name,
        "avg_pts": mean_home,
        "median_pts": q(hscores, 50),
        "p05": q(hscores, 5),
        "p50": q(hscores, 50),
        "p95": q(hscores, 95),
    }
    away_stats = {
        "name": as_.name,
        "avg_pts": mean_away,
        "median_pts": q(ascores, 50),
        "p05": q(ascores, 5),
        "p50": q(ascores, 50),
        "p95": q(ascores, 95),
    }

    resp = {
        "meta": {
            "seasons_used": _completed_seasons(),
        },
        "samples": req.n,
        "home_stats": home_stats,
        "away_stats": away_stats,
        # for backwards compatibility also expose as `home`/`away`
        "home": home_stats,
        "away": away_stats,
        "home_win_pct": (hw / req.n) if req.n else 0.0,
        "away_win_pct": 1 - ((hw / req.n) if req.n else 0.0),
        "ot_rate": (ot / req.n) if req.n else 0.0,
        "mean_score_home": mean_home,
        "mean_score_away": mean_away,
        "mean_total": (mean_home + mean_away),
        "expected_spread": (mean_home - mean_away),
        "stdev_score_home": (
            (sum((x - mean_home) ** 2 for x in hscores) / len(hscores)) ** 0.5
            if hscores else 0.0
        ),
        "stdev_score_away": (
            (sum((x - mean_away) ** 2 for x in ascores) / len(ascores)) ** 0.5
            if ascores else 0.0
        ),
        "quantiles": {
            "home": {"p05": q(hscores, 5), "p50": q(hscores, 50), "p95": q(hscores, 95)},
            "away": {"p05": q(ascores, 5), "p50": q(ascores, 50), "p95": q(ascores, 95)},
            "spread": {
                "p05": q(spread_samples, 5),
                "p50": q(spread_samples, 50),
                "p95": q(spread_samples, 95),
            },
            "total": {
                "p05": q(total_samples, 5),
                "p50": q(total_samples, 50),
                "p95": q(total_samples, 95),
            },
        },
    }

    # ---------------- Phase 2: Model vs Market (Option A: edge = model - market) ----------------
    market = None
    try:
        hid = int(home_r.get("team_id")) if home_r.get("team_id") is not None else None
        aid = int(away_r.get("team_id")) if away_r.get("team_id") is not None else None
        if hid is not None and aid is not None:
            market = _lookup_closing_line(hid, aid)
    except Exception:
        market = None

    model_spread_home = float(resp.get("expected_spread", 0.0))
    model_total = float(resp.get("mean_total", 0.0))

    weights = {"model": 0.65, "market": 0.35}
    market_spread_home = (float(market["spread_home"]) if market and market.get("spread_home") is not None else None)
    market_total = (float(market["total"]) if market and market.get("total") is not None else None)

    blended_spread_home = (
        weights["model"] * model_spread_home + weights["market"] * market_spread_home
        if market_spread_home is not None else model_spread_home
    )
    blended_total = (
        weights["model"] * model_total + weights["market"] * market_total
        if market_total is not None else model_total
    )

    edge_spread = (model_spread_home - market_spread_home) if market_spread_home is not None else None
    edge_total = (model_total - market_total) if market_total is not None else None

    resp["market"] = market or {
        "spread_home": None,
        "total": None,
        "season": None,
        "week": None,
        "date": None,
        "source": "closing",
    }
    resp["blended"] = {
        "weights": weights,
        "spread_home": blended_spread_home,
        "total": blended_total,
    }
    resp["edge"] = {
        "spread_home": edge_spread,
        "total": edge_total,
        "definition": "edge = model - market",
    }

    if req.include_samples and req.n <= 2000:
        resp["samples_detail"] = {"home": hscores, "away": ascores}
    return resp

@app.get("/debug")
def debug():
    """
    Debug endpoint: shows which critical env vars are set and which DB URL
    the app is using. Secrets are masked.
    """
    import os
    from .db import DATABASE_URL
    env = {}
    for k in ("DATABASE_URL", "CFBD_API_KEY"):
        if k in os.environ:
            env[k] = "set"
        else:
            env[k] = None
    return {"ok": True, "env": env, "db_url": DATABASE_URL}


# ---------------- Backtesting (shipping gate) ----------------


class BacktestIn(BaseModel):
    test_season: int = 2023
    n_games: int = 200
    n_sims: int = 200
    seed: int = 1337
    label: str | None = None


@app.get("/backtest/baseline")
def backtest_get_baseline(test_season: int = 2023) -> dict:
    """Fetch the stored baseline metrics for a test season."""
    with SessionLocal() as sess:
        row = (
            sess.execute(
                select(ModelEval)
                .where(ModelEval.test_season == int(test_season), ModelEval.is_baseline == 1)
                .order_by(func.coalesce(ModelEval.created_at, "").desc())
                .limit(1)
            )
            .scalars()
            .first()
        )
        if not row:
            return {"ok": True, "baseline": None}
        return {
            "ok": True,
            "baseline": {
                "id": row.id,
                "created_at": row.created_at,
                "label": row.label,
                "test_season": row.test_season,
                "n_games": row.n_games,
                "n_sims": row.n_sims,
                "spread_mae": row.spread_mae,
                "total_mae": row.total_mae,
                "brier": row.brier,
            },
        }


@app.post("/backtest/set-baseline")
def backtest_set_baseline(req: BacktestIn) -> dict:
    """Run a backtest and store it as the baseline for that season."""
    metrics = _run_backtest(
        test_season=req.test_season,
        n_games=req.n_games,
        n_sims=req.n_sims,
        seed=req.seed,
        require_lines=True,
    )

    # If we can't compute core metrics, don't silently store bad baselines.
    if metrics.get("spread_mae") is None or metrics.get("total_mae") is None or metrics.get("brier") is None:
        raise HTTPException(
            status_code=400,
            detail="Backtest could not compute spread_mae/total_mae/brier. Ensure the season has games with closing spread AND total lines.",
        )
    label = (req.label or f"baseline {req.test_season}").strip()
    with SessionLocal() as sess:
        # Clear existing baseline for that season.
        sess.query(ModelEval).filter(
            ModelEval.test_season == int(req.test_season), ModelEval.is_baseline == 1
        ).update({ModelEval.is_baseline: 0})
        row = ModelEval(
            created_at=_now_iso(),
            label=label,
            is_baseline=1,
            test_season=int(metrics["test_season"]),
            n_games=int(metrics["n_games"]),
            n_sims=int(metrics["n_sims"]),
            spread_mae=float(metrics["spread_mae"]),
            total_mae=float(metrics["total_mae"]),
            brier=float(metrics["brier"]),
        )
        sess.add(row)
        sess.commit()
    return {"ok": True, "baseline": metrics}


@app.post("/backtest/run")
def backtest_run(req: BacktestIn) -> dict:
    """Run a candidate backtest and compare it to the stored baseline.

    Returns verdict PASS/EVEN/FAIL. The app reports; you decide to ship.
    """
    metrics = _run_backtest(
        test_season=req.test_season,
        n_games=req.n_games,
        n_sims=req.n_sims,
        seed=req.seed,
        require_lines=True,
    )

    baseline = None
    with SessionLocal() as sess:
        baseline = (
            sess.execute(
                select(ModelEval)
                .where(ModelEval.test_season == int(req.test_season), ModelEval.is_baseline == 1)
                .order_by(func.coalesce(ModelEval.created_at, "").desc())
                .limit(1)
            )
            .scalars()
            .first()
        )

        # Store the candidate run for history.
        sess.add(
            ModelEval(
                created_at=_now_iso(),
                label=(req.label or "candidate"),
                is_baseline=0,
                test_season=int(metrics["test_season"]),
                n_games=int(metrics["n_games"]),
                n_sims=int(metrics["n_sims"]),
                spread_mae=float(metrics["spread_mae"]),
                total_mae=float(metrics["total_mae"]),
                brier=float(metrics["brier"]),
            )
        )
        sess.commit()

    if not baseline:
        return {
            "ok": True,
            "verdict": "NO_BASELINE",
            "reason": "No baseline stored yet. Run /backtest/set-baseline first.",
            "baseline": None,
            # Candidate metrics (kept for backwards-compatibility)
            "candidate": metrics,
            # Also expose metrics at the top-level for the UI
            "metrics": metrics,
            **metrics,
        }

    delta_spread = float(baseline.spread_mae - metrics["spread_mae"])  # positive = candidate better
    delta_total = float(baseline.total_mae - metrics["total_mae"])
    delta_brier = float(baseline.brier - metrics["brier"])

    thr = _backtest_thresholds()

    improved = (
        (delta_spread >= thr["spread_improve"])
        or (delta_total >= thr["total_improve"])
        or (delta_brier >= thr["brier_improve"])
    )
    regressed = (
        (delta_spread <= -thr["spread_regress"])
        or (delta_total <= -thr["total_regress"])
        or (delta_brier <= -thr["brier_regress"])
    )

    if improved and not regressed:
        verdict = "PASS"
        reason = "Candidate improves at least one primary metric with no material regressions."
    elif regressed and not improved:
        verdict = "FAIL"
        reason = "Candidate materially regresses at least one primary metric."
    else:
        verdict = "EVEN"
        reason = "Changes are within noise or mixed (improve some while regressing others)."

    return {
        "ok": True,
        "verdict": verdict,
        "reason": reason,
        "thresholds": thr,
        "baseline": {
            "created_at": baseline.created_at,
            "label": baseline.label,
            "test_season": baseline.test_season,
            "n_games": baseline.n_games,
            "n_sims": baseline.n_sims,
            "spread_mae": baseline.spread_mae,
            "total_mae": baseline.total_mae,
            "brier": baseline.brier,
        },
        "candidate": metrics,
        "metrics": metrics,
        **metrics,
"candidate": metrics,
        "delta": {
            "spread_mae": delta_spread,
            "total_mae": delta_total,
            "brier": delta_brier,
            "definition": "delta = baseline - candidate (positive means candidate is better)",
        },
    }
