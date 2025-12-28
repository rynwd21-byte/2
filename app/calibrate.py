from __future__ import annotations
import random
from statistics import mean
from typing import Optional
from sqlalchemy import select
from .db import SessionLocal
from .models import Game, Team
from .sim_engine import Simulator, TeamState, GameState
from .model_params import set_param

def league_ppg(season: int) -> Optional[float]:
    with SessionLocal() as sess:
        rows = sess.execute(select(Game).where(Game.season == season)).scalars().all()
        pts = []
        for g in rows:
            if g.home_pts is not None and g.away_pts is not None:
                pts.extend([g.home_pts, g.away_pts])
        if not pts: return None
        return mean(pts)

def sample_random_matchups(k: int):
    with SessionLocal() as sess:
        teams = sess.execute(select(Team)).scalars().all()
        if len(teams) < 2: return []
        import random as R
        out = []
        for _ in range(k):
            a, b = R.sample(teams, 2)
            a_state = TeamState(name=a.name, off_rush=a.off_rush or 0, off_pass=a.off_pass or 0,
                                def_rush=a.def_rush or 0, def_pass=a.def_pass or 0, st=a.st or 0)
            b_state = TeamState(name=b.name, off_rush=b.off_rush or 0, off_pass=b.off_pass or 0,
                                def_rush=b.def_rush or 0, def_pass=b.def_pass or 0, st=b.st or 0)
            out.append((a_state, b_state))
        return out

def calibrate_coef_scale(target_ppg: float, samples: int = 2000, seed: int | None = None) -> float:
    rng = random.Random(seed)
    sim = Simulator()
    def eval_scale(scale: float, trials: int = 400) -> float:
        sim.set_coef_scale(scale)
        m = sample_random_matchups(trials)
        if not m: return 0.0
        pts = []
        for (h, a) in m:
            s = rng.randrange(0, 10_000_000)
            out = sim.sim_game(GameState(home=h, away=a), seed=s)
            pts.extend([out.score_home, out.score_away])
        return (sum(pts) / len(pts)) if pts else 0.0
    lo, hi = 0.5, 2.0; best_scale, best_err = 1.0, float("inf")
    for _ in range(12):
        mid = (lo + hi) / 2; ppg = eval_scale(mid)
        err = abs(ppg - target_ppg)
        if err < best_err: best_err, best_scale = err, mid
        if ppg < target_ppg: lo = mid
        else: hi = mid
    set_param("coef_scale", float(best_scale))
    return float(best_scale)

def run(season: int, samples: int = 2000, seed: int | None = None) -> dict:
    target = league_ppg(season)
    if target is None:
        return {"ok": False, "error": "No completed games in DB for that season."}
    scale = calibrate_coef_scale(target_ppg=target, samples=samples, seed=seed)
    return {"ok": True, "target_ppg": target, "coef_scale": scale}
