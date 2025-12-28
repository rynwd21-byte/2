import random
from dataclasses import dataclass
from typing import Dict, Optional

from .features import DriveContext, to_features
from .model_params import get_param

RESULTS = ["TD", "FG", "PUNT", "TO", "DOWNS", "ENDHALF"]


@dataclass
class TeamState:
    name: str
    off_rush: float
    off_pass: float
    def_rush: float
    def_pass: float
    st: float
    # Approximate pace on a 0–100-ish scale; 60 ~= neutral.
    pace: float = 60.0


@dataclass
class GameState:
    home: TeamState
    away: TeamState
    seconds_left: int = 3600
    score_home: int = 0
    score_away: int = 0
    possession: str = "home"
    ot_periods: int = 0
    drives_simmed: int = 0  # regulation drive counter for pace cap


class DriveModel:
    def __init__(self, rng: Optional[random.Random] = None):
        # Base drive coefficients (same shape as feature vector)
        self.base_coef = [0.6, 0.1, -0.05, 0.25, 0.25, 0.05, 0.02, -0.02]
        self.coef_scale = get_param("coef_scale", 1.0)
        self._refresh()
        self.rng: random.Random = rng or random.Random()

    def _refresh(self) -> None:
        s = float(self.coef_scale or 1.0)
        self.coef = [c * s for c in self.base_coef]

    def probs(self, x):
        # Simple linear index into a softmax-ish distribution.
        z = sum(c * v for c, v in zip(self.coef, x))
        base = {
            "TD": 0.18 + 0.20 * z,
            "FG": 0.10 + 0.05 * z,
            "PUNT": 0.52 - 0.30 * z,
            "TO": 0.08 - 0.02 * z,
            "DOWNS": 0.07 - 0.02 * z,
            "ENDHALF": 0.05 - 0.01 * z,
        }
        total = sum(max(0.001, v) for v in base.values())
        return {k: max(0.001, v) / total for k, v in base.items()}

    def sample(self, probs: Dict[str, float]) -> str:
        r = self.rng.random()
        cum = 0.0
        for k in RESULTS:
            cum += probs[k]
            if r <= cum:
                return k
        return RESULTS[-1]

    def _xp_good(self, offense: TeamState, defense: TeamState) -> bool:
        """Simple XP model, lightly influenced by special teams."""
        base = 0.93
        adj = 0.002 * ((offense.st or 0.0) - (defense.st or 0.0))
        p = max(0.85, min(0.99, base + adj))
        return self.rng.random() < p

    def _two_point_good(self, offense: TeamState, defense: TeamState) -> bool:
        base = 0.45
        adj = 0.002 * (
            (offense.off_rush + offense.off_pass) - (defense.def_rush + defense.def_pass)
        ) / 100.0
        p = max(0.30, min(0.60, base + adj))
        return self.rng.random() < p


class Simulator:
    def __init__(self, rng: Optional[random.Random] = None):
        # Base RNG used to seed per-game RNGs when no explicit seed is provided.
        self._base_rng: random.Random = rng or random.Random()
        self.model = DriveModel(rng=self._base_rng)

    def set_coef_scale(self, scale: float) -> None:
        self.model.coef_scale = scale
        self.model._refresh()

    def _drive_seconds(self, gs: GameState, offense: TeamState, defense: TeamState, result: str) -> int:
        base = {
            "TD": 180,
            "FG": 150,
            "PUNT": 180,
            "TO": 130,
            "DOWNS": 140,
            "ENDHALF": gs.seconds_left,
        }
        off_pace = offense.pace or 60.0
        def_pace = defense.pace or 60.0
        avg_pace = max(30.0, min(90.0, (off_pace + def_pace) / 2.0))
        # Faster pace (higher value) => more plays => smaller time per drive.
        pace_scale = 60.0 / avg_pace  # >1 = slower, <1 = faster
        sec = int(base.get(result, 150) * pace_scale)
        sec = max(10, min(gs.seconds_left, sec))
        return sec

    def sim_game(self, gs: GameState, seed: Optional[int] = None) -> GameState:
        """Simulate a single game.

        Uses a per-game RNG so concurrent requests do not interfere and
        deterministic seeds are truly reproducible.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random(self._base_rng.randrange(0, 2**32 - 1))

        # Temporarily point the drive model at this RNG.
        old_rng = self.model.rng
        self.model.rng = rng
        # ---------------- Pace -> expected drive count -----------------
        # We are primarily clock-based (3600s). Pace already shrinks/expands
        # time per drive, but in practice the distribution of outcomes can
        # still under/over-shoot possessions for extreme pace teams.
        #
        # Add a soft cap on the number of regulation drives based on the
        # average pace of the two teams. This makes fast matchups naturally
        # generate more possessions, and slow matchups fewer, even if drive
        # durations vary.
        base_total_drives = float(get_param("base_total_drives", 24.0))
        off_pace = gs.home.pace or 60.0
        def_pace = gs.away.pace or 60.0
        avg_pace = max(30.0, min(90.0, (off_pace + def_pace) / 2.0))
        expected_total_drives = base_total_drives * (avg_pace / 60.0)
        max_drives = int(max(10.0, min(40.0, expected_total_drives)))

        try:
            while gs.seconds_left > 0 and gs.drives_simmed < max_drives:
                gs = self._simulate_one_drive(gs)

            # If we hit our pace-based possession cap before the clock is out,
            # treat the remainder as run-out time.
            if gs.seconds_left > 0 and gs.drives_simmed >= max_drives:
                gs.seconds_left = 0
            if gs.score_home == gs.score_away:
                self._simulate_overtime(gs, rng)
            return gs
        finally:
            self.model.rng = old_rng

    def _simulate_one_drive(self, gs: GameState) -> GameState:
        offense = gs.home if gs.possession == "home" else gs.away
        defense = gs.away if gs.possession == "home" else gs.home
        score_diff = (
            gs.score_home - gs.score_away
            if gs.possession == "home"
            else gs.score_away - gs.score_home
        )
        ctx = DriveContext(
            yardline=75,
            seconds_left=gs.seconds_left,
            score_diff=score_diff,
            off_rush=offense.off_rush,
            off_pass=offense.off_pass,
            def_rush=defense.def_rush,
            def_pass=defense.def_pass,
            st=offense.st,
            timeouts_off=3,
            timeouts_def=3,
        )
        probs = self.model.probs(to_features(ctx))
        result = self.model.sample(probs)

        # Track regulation drive count for pace-based possession cap.
        gs.drives_simmed += 1

        # scoring
        if result == "TD":
            if gs.possession == "home":
                gs.score_home += 6
                if self.model._xp_good(gs.home, gs.away):
                    gs.score_home += 1
            else:
                gs.score_away += 6
                if self.model._xp_good(gs.away, gs.home):
                    gs.score_away += 1
        elif result == "FG":
            if gs.possession == "home":
                gs.score_home += 3
            else:
                gs.score_away += 3

        # clock
        elapsed = self._drive_seconds(gs, offense, defense, result)
        gs.seconds_left -= elapsed

        # possession switch / half end
        if result != "ENDHALF":
            gs.possession = "away" if gs.possession == "home" else "home"
        else:
            gs.seconds_left = 0

        return gs

    def _simulate_overtime(self, gs: GameState, rng: random.Random) -> None:
        """Simple college-style overtime."""
        max_periods = 20
        start = "home"
        while gs.score_home == gs.score_away and gs.ot_periods < max_periods:
            gs.ot_periods += 1
            order = (start, "away" if start == "home" else "home")
            for side in order:
                offense = gs.home if side == "home" else gs.away
                defense = gs.away if side == "home" else gs.home
                pts = self._overtime_possession(offense, defense)
                if side == "home":
                    gs.score_home += pts
                else:
                    gs.score_away += pts
            if gs.score_home != gs.score_away:
                break
            start = "away" if start == "home" else "home"

        if gs.score_home == gs.score_away:
            # Sudden-death style coin flip if still tied after many OTs.
            if rng.random() < 0.5:
                gs.score_home += 2
            else:
                gs.score_away += 2

    def _overtime_possession(self, offense: TeamState, defense: TeamState) -> int:
        # Use the same drive model but ignore the real clock.
        ctx = DriveContext(
            yardline=25,
            seconds_left=600,
            score_diff=0,
            off_rush=offense.off_rush,
            off_pass=offense.off_pass,
            def_rush=defense.def_rush,
            def_pass=defense.def_pass,
            st=offense.st,
            timeouts_off=1,
            timeouts_def=1,
        )
        probs = self.model.probs(to_features(ctx))
        result = self.model.sample(probs)
        if result == "TD":
            pts = 6
            # Mix of XP and 2-pt behavior.
            if self.model._two_point_good(offense, defense):
                pts += 2
            else:
                if self.model._xp_good(offense, defense):
                    pts += 1
            return pts
        elif result == "FG":
            return 3
        else:
            return 0