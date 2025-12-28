from dataclasses import dataclass
@dataclass
class DriveContext:
    yardline: int
    seconds_left: int
    score_diff: int
    off_rush: float
    off_pass: float
    def_rush: float
    def_pass: float
    st: float
    timeouts_off: int
    timeouts_def: int
def to_features(ctx: DriveContext):
    return [ctx.yardline/100.0, ctx.seconds_left/3600.0, max(-50, min(50, ctx.score_diff))/50.0,
            (ctx.off_rush-ctx.def_rush)/200.0, (ctx.off_pass-ctx.def_pass)/200.0, ctx.st/100.0,
            ctx.timeouts_off/3.0, ctx.timeouts_def/3.0]
