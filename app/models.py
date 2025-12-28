from sqlalchemy import Column, Integer, String, Float, ForeignKey, Index, UniqueConstraint
from .db import Base
class Team(Base):
    __tablename__ = "teams"
    __table_args__ = (Index("ix_teams_name", "name"),)
    team_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    conference = Column(String)
    elo = Column(Float, default=1500)
    off_rush = Column(Float, default=0)
    off_pass = Column(Float, default=0)
    def_rush = Column(Float, default=0)
    def_pass = Column(Float, default=0)
    st = Column(Float, default=0)
    last_updated = Column(String)
class Game(Base):
    __tablename__ = "games"
    game_id = Column(Integer, primary_key=True)
    season = Column(Integer, nullable=False)
    week = Column(Integer)
    date = Column(String)
    neutral = Column(Integer, default=0)
    home_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    home_pts = Column(Integer)
    away_pts = Column(Integer)
    ot_periods = Column(Integer, default=0)
    # Regular vs postseason flag (mirrors CFBD's seasonType)
    season_type = Column(String)  # e.g., 'regular', 'postseason'
    # Vegas market information (from home perspective)
    closing_spread = Column(Float)  # home - away spread (negative if home is favored)
    closing_total = Column(Float)   # closing over/under total points


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        Index("ix_ratings_team_id", "team_id"),
        Index("ix_ratings_team_source", "team_id", "source"),
    )
    rating_id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    source = Column(String, nullable=False)  # e.g., 'cfbd_ppa_off', 'cfbd_ppa_def', 'market'
    off_rush = Column(Float)
    off_pass = Column(Float)
    def_rush = Column(Float)
    def_pass = Column(Float)
    st = Column(Float)
    pace = Column(Float)
    spread = Column(Float)
    total = Column(Float)
    updated_at = Column(String)


class Drive(Base):
    """Drive-level summaries used for pace normalization.

    Source: CFBD `/drives` endpoint.
    Times are stored as (minutes, seconds) remaining in the period.
    """

    __tablename__ = "drives"
    __table_args__ = (
        Index("ix_drives_game_id", "game_id"),
        Index("ix_drives_season_team", "season", "offense_id"),
        Index("ix_drives_season_def", "season", "defense_id"),
        UniqueConstraint("drive_uid", name="uq_drives_drive_uid"),
    )

    # CFBD provides an `id` for drives; we store it as a unique UID.
    drive_id = Column(Integer, primary_key=True, autoincrement=True)
    drive_uid = Column(Integer, nullable=False)

    game_id = Column(Integer, ForeignKey("games.game_id"), nullable=False)
    season = Column(Integer, nullable=False)
    week = Column(Integer)
    season_type = Column(String)

    offense_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    defense_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    offense_name = Column(String)
    defense_name = Column(String)

    drive_number = Column(Integer)
    drive_result = Column(String)
    scoring = Column(Integer, default=0)

    start_period = Column(Integer)
    start_minutes = Column(Integer)
    start_seconds = Column(Integer)
    end_period = Column(Integer)
    end_minutes = Column(Integer)
    end_seconds = Column(Integer)

    plays = Column(Integer)
    yards = Column(Integer)


class TeamSeasonPace(Base):
    """Materialized pace metrics per team-season."""

    __tablename__ = "team_season_pace"
    __table_args__ = (
        UniqueConstraint("season", "team_id", name="uq_tsp_season_team"),
        Index("ix_tsp_team", "team_id"),
    )

    id = Column(Integer, primary_key=True)
    season = Column(Integer, nullable=False)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    games_played = Column(Integer, nullable=False)
    drives_for = Column(Integer, nullable=False)
    drives_against = Column(Integer, nullable=False)
    drives_for_per_game = Column(Float)
    drives_against_per_game = Column(Float)
    sec_per_drive_for = Column(Float)
    sec_per_drive_against = Column(Float)
    sec_per_drive = Column(Float)
    pace_rating = Column(Float)
    updated_at = Column(String)


class ModelEval(Base):
    """Stored evaluation metrics for a given test season.

    We use these to compare the currently deployed "candidate" model against
    a frozen baseline. The app reports PASS/EVEN/FAIL; you decide whether to
    ship/promote.
    """

    __tablename__ = "model_evals"
    __table_args__ = (
        Index("ix_model_evals_season", "test_season"),
        Index("ix_model_evals_season_baseline", "test_season", "is_baseline"),
    )

    id = Column(Integer, primary_key=True)
    created_at = Column(String)
    label = Column(String)  # e.g. 'baseline v1.0', 'candidate'
    is_baseline = Column(Integer, default=0)  # 1=true
    test_season = Column(Integer, nullable=False)
    n_games = Column(Integer, nullable=False)
    n_sims = Column(Integer, nullable=False)
    spread_mae = Column(Float)
    total_mae = Column(Float)
    brier = Column(Float)

