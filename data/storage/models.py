from sqlalchemy import (
    Column, Integer, String, Boolean, 
    Numeric, Date, DateTime, ForeignKey, Text, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Team(Base):
    __tablename__ = "teams"

    team_id         = Column(Integer, primary_key=True)
    nba_team_id     = Column(Integer, unique=True, nullable=False)
    abbreviation    = Column(String(3), unique=True, nullable=False)
    full_name       = Column(String(100), nullable=False)
    city            = Column(String(100))
    conference      = Column(String(4))
    division        = Column(String(20))
    arena           = Column(String(100))
    arena_state     = Column(String(50))
    altitude_ft     = Column(Integer, default=0)
    created_at      = Column(DateTime, default=func.now())
    updated_at      = Column(DateTime, default=func.now())


class Player(Base):
    __tablename__ = "players"

    player_id       = Column(Integer, primary_key=True)
    nba_player_id   = Column(Integer, unique=True, nullable=False)
    full_name       = Column(String(150), nullable=False)
    first_name      = Column(String(75))
    last_name       = Column(String(75))
    position        = Column(String(30))
    height_inches   = Column(Integer)
    weight_lbs      = Column(Integer)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=func.now())
    updated_at      = Column(DateTime, default=func.now())


class Game(Base):
    __tablename__ = "games"

    game_id         = Column(Integer, primary_key=True)
    nba_game_id     = Column(String(20), unique=True, nullable=False)
    season          = Column(String(10), nullable=False)
    season_type     = Column(String(20))
    game_date       = Column(Date, nullable=False)
    tipoff_time_utc = Column(DateTime)
    home_team_id    = Column(Integer, ForeignKey("teams.team_id"))
    away_team_id    = Column(Integer, ForeignKey("teams.team_id"))
    playoff_round   = Column(String(30))
    series_game_num = Column(Integer)
    home_series_wins = Column(Integer)
    away_series_wins = Column(Integer)
    is_elimination  = Column(Boolean, default=False)
    home_score      = Column(Integer)
    away_score      = Column(Integer)
    is_final        = Column(Boolean, default=False)
    status          = Column(String(20), default="scheduled")
    created_at      = Column(DateTime, default=func.now())
    updated_at      = Column(DateTime, default=func.now())
    series_id = Column(String(50), nullable=True)
    series_home_wins = Column(Integer, nullable=True)
    series_away_wins = Column(Integer, nullable=True)


class PlayerGameStatus(Base):
    __tablename__ = "player_game_status"

    id              = Column(Integer, primary_key=True)
    game_id         = Column(Integer, ForeignKey("games.game_id"))
    player_id       = Column(Integer, ForeignKey("players.player_id"))
    team_id         = Column(Integer, ForeignKey("teams.team_id"))
    status          = Column(String(20))
    did_play        = Column(Boolean)
    dnp_reason      = Column(String(100))
    status_reported_at = Column(DateTime)
    is_confirmed    = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=func.now())
    updated_at      = Column(DateTime, default=func.now())


class PlayerBoxScore(Base):
    __tablename__ = "player_box_scores"

    id              = Column(Integer, primary_key=True)
    game_id         = Column(Integer, ForeignKey("games.game_id"))
    player_id       = Column(Integer, ForeignKey("players.player_id"))
    team_id         = Column(Integer, ForeignKey("teams.team_id"))
    minutes_played  = Column(Numeric(5, 2))
    points          = Column(Integer)
    rebounds        = Column(Integer)
    assists         = Column(Integer)
    steals          = Column(Integer)
    blocks          = Column(Integer)
    turnovers       = Column(Integer)
    fouls           = Column(Integer)
    fgm             = Column(Integer)
    fga             = Column(Integer)
    fg_pct          = Column(Numeric(5, 4))
    fg3m            = Column(Integer)
    fg3a            = Column(Integer)
    fg3_pct         = Column(Numeric(5, 4))
    ftm             = Column(Integer)
    fta             = Column(Integer)
    ft_pct          = Column(Numeric(5, 4))
    plus_minus      = Column(Integer)
    usage_rate      = Column(Numeric(5, 2))
    true_shooting   = Column(Numeric(5, 4))
    created_at      = Column(DateTime, default=func.now())
    oreb       = Column(Integer)
    dreb       = Column(Integer)
    efg_pct    = Column(Numeric(5, 4))
    oreb_pct   = Column(Numeric(5, 4))
    dreb_pct   = Column(Numeric(5, 4))
    ast_pct    = Column(Numeric(5, 4))
    tov_pct    = Column(Numeric(5, 4))
    blk_pct    = Column(Numeric(5, 4))
    stl_pct    = Column(Numeric(5, 4))
    off_rating = Column(Numeric(6, 2))
    def_rating = Column(Numeric(6, 2))
    net_rating = Column(Numeric(6, 2))
    pace       = Column(Numeric(6, 2))


class TeamBoxScore(Base):
    __tablename__ = "team_box_scores"

    id              = Column(Integer, primary_key=True)
    game_id         = Column(Integer, ForeignKey("games.game_id"))
    team_id         = Column(Integer, ForeignKey("teams.team_id"))
    is_home         = Column(Boolean)
    points          = Column(Integer)
    rebounds        = Column(Integer)
    assists         = Column(Integer)
    steals          = Column(Integer)
    blocks          = Column(Integer)
    turnovers       = Column(Integer)
    fgm             = Column(Integer)
    fga             = Column(Integer)
    fg_pct          = Column(Numeric(5, 4))
    fg3m            = Column(Integer)
    fg3a            = Column(Integer)
    fg3_pct         = Column(Numeric(5, 4))
    ftm             = Column(Integer)
    fta             = Column(Integer)
    ft_pct          = Column(Numeric(5, 4))
    offensive_rating = Column(Numeric(6, 2))
    defensive_rating = Column(Numeric(6, 2))
    net_rating       = Column(Numeric(6, 2))
    pace             = Column(Numeric(6, 2))
    true_shooting    = Column(Numeric(5, 4))
    efg_pct          = Column(Numeric(5, 4))
    tov_pct          = Column(Numeric(6, 2))   # was Numeric(5,4)
    oreb_pct         = Column(Numeric(6, 2))   # was Numeric(5,4)
    ft_rate          = Column(Numeric(6, 2))   # was Numeric(5,4)
    paint_points        = Column(Integer)
    fastbreak_points    = Column(Integer)
    second_chance_pts   = Column(Integer)
    created_at          = Column(DateTime, default=func.now())
    q1_points  = Column(Integer)
    q2_points  = Column(Integer)
    q3_points  = Column(Integer)
    q4_points  = Column(Integer)


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id       = Column(Integer, primary_key=True)
    game_id             = Column(Integer, ForeignKey("games.game_id"))
    model_version       = Column(String(20), nullable=False)
    predicted_at        = Column(DateTime, default=func.now())
    lineup_confirmed    = Column(Boolean, default=False)
    home_score_predicted = Column(Numeric(6, 2))
    away_score_predicted = Column(Numeric(6, 2))
    home_score_low      = Column(Numeric(6, 2))
    home_score_high     = Column(Numeric(6, 2))
    away_score_low      = Column(Numeric(6, 2))
    away_score_high     = Column(Numeric(6, 2))
    predicted_winner_id = Column(Integer, ForeignKey("teams.team_id"))
    home_win_probability = Column(Numeric(5, 4))
    confidence_score    = Column(Numeric(5, 4))
    uncertainty_flag    = Column(Boolean, default=False)
    key_factors         = Column(JSONB)
    feature_vector      = Column(JSONB)
    home_score_actual   = Column(Integer)
    away_score_actual   = Column(Integer)
    actual_winner_id    = Column(Integer, ForeignKey("teams.team_id"))
    winner_correct      = Column(Boolean)
    home_score_error    = Column(Numeric(6, 2))
    away_score_error    = Column(Numeric(6, 2))
    total_score_error   = Column(Numeric(6, 2))
    evaluated_at        = Column(DateTime)




class TeamRollingStats(Base):
    __tablename__ = "team_rolling_stats"

    id              = Column(Integer, primary_key=True)
    team_id         = Column(Integer, ForeignKey("teams.team_id"))
    as_of_date      = Column(Date, nullable=False)
    window          = Column(Integer, nullable=False)
    season          = Column(String(10))

    avg_points              = Column(Numeric(6, 2))
    avg_offensive_rating    = Column(Numeric(6, 2))
    avg_pace                = Column(Numeric(6, 2))
    avg_fg_pct              = Column(Numeric(5, 4))
    avg_fg3_pct             = Column(Numeric(5, 4))
    avg_ft_rate             = Column(Numeric(6, 2))
    three_point_rate        = Column(Numeric(5, 4))
    avg_points_allowed      = Column(Numeric(6, 2))
    avg_defensive_rating    = Column(Numeric(6, 2))
    avg_opp_fg_pct          = Column(Numeric(5, 4))
    avg_opp_fg3_pct         = Column(Numeric(5, 4))
    home_avg_points         = Column(Numeric(6, 2))
    away_avg_points         = Column(Numeric(6, 2))
    home_avg_points_allowed = Column(Numeric(6, 2))
    away_avg_points_allowed = Column(Numeric(6, 2))
    wins                    = Column(Integer)
    losses                  = Column(Integer)
    win_pct                 = Column(Numeric(5, 4))
    games_counted           = Column(Integer)
    created_at              = Column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint('team_id', 'as_of_date', 'window',
                         name='uq_team_rolling_stats'),
    )


class GameOdds(Base):
    __tablename__ = "game_odds"

    id                 = Column(Integer, primary_key=True)
    game_id            = Column(Integer, ForeignKey("games.game_id"))
    fetched_at         = Column(DateTime, default=func.now())
    home_spread        = Column(Numeric(5, 1))
    away_spread        = Column(Numeric(5, 1))
    total_line         = Column(Numeric(5, 1))
    vegas_home_implied = Column(Numeric(6, 2))
    vegas_away_implied = Column(Numeric(6, 2))
    bookmaker          = Column(String(50))


class TeamStyleVector(Base):
    __tablename__ = "team_style_vectors"

    id                  = Column(Integer, primary_key=True)
    team_id             = Column(Integer, ForeignKey("teams.team_id"))
    as_of_date          = Column(Date, nullable=False)
    season              = Column(String(10))
    avg_pace            = Column(Numeric(6, 2))
    three_point_rate    = Column(Numeric(5, 4))
    offensive_rating    = Column(Numeric(6, 2))
    defensive_rating    = Column(Numeric(6, 2))
    efg_pct             = Column(Numeric(5, 4))
    tov_pct             = Column(Numeric(6, 2))
    oreb_pct            = Column(Numeric(6, 2))
    ft_rate             = Column(Numeric(6, 2))
    fg3_pct             = Column(Numeric(5, 4))
    created_at          = Column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint('team_id', 'as_of_date',
                         name='uq_team_style_vector'),
    )


class TeamSimilarity(Base):
    __tablename__ = "team_similarity"

    id               = Column(Integer, primary_key=True)
    team_a_id        = Column(Integer, ForeignKey("teams.team_id"))
    team_b_id        = Column(Integer, ForeignKey("teams.team_id"))
    as_of_date       = Column(Date, nullable=False)
    similarity_score = Column(Numeric(8, 6))
    created_at       = Column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint('team_a_id', 'team_b_id', 'as_of_date',
                         name='uq_team_similarity'),
    )



class PlayerRollingStats(Base):
    __tablename__ = "player_rolling_stats"

    id               = Column(Integer, primary_key=True)
    player_id        = Column(Integer, ForeignKey("players.player_id"))
    team_id          = Column(Integer, ForeignKey("teams.team_id"))
    as_of_date       = Column(Date, nullable=False)
    window           = Column(Integer, nullable=False)
    season           = Column(String(10))
    avg_points       = Column(Numeric(6, 2))
    avg_minutes      = Column(Numeric(5, 2))
    avg_usage_rate   = Column(Numeric(5, 4))
    avg_fg_pct       = Column(Numeric(5, 4))
    avg_fg3_pct      = Column(Numeric(5, 4))
    avg_true_shooting = Column(Numeric(5, 4))
    avg_rebounds     = Column(Numeric(5, 2))
    avg_assists      = Column(Numeric(5, 2))
    avg_steals       = Column(Numeric(5, 2))
    avg_blocks       = Column(Numeric(5, 2))
    avg_turnovers    = Column(Numeric(5, 2))
    avg_plus_minus   = Column(Numeric(5, 2))
    games_counted    = Column(Integer)
    avg_efg_pct    = Column(Numeric(5, 4))
    avg_oreb_pct   = Column(Numeric(5, 4))
    avg_dreb_pct   = Column(Numeric(5, 4))
    avg_ast_pct    = Column(Numeric(5, 4))
    avg_off_rating = Column(Numeric(6, 2))
    avg_def_rating = Column(Numeric(6, 2))
    avg_net_rating = Column(Numeric(6, 2))
    avg_blk_pct    = Column(Numeric(5, 4))
    avg_stl_pct    = Column(Numeric(5, 4))
    created_at       = Column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint(
            'player_id', 'team_id', 'as_of_date', 'window',
            name='uq_player_rolling_stats'
        ),
    )