import sys
import os
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from data.storage.db import engine


CSV_PATH = r"C:\Users\mouni\OneDrive\Desktop\NBA-Predictor\logs\predictions_log.csv"


CREATE_STAGING_SQL = """
DROP TABLE IF EXISTS prediction_log_staging;

CREATE TABLE prediction_log_staging (
    row_id BIGSERIAL PRIMARY KEY,

    game_date DATE,
    saved_at TIMESTAMP NULL,
    model_version TEXT NULL,
    game_id INTEGER NULL,

    home_team TEXT,
    away_team TEXT,

    home_pred NUMERIC,
    away_pred NUMERIC,
    predicted_winner TEXT,
    margin NUMERIC,
    confidence TEXT,

    home_pred_team NUMERIC,
    away_pred_team NUMERIC,
    home_pred_player NUMERIC,
    away_pred_player NUMERIC,

    is_playoff BOOLEAN,
    season_type TEXT
);
"""


INSERT_ROW_SQL = """
INSERT INTO prediction_log_staging (
    game_date,
    saved_at,
    model_version,
    game_id,
    home_team,
    away_team,
    home_pred,
    away_pred,
    predicted_winner,
    margin,
    confidence,
    home_pred_team,
    away_pred_team,
    home_pred_player,
    away_pred_player,
    is_playoff,
    season_type
)
VALUES (
    :game_date,
    :saved_at,
    :model_version,
    :game_id,
    :home_team,
    :away_team,
    :home_pred,
    :away_pred,
    :predicted_winner,
    :margin,
    :confidence,
    :home_pred_team,
    :away_pred_team,
    :home_pred_player,
    :away_pred_player,
    :is_playoff,
    :season_type
);
"""


def clean(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value != "" else None


def to_float(value):
    value = clean(value)
    if value is None:
        return None
    return float(value)


def to_int(value):
    value = clean(value)
    if value is None:
        return None
    return int(float(value))


def to_bool(value):
    value = clean(value)
    if value is None:
        return None
    return value.lower() in {"true", "1", "yes", "y"}


def parse_row(row):
    """
    Supports both CSV formats.

    Old:
    0 date
    1 home_team
    2 away_team
    3 home_pred
    4 away_pred
    5 predicted_winner
    6 margin
    7 confidence
    8 home_pred_team
    9 away_pred_team
    10 home_pred_player
    11 away_pred_player
    12 is_playoff
    13 season_type

    New:
    0 date
    1 saved_at
    2 model_version
    3 game_id
    4 home_team
    5 away_team
    6 home_pred
    7 away_pred
    8 predicted_winner
    9 margin
    10 confidence
    11 home_pred_team
    12 away_pred_team
    13 home_pred_player
    14 away_pred_player
    15 is_playoff
    16 season_type
    """

    # Skip empty rows
    if not row or all(clean(x) is None for x in row):
        return None

    # Skip header rows
    first = clean(row[0])
    if first and first.lower() in {"date", "game_date"}:
        return None

    # New format from your updated save_predictions()
    if len(row) >= 17:
        return {
            "game_date": clean(row[0]),
            "saved_at": clean(row[1]),
            "model_version": clean(row[2]),
            "game_id": to_int(row[3]),
            "home_team": clean(row[4]),
            "away_team": clean(row[5]),
            "home_pred": to_float(row[6]),
            "away_pred": to_float(row[7]),
            "predicted_winner": clean(row[8]),
            "margin": to_float(row[9]),
            "confidence": clean(row[10]),
            "home_pred_team": to_float(row[11]),
            "away_pred_team": to_float(row[12]),
            "home_pred_player": to_float(row[13]),
            "away_pred_player": to_float(row[14]),
            "is_playoff": to_bool(row[15]),
            "season_type": clean(row[16]),
        }

    # Old format from earlier logs
    if len(row) >= 14:
        return {
            "game_date": clean(row[0]),
            "saved_at": None,
            "model_version": "legacy_csv",
            "game_id": None,
            "home_team": clean(row[1]),
            "away_team": clean(row[2]),
            "home_pred": to_float(row[3]),
            "away_pred": to_float(row[4]),
            "predicted_winner": clean(row[5]),
            "margin": to_float(row[6]),
            "confidence": clean(row[7]),
            "home_pred_team": to_float(row[8]),
            "away_pred_team": to_float(row[9]),
            "home_pred_player": to_float(row[10]),
            "away_pred_player": to_float(row[11]),
            "is_playoff": to_bool(row[12]),
            "season_type": clean(row[13]),
        }

    print(f"Skipping malformed row with {len(row)} columns: {row}")
    return None


def import_csv():
    inserted = 0
    skipped = 0

    with engine.begin() as conn:
        conn.execute(text(CREATE_STAGING_SQL))

        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                try:
                    data = parse_row(row)

                    if data is None:
                        skipped += 1
                        continue

                    conn.execute(text(INSERT_ROW_SQL), data)
                    inserted += 1

                except Exception as e:
                    skipped += 1
                    print(f"Skipping row due to error: {e}")
                    print(row)

    print(f"✅ Imported {inserted} rows into prediction_log_staging")
    print(f"⏭️ Skipped {skipped} rows")


if __name__ == "__main__":
    import_csv()