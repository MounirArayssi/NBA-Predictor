import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from data.storage.db import engine
from data.storage.models import TeamStyleVector, TeamSimilarity


def compute_style_vectors(as_of_date=None):
    """Compute style vectors for all teams based on season stats."""
    print("Computing team style vectors...")

    query = text("""
        SELECT
            tbs.team_id,
            t.abbreviation,
            AVG(tbs.pace)        AS avg_pace,
            AVG(tbs.fg3a::float / NULLIF(tbs.fga, 0)) AS three_point_rate,
            AVG(tbs.offensive_rating)  AS offensive_rating,
            AVG(tbs.defensive_rating)  AS defensive_rating,
            AVG(tbs.efg_pct)           AS efg_pct,
            AVG(tbs.tov_pct)           AS tov_pct,
            AVG(tbs.oreb_pct)          AS oreb_pct,
            AVG(tbs.ft_rate)           AS ft_rate,
            AVG(tbs.fg3_pct)           AS fg3_pct,
            COUNT(*)                   AS games_played
        FROM team_box_scores tbs
        JOIN teams t ON tbs.team_id = t.team_id
        JOIN games g ON tbs.game_id = g.game_id
        WHERE g.season = '2025-26'
        AND g.is_final = TRUE
        GROUP BY tbs.team_id, t.abbreviation
        HAVING COUNT(*) >= 10
    """)

    df = pd.read_sql(query, engine)
    print(f"  Got style data for {len(df)} teams")

    return df


def compute_and_store_similarity(season='2025-26'):
    """Compute cosine similarity between all team pairs."""
    print(f"\nComputing team similarity for {season}...")

    style_df = compute_style_vectors()

    if style_df.empty:
        print("❌ No style data found")
        return

    # Style features to use for similarity
    style_features = [
        'avg_pace',
        'three_point_rate',
        'offensive_rating',
        'defensive_rating',
        'efg_pct',
        'tov_pct',
        'oreb_pct',
        'ft_rate',
        'fg3_pct',
    ]

    # Drop rows with missing values
    style_df = style_df.dropna(subset=style_features)
    print(f"  Computing similarity for {len(style_df)} teams")

    # Normalize features
    scaler = StandardScaler()
    style_matrix = scaler.fit_transform(style_df[style_features])

    # Compute cosine similarity
    sim_matrix = cosine_similarity(style_matrix)

    today = pd.Timestamp.today().date()

    with Session(engine) as session:
        stored = 0
        for i in range(len(style_df)):
            for j in range(len(style_df)):
                if i == j:
                    continue

                team_a_id = int(style_df.iloc[i]['team_id'])
                team_b_id = int(style_df.iloc[j]['team_id'])
                similarity = float(sim_matrix[i][j])

                # Use consistent ordering (smaller id first)
                if team_a_id > team_b_id:
                    continue

                from data.storage.models import TeamSimilarity as TS
                existing = session.query(TS).filter_by(
                    team_a_id=team_a_id,
                    team_b_id=team_b_id,
                    as_of_date=today
                ).first()

                if existing:
                    existing.similarity_score = similarity
                else:
                    record = TS(
                        team_a_id        = team_a_id,
                        team_b_id        = team_b_id,
                        as_of_date       = today,
                        similarity_score = similarity
                    )
                    session.add(record)
                    stored += 1

        session.commit()
        print(f"  ✅ Stored {stored} similarity pairs")

    # Print most similar teams for sanity check
    print("\nMost similar team pairs:")
    results = []
    for i in range(len(style_df)):
        for j in range(i+1, len(style_df)):
            results.append({
                'team_a': style_df.iloc[i]['abbreviation'],
                'team_b': style_df.iloc[j]['abbreviation'],
                'similarity': sim_matrix[i][j]
            })

    results_df = pd.DataFrame(results).sort_values(
        'similarity', ascending=False
    )
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    compute_and_store_similarity()