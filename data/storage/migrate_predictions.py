from sqlalchemy import text
from db import engine


def migrate_predictions_table():
    queries = [
        # Core tracking fields
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS is_official BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS prediction_type VARCHAR(30);",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS lock_reason VARCHAR(100);",

        # Model outputs
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_margin NUMERIC(6,2);",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_total NUMERIC(6,2);",

        # Market comparison
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS vegas_spread NUMERIC(6,2);",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS vegas_total NUMERIC(6,2);",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS spread_edge NUMERIC(6,2);",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS total_edge NUMERIC(6,2);",
    ]

    with engine.connect() as conn:
        for q in queries:
            conn.execute(text(q))
            print(f"✅ Executed: {q}")
        conn.commit()


if __name__ == "__main__":
    print("DB URL:", engine.url)
    print("🔧 Running predictions table migration...\n")
    migrate_predictions_table()
    print("\n✅ Migration complete.")