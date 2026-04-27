import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

from sqlalchemy import create_engine, text
from data.storage.models import Base


def get_database_url():
    """
    Works both locally and on Streamlit Cloud.

    Priority:
    1. Streamlit Cloud secrets
    2. Environment variable DATABASE_URL
    3. config.settings DATABASE_URL fallback
    """
    try:
        import streamlit as st
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
    except Exception:
        pass

    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    try:
        from config.settings import DATABASE_URL
        return DATABASE_URL
    except Exception:
        return None


DATABASE_URL = get_database_url()

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing. Add it to Streamlit Cloud secrets or local config."
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)


def init_db():
    Base.metadata.create_all(engine)
    print("✅ All tables created successfully")


def test_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    test_connection()
    init_db()