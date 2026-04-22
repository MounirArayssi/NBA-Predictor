import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import create_engine, text
from config.settings import DATABASE_URL
from data.storage.models import Base

engine = create_engine(DATABASE_URL)

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