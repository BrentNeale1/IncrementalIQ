import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Switch to PostgreSQL by changing this single env var:
#   DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///incrementiq.db")

engine = create_engine(
    DATABASE_URL,
    # SQLite needs check_same_thread=False for FastAPI's threaded usage
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
