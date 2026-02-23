import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Switch to PostgreSQL by changing this single env var:
#   DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///incrementiq.db")

_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False, "timeout": 30}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)

# Enable WAL mode for SQLite so readers don't block writers
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

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
