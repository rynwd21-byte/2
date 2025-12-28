import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Use ephemeral writeable storage on serverless
    os.makedirs("/tmp", exist_ok=True)
    DATABASE_URL = "sqlite:////tmp/cfb.sqlite3"
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
