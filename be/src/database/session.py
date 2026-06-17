import os
import logging

from dotenv import load_dotenv
from sqlmodel import create_engine, Session, SQLModel

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    db_user = os.getenv("DB_USER", "admin")
    db_password = os.getenv("DB_PASSWORD", "admin123")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "SCT")
    DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(
    url=DATABASE_URL,
)

from sqlalchemy import text

def init_db():
    try: 
        # Ensure pgvector extension is enabled in Postgres
        with Session(engine) as session:
            session.exec(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()
            
        SQLModel.metadata.create_all(engine)
        
        # Automatic column migration for Camera.rois
        with Session(engine) as session:
            session.exec(text("ALTER TABLE camera ADD COLUMN IF NOT EXISTS rois JSON;"))
            session.exec(text("ALTER TABLE camera ADD COLUMN IF NOT EXISTS is_primary BOOLEAN DEFAULT true;"))
            session.exec(text("ALTER TABLE video_segment ADD COLUMN IF NOT EXISTS batch_number INTEGER DEFAULT 1;"))
            session.commit()
            
        logger.info("Init database success")
        from database.init_data import seed_db
        seed_db()
    except Exception as e:
        logger.error("Can not init database")
        raise e
    
def get_session():
    with Session(engine) as session:
        yield session