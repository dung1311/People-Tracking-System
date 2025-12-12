import os
import logging

from dotenv import load_dotenv
from sqlmodel import create_engine, Session, SQLModel

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    url=DATABASE_URL,
)

def init_db():
    try: 
        SQLModel.metadata.create_all(engine)
        logger.info("Init database success")
    except Exception as e:
        logger.error("Can not init database")
        raise e
    
def get_session():
    with Session(engine) as session:
        yield session