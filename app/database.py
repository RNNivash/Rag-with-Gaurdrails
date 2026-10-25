import logging
from pymongo import MongoClient
from app.config import settings

logger = logging.getLogger(__name__)

class Database:
    client: MongoClient | None = None

    @classmethod
    def get_client(cls) -> MongoClient:
        if cls.client is None:
            if not settings.MONGODB_URI:
                raise ValueError("MONGODB_URI is not configured in the environment.")
            cls.client = MongoClient(settings.MONGODB_URI)
            logger.info("Connected to MongoDB successfully.")
        return cls.client

    @classmethod
    def get_collection(cls):
        client = cls.get_client()
        db = client[settings.MONGODB_DB_NAME]
        return db[settings.MONGODB_COLLECTION_NAME]

    @classmethod
    def ping(cls) -> bool:
        try:
            client = cls.get_client()
            client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB ping failed: {e}")
            return False

    @classmethod
    def close(cls):
        if cls.client is not None:
            cls.client.close()
            cls.client = None
            logger.info("MongoDB connection closed.")
