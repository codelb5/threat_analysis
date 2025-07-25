import os
from dotenv import load_dotenv

load_dotenv()


class DatabaseConstants:
    """Constants for database configuration"""

    # Common database connection parameters from .env
    DEFAULT_HOST = os.getenv("DB_HOST", "localhost")
    DEFAULT_PORT = int(os.getenv("DB_PORT", "5432"))
    DEFAULT_USERNAME = os.getenv("DB_USERNAME")
    DEFAULT_PASSWORD = os.getenv("DB_PASSWORD")

    # Validate required environment variables
    if not DEFAULT_HOST:
        raise ValueError("DB_HOST is required in .env file")
    if not DEFAULT_USERNAME:
        raise ValueError("DB_USERNAME is required in .env file")
    if not DEFAULT_PASSWORD:
        raise ValueError("DB_PASSWORD is required in .env file")

    # Database names
    DATABASE_NAMES = {
        "users_db": os.getenv("USERS_DB_NAME", "users_database"),
        "products_db": os.getenv("PRODUCTS_DB_NAME", "products_database"),
        "orders_db": os.getenv("ORDERS_DB_NAME", "orders_database"),
    }

    # Connection pool settings
    POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

    # CSV export settings
    DEFAULT_CSV_PATH = os.getenv("CSV_EXPORT_PATH", "./exports")
    CSV_CHUNK_SIZE = int(os.getenv("CSV_CHUNK_SIZE", "1000"))
