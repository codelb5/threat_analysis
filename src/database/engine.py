# database_engine.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from typing import Dict, Optional
import asyncio
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

from config.db_config import DatabaseConfig
from constants.db_constants import DatabaseConstants


class DatabaseEngine:
    """Generic database engine client for creating and managing database connections"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self._initialized = False

    async def initialize(self):
        """Initialize the database engine and session factory"""
        try:
            self.engine = create_async_engine(
                self.config.get_connection_string(),
                # poolclass=QueuePool,
                # pool_size=DatabaseConstants.POOL_SIZE,
                # max_overflow=DatabaseConstants.MAX_OVERFLOW,
                # pool_timeout=DatabaseConstants.POOL_TIMEOUT,
                # pool_recycle=DatabaseConstants.POOL_RECYCLE,
                # echo=False,  # Set to True for SQL logging in development
                # future=True
            )

            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )

            self._initialized = True
            logger.info(
                f"Database engine initialized for database: {self.config.database}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database engine: {str(e)}")
            raise

    async def create_tables(self, base_class):
        """Create tables for the given base class"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(base_class.metadata.create_all)
            logger.info(f"Tables created for database: {self.config.database}")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise

    @asynccontextmanager
    async def get_session(self):
        """Get an async database session"""
        if not self._initialized:
            await self.initialize()

        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {str(e)}")
                raise
            finally:
                await session.close()

    async def close(self):
        """Close the database engine"""
        if self.engine:
            await self.engine.dispose()
            logger.info(f"Database engine closed for database: {self.config.database}")


class MultiDatabaseManager:
    """Manager for handling multiple database connections"""

    def __init__(self):
        self.engines: Dict[str, DatabaseEngine] = {}
        self.configs: Dict[str, DatabaseConfig] = {}

    def add_database(self, name: str, config: DatabaseConfig):
        """Add a database configuration"""
        self.configs[name] = config
        self.engines[name] = DatabaseEngine(config)

    async def initialize_all(self):
        """Initialize all database engines"""
        tasks = []
        for name, engine in self.engines.items():
            tasks.append(engine.initialize())

        await asyncio.gather(*tasks)
        logger.info("All database engines initialized")

    def get_engine(self, database_name: str) -> DatabaseEngine:
        """Get database engine by name"""
        if database_name not in self.engines:
            raise ValueError(f"Database {database_name} not configured")
        return self.engines[database_name]

    async def close_all(self):
        """Close all database engines"""
        tasks = []
        for engine in self.engines.values():
            tasks.append(engine.close())

        await asyncio.gather(*tasks)
        logger.info("All database engines closed")

    # async def create_all_tables(self):
    #     """Create tables in all databases"""
    #     # Map models to their respective databases
    #     model_mappings = {
    #         "users_db": User,
    #         "products_db": Product,
    #         "orders_db": Order
    #     }

    #     tasks = []
    #     for db_name, engine in self.engines.items():
    #         if db_name in model_mappings:
    #             # Create a base for each model
    #             model_class = model_mappings[db_name]
    #             base = type(model_class).__bases__[0]  # Get the Base class
    #             tasks.append(engine.create_tables(base))

    #     await asyncio.gather(*tasks)
    #     logger.info("All tables created")
