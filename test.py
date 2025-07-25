# main.py - Pipeline Runner
import argparse
import sys
from pathlib import Path


async def run_database_pipeline(export_path: Optional[str] = None):
    """Main pipeline function to demonstrate the database setup and CSV export"""

    db_manager = DatabaseManager()

    try:
        # Setup databases
        print("🚀 Setting up databases...")
        await db_manager.setup_databases()
        print("✅ Database setup completed!")

        # Health check
        print("\n🔍 Performing health check...")
        health_status = await db_manager.health_check()
        for db_name, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {db_name}: {'Healthy' if status else 'Unhealthy'}")

        # Get database statistics
        print("\n📊 Getting database statistics...")
        stats = await db_manager.get_database_statistics()
        for entity, count in stats.items():
            print(f"   {entity.capitalize()}: {count} records")

        # Demo operations (create sample data if tables are empty)
        total_records = sum(stats.values())
        if total_records == 0:
            print("\n📝 Creating sample data...")
            await create_sample_data(db_manager)

            # Update statistics
            stats = await db_manager.get_database_statistics()
            print("\n📊 Updated database statistics:")
            for entity, count in stats.items():
                print(f"   {entity.capitalize()}: {count} records")

        # CSV Export operations
        if export_path:
            print(f"\n💾 Starting CSV export to: {export_path}")
            csv_manager = db_manager.get_csv_manager()

            # Export all tables
            exported_files = await csv_manager.export_all_tables(export_path)

            print(f"✅ Successfully exported {len(exported_files)} files:")
            for file_path in exported_files:
                file_size = Path(file_path).stat().st_size / 1024  # KB
                print(f"   📄 {Path(file_path).name} ({file_size:.1f} KB)")

            # Example: Export users with specific filters
            print("\n🔍 Exporting filtered data (active users only)...")
            user_filter_file = await csv_manager.export_with_custom_query(
                entity="users",
                export_path=export_path,
                query_filters={"is_active": True},
                filename="active_users.csv",
            )

            file_size = Path(user_filter_file).stat().st_size / 1024
            print(
                f"✅ Exported filtered users: {Path(user_filter_file).name} ({file_size:.1f} KB)"
            )

        print("\n🎉 Pipeline completed successfully!")

    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise
    finally:
        # Cleanup
        print("\n🧹 Shutting down...")
        await db_manager.shutdown()
        print("✅ Shutdown completed!")


async def create_sample_data(db_manager: DatabaseManager):
    """Create sample data for demonstration"""

    # Sample users
    user_ops = db_manager.get_operations("users")
    sample_users = [
        {
            "username": "john_doe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "hashed_password": "hashed_password_123",
            "phone": "+1234567890",
            "address": "123 Main St, City, State",
        },
        {
            "username": "jane_smith",
            "email": "jane@example.com",
            "full_name": "Jane Smith",
            "hashed_password": "hashed_password_456",
            "phone": "+1234567891",
            "address": "456 Oak Ave, City, State",
        },
        {
            "username": "bob_wilson",
            "email": "bob@example.com",
            "full_name": "Bob Wilson",
            "hashed_password": "hashed_password_789",
            "phone": "+1234567892",
            "address": "789 Pine Rd, City, State",
        },
    ]

    user_ids = await user_ops.insert_bulk(sample_users)
    print(f"   Created {len(user_ids)} users")

    # Sample products
    product_ops = db_manager.get_operations("products")
    sample_products = [
        {
            "name": "Laptop",
            "description": "High-performance laptop for work",
            "price": 999.99,
            "category": "Electronics",
            "stock_quantity": 50,
            "sku": "LAP001",
            "brand": "TechBrand",
        },
        {
            "name": "Smartphone",
            "description": "Latest smartphone with great camera",
            "price": 699.99,
            "category": "Electronics",
            "stock_quantity": 100,
            "sku": "PHN001",
            "brand": "PhoneBrand",
        },
        {
            "name": "Office Chair",
            "description": "Ergonomic office chair",
            "price": 299.99,
            "category": "Furniture",
            "stock_quantity": 30,
            "sku": "CHR001",
            "brand": "FurnitureBrand",
        },
        {
            "name": "Desk Lamp",
            "description": "LED desk lamp with adjustable brightness",
            "price": 49.99,
            "category": "Office",
            "stock_quantity": 75,
            "sku": "LMP001",
            "brand": "LightBrand",
        },
        {
            "name": "Wireless Mouse",
            "description": "Ergonomic wireless mouse",
            "price": 29.99,
            "category": "Electronics",
            "stock_quantity": 200,
            "sku": "MOU001",
            "brand": "TechBrand",
        },
    ]

    product_ids = await product_ops.insert_bulk(sample_products)
    print(f"   Created {len(product_ids)} products")

    # Sample orders
    order_ops = db_manager.get_operations("orders")
    sample_orders = [
        {
            "user_id": user_ids[0],
            "product_id": product_ids[0],
            "quantity": 1,
            "total_amount": 999.99,
            "status": "completed",
            "shipping_address": "123 Main St, City, State",
        },
        {
            "user_id": user_ids[1],
            "product_id": product_ids[1],
            "quantity": 2,
            "total_amount": 1399.98,
            "status": "pending",
            "shipping_address": "456 Oak Ave, City, State",
        },
        {
            "user_id": user_ids[0],
            "product_id": product_ids[4],
            "quantity": 3,
            "total_amount": 89.97,
            "status": "shipped",
            "shipping_address": "123 Main St, City, State",
        },
        {
            "user_id": user_ids[2],
            "product_id": product_ids[2],
            "quantity": 1,
            "total_amount": 299.99,
            "status": "completed",
            "shipping_address": "789 Pine Rd, City, State",
        },
        {
            "user_id": user_ids[1],
            "product_id": product_ids[3],
            "quantity": 2,
            "total_amount": 99.98,
            "status": "pending",
            "shipping_address": "456 Oak Ave, City, State",
        },
    ]

    order_ids = await order_ops.insert_bulk(sample_orders)
    print(f"   Created {len(order_ids)} orders")


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Multi-Database PostgreSQL Setup and CSV Export Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline without CSV export
  python main.py

  # Run pipeline with CSV export to specific path
  python main.py --export-path /path/to/export/folder

  # Run pipeline with CSV export to current directory
  python main.py --export-path ./exports
        """,
    )

    parser.add_argument(
        "--export-path",
        type=str,
        help="Path where CSV files will be exported. If not provided, no CSV export will be performed.",
    )

    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Force creation of sample data even if tables are not empty",
    )

    args = parser.parse_args()

    # Validate export path if provided
    if args.export_path:
        export_path = Path(args.export_path)
        try:
            export_path.mkdir(parents=True, exist_ok=True)
            if not export_path.is_dir():
                print(f"❌ Error: Export path '{args.export_path}' is not a directory")
                sys.exit(1)
        except PermissionError:
            print(
                f"❌ Error: Permission denied to create directory '{args.export_path}'"
            )
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: Unable to create export directory: {str(e)}")
            sys.exit(1)

    # Run the pipeline
    try:
        asyncio.run(run_database_pipeline(args.export_path))
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# requirements.txt
"""
Required dependencies for the multi-database setup:

asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23
python-dotenv==1.0.0
pandas==2.1.4
pathlib==1.0.1
"""

# .env.example
"""
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=your_password_here

# Database Names
USERS_DB_NAME=users_database
PRODUCTS_DB_NAME=products_database
ORDERS_DB_NAME=orders_database

# Connection Pool Settings (Optional)
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# CSV Export Settings (Optional)
CSV_EXPORT_PATH=./exports
CSV_CHUNK_SIZE=1000
"""
# constants.py
from typing import Dict, Any
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration dataclass"""

    host: str
    port: int
    username: str
    password: str
    database: str

    def get_connection_string(self) -> str:
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class DatabaseConstants:
    """Constants for database configuration"""

    # Common database connection parameters from .env
    DEFAULT_HOST = os.getenv("DB_HOST")
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


# db_models.py
from sqlalchemy import Column, Integer, String, DateTime, Numeric, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

# Base class for all models
Base = declarative_base()


class BaseModel:
    """Base model with common fields"""

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )
    is_active = Column(Boolean, default=True, nullable=False)


class User(Base, BaseModel):
    """User model for users database"""

    __tablename__ = "users"

    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=True)
    address = Column(Text, nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class Product(Base, BaseModel):
    """Product model for products database"""

    __tablename__ = "products"

    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    price = Column(Numeric(10, 2), nullable=False)
    category = Column(String(50), nullable=False, index=True)
    stock_quantity = Column(Integer, default=0, nullable=False)
    sku = Column(String(50), unique=True, nullable=False, index=True)
    brand = Column(String(50), nullable=True)

    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', price={self.price})>"


class Order(Base, BaseModel):
    """Order model for orders database"""

    __tablename__ = "orders"

    user_id = Column(Integer, nullable=False, index=True)
    product_id = Column(Integer, nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    total_amount = Column(Numeric(10, 2), nullable=False)
    status = Column(String(20), default="pending", nullable=False, index=True)
    order_date = Column(DateTime, default=func.now(), nullable=False)
    shipping_address = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Order(id={self.id}, user_id={self.user_id}, total_amount={self.total_amount})>"


# database_engine.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from typing import Dict, Optional
import asyncio
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


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
                poolclass=QueuePool,
                pool_size=DatabaseConstants.POOL_SIZE,
                max_overflow=DatabaseConstants.MAX_OVERFLOW,
                pool_timeout=DatabaseConstants.POOL_TIMEOUT,
                pool_recycle=DatabaseConstants.POOL_RECYCLE,
                echo=False,  # Set to True for SQL logging in development
                future=True,
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

    async def create_all_tables(self):
        """Create tables in all databases"""
        # Map models to their respective databases
        model_mappings = {"users_db": User, "products_db": Product, "orders_db": Order}

        tasks = []
        for db_name, engine in self.engines.items():
            if db_name in model_mappings:
                # Create a base for each model
                model_class = model_mappings[db_name]
                base = type(model_class).__bases__[0]  # Get the Base class
                tasks.append(engine.create_tables(base))

        await asyncio.gather(*tasks)
        logger.info("All tables created")

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


# database_operations.py
from sqlalchemy import select, insert, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional, Type, TypeVar
from abc import ABC, abstractmethod
import logging
import pandas as pd
import os
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Base)


class DatabaseOperations(ABC):
    """Abstract base class for database operations"""

    def __init__(self, engine: DatabaseEngine, model_class: Type[T]):
        self.engine = engine
        self.model_class = model_class

    async def fetch_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[T]:
        """Fetch all records with optional filters, limit, and offset"""
        async with self.engine.get_session() as session:
            query = select(self.model_class)

            # Apply filters
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model_class, key):
                        query = query.where(getattr(self.model_class, key) == value)

            # Apply pagination
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            records = result.scalars().all()

            logger.info(
                f"Fetched {len(records)} records from {self.model_class.__tablename__}"
            )
            return records

    async def fetch_all_as_dict(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all records as dictionaries for CSV export"""
        records = await self.fetch_all(filters, limit, offset)

        # Convert SQLAlchemy objects to dictionaries
        result_dicts = []
        for record in records:
            record_dict = {}
            for column in self.model_class.__table__.columns:
                value = getattr(record, column.name)
                # Handle datetime objects
                if isinstance(value, datetime):
                    record_dict[column.name] = value.isoformat()
                else:
                    record_dict[column.name] = value
            result_dicts.append(record_dict)

        return result_dicts

    async def export_to_csv(
        self,
        file_path: str,
        filters: Optional[Dict[str, Any]] = None,
        chunk_size: int = None,
    ) -> str:
        """Export data to CSV file with chunking for large datasets"""
        if chunk_size is None:
            chunk_size = DatabaseConstants.CSV_CHUNK_SIZE

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Get total count for progress tracking
        total_count = await self.count_records(filters)
        logger.info(
            f"Exporting {total_count} records from {self.model_class.__tablename__} to {file_path}"
        )

        exported_count = 0
        first_chunk = True

        # Process data in chunks
        for offset in range(0, total_count, chunk_size):
            chunk_data = await self.fetch_all_as_dict(
                filters=filters, limit=chunk_size, offset=offset
            )

            if chunk_data:
                df = pd.DataFrame(chunk_data)

                # Write to CSV (append mode after first chunk)
                df.to_csv(
                    file_path,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False,
                )

                exported_count += len(chunk_data)
                first_chunk = False

                logger.info(f"Exported {exported_count}/{total_count} records")

        logger.info(f"Successfully exported {exported_count} records to {file_path}")
        return file_path

    async def fetch_by_id(self, record_id: int) -> Optional[T]:
        """Fetch a single record by ID"""
        async with self.engine.get_session() as session:
            result = await session.get(self.model_class, record_id)
            logger.info(
                f"Fetched record with ID {record_id} from {self.model_class.__tablename__}"
            )
            return result

    async def insert_record(self, data: Dict[str, Any]) -> T:
        """Insert a single record"""
        async with self.engine.get_session() as session:
            record = self.model_class(**data)
            session.add(record)
            await session.flush()  # To get the ID
            await session.refresh(record)

            logger.info(
                f"Inserted record with ID {record.id} into {self.model_class.__tablename__}"
            )
            return record

    async def insert_bulk(self, data_list: List[Dict[str, Any]]) -> List[int]:
        """Insert multiple records in bulk"""
        async with self.engine.get_session() as session:
            stmt = insert(self.model_class).returning(self.model_class.id)
            result = await session.execute(stmt, data_list)
            record_ids = result.scalars().all()

            logger.info(
                f"Bulk inserted {len(record_ids)} records into {self.model_class.__tablename__}"
            )
            return record_ids

    async def update_record(self, record_id: int, data: Dict[str, Any]) -> bool:
        """Update a record by ID"""
        async with self.engine.get_session() as session:
            stmt = (
                update(self.model_class)
                .where(self.model_class.id == record_id)
                .values(**data)
            )

            result = await session.execute(stmt)
            success = result.rowcount > 0

            logger.info(
                f"Updated record with ID {record_id} in {self.model_class.__tablename__}: {success}"
            )
            return success

    async def delete_record(self, record_id: int) -> bool:
        """Delete a record by ID"""
        async with self.engine.get_session() as session:
            stmt = delete(self.model_class).where(self.model_class.id == record_id)
            result = await session.execute(stmt)
            success = result.rowcount > 0

            logger.info(
                f"Deleted record with ID {record_id} from {self.model_class.__tablename__}: {success}"
            )
            return success

    async def count_records(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filters"""
        async with self.engine.get_session() as session:
            query = select(func.count(self.model_class.id))

            if filters:
                for key, value in filters.items():
                    if hasattr(self.model_class, key):
                        query = query.where(getattr(self.model_class, key) == value)

            result = await session.execute(query)
            count = result.scalar()

            logger.info(f"Counted {count} records in {self.model_class.__tablename__}")
            return count


class UserOperations(DatabaseOperations):
    """Specific operations for User model"""

    def __init__(self, engine: DatabaseEngine):
        super().__init__(engine, User)

    async def fetch_by_username(self, username: str) -> Optional[User]:
        """Fetch user by username"""
        async with self.engine.get_session() as session:
            query = select(User).where(User.username == username)
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def fetch_by_email(self, email: str) -> Optional[User]:
        """Fetch user by email"""
        async with self.engine.get_session() as session:
            query = select(User).where(User.email == email)
            result = await session.execute(query)
            return result.scalar_one_or_none()


class ProductOperations(DatabaseOperations):
    """Specific operations for Product model"""

    def __init__(self, engine: DatabaseEngine):
        super().__init__(engine, Product)

    async def fetch_by_category(self, category: str) -> List[Product]:
        """Fetch products by category"""
        return await self.fetch_all(filters={"category": category})

    async def fetch_by_sku(self, sku: str) -> Optional[Product]:
        """Fetch product by SKU"""
        async with self.engine.get_session() as session:
            query = select(Product).where(Product.sku == sku)
            result = await session.execute(query)
            return result.scalar_one_or_none()


class OrderOperations(DatabaseOperations):
    """Specific operations for Order model"""

    def __init__(self, engine: DatabaseEngine):
        super().__init__(engine, Order)

    async def fetch_by_user(self, user_id: int) -> List[Order]:
        """Fetch orders by user ID"""
        return await self.fetch_all(filters={"user_id": user_id})

    async def fetch_by_status(self, status: str) -> List[Order]:
        """Fetch orders by status"""
        return await self.fetch_all(filters={"status": status})


# database_manager.py
import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class CSVExportManager:
    """Manager for CSV export operations"""

    def __init__(self, db_manager: "DatabaseManager"):
        self.db_manager = db_manager

    async def export_single_table(
        self,
        entity: str,
        export_path: str,
        filters: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Export a single table to CSV"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{entity}_{timestamp}.csv"

        file_path = Path(export_path) / filename

        operations = self.db_manager.get_operations(entity)
        exported_file = await operations.export_to_csv(str(file_path), filters)

        return exported_file

    async def export_all_tables(
        self, export_path: str, filters: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[str]:
        """Export all tables to CSV files"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_folder = Path(export_path) / f"database_export_{timestamp}"
        export_folder.mkdir(parents=True, exist_ok=True)

        exported_files = []
        entities = ["users", "products", "orders"]

        for entity in entities:
            try:
                entity_filters = filters.get(entity) if filters else None
                filename = f"{entity}.csv"

                exported_file = await self.export_single_table(
                    entity=entity,
                    export_path=str(export_folder),
                    filters=entity_filters,
                    filename=filename,
                )

                exported_files.append(exported_file)

            except Exception as e:
                logging.error(f"Failed to export {entity}: {str(e)}")
                continue

        logging.info(f"Exported {len(exported_files)} tables to {export_folder}")
        return exported_files

    async def export_with_custom_query(
        self,
        entity: str,
        export_path: str,
        query_filters: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """Export data with custom filtering"""

        return await self.export_single_table(
            entity=entity,
            export_path=export_path,
            filters=query_filters,
            filename=filename,
        )


class DatabaseManager:
    """Main database manager that orchestrates all database operations"""

    def __init__(self):
        self.multi_db_manager = MultiDatabaseManager()
        self.operations: Dict[str, DatabaseOperations] = {}
        self.csv_manager: Optional[CSVExportManager] = None
        self._initialized = False

    async def setup_databases(self):
        """Setup all database configurations and connections"""
        try:
            # Create database configurations
            db_configs = {
                "users_db": DatabaseConfig(
                    host=DatabaseConstants.DEFAULT_HOST,
                    port=DatabaseConstants.DEFAULT_PORT,
                    username=DatabaseConstants.DEFAULT_USERNAME,
                    password=DatabaseConstants.DEFAULT_PASSWORD,
                    database=DatabaseConstants.DATABASE_NAMES["users_db"],
                ),
                "products_db": DatabaseConfig(
                    host=DatabaseConstants.DEFAULT_HOST,
                    port=DatabaseConstants.DEFAULT_PORT,
                    username=DatabaseConstants.DEFAULT_USERNAME,
                    password=DatabaseConstants.DEFAULT_PASSWORD,
                    database=DatabaseConstants.DATABASE_NAMES["products_db"],
                ),
                "orders_db": DatabaseConfig(
                    host=DatabaseConstants.DEFAULT_HOST,
                    port=DatabaseConstants.DEFAULT_PORT,
                    username=DatabaseConstants.DEFAULT_USERNAME,
                    password=DatabaseConstants.DEFAULT_PASSWORD,
                    database=DatabaseConstants.DATABASE_NAMES["orders_db"],
                ),
            }

            # Add databases to manager
            for name, config in db_configs.items():
                self.multi_db_manager.add_database(name, config)

            # Initialize all engines
            await self.multi_db_manager.initialize_all()

            # Create tables
            await self.multi_db_manager.create_all_tables()

            # Setup operation classes
            self.operations = {
                "users": UserOperations(self.multi_db_manager.get_engine("users_db")),
                "products": ProductOperations(
                    self.multi_db_manager.get_engine("products_db")
                ),
                "orders": OrderOperations(
                    self.multi_db_manager.get_engine("orders_db")
                ),
            }

            # Initialize CSV manager
            self.csv_manager = CSVExportManager(self)

            self._initialized = True
            logging.info("Database manager setup completed successfully")

        except Exception as e:
            logging.error(f"Failed to setup databases: {str(e)}")
            raise

    def get_operations(self, entity: str) -> DatabaseOperations:
        """Get operations instance for a specific entity"""
        if not self._initialized:
            raise RuntimeError(
                "Database manager not initialized. Call setup_databases() first."
            )

        if entity not in self.operations:
            raise ValueError(
                f"Entity '{entity}' not supported. Available: {list(self.operations.keys())}"
            )

        return self.operations[entity]

    def get_csv_manager(self) -> CSVExportManager:
        """Get CSV export manager"""
        if not self._initialized:
            raise RuntimeError(
                "Database manager not initialized. Call setup_databases() first."
            )

        return self.csv_manager

    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all databases"""
        health_status = {}

        for db_name, engine in self.multi_db_manager.engines.items():
            try:
                async with engine.get_session() as session:
                    await session.execute(select(1))
                health_status[db_name] = True
            except Exception as e:
                logging.error(f"Health check failed for {db_name}: {str(e)}")
                health_status[db_name] = False

        return health_status

    async def get_database_statistics(self) -> Dict[str, int]:
        """Get record counts for all tables"""
        stats = {}

        for entity_name, operations in self.operations.items():
            try:
                count = await operations.count_records()
                stats[entity_name] = count
            except Exception as e:
                logging.error(f"Failed to get count for {entity_name}: {str(e)}")
                stats[entity_name] = 0

        return stats

    async def shutdown(self):
        """Gracefully shutdown all database connections"""
        await self.multi_db_manager.close_all()
        logging.info("Database manager shutdown completed")


# main.py - Pipeline Runner
async def run_database_pipeline():
    """Main pipeline function to demonstrate the database setup"""

    db_manager = DatabaseManager()

    try:
        # Setup databases
        print("🚀 Setting up databases...")
        await db_manager.setup_databases()
        print("✅ Database setup completed!")

        # Health check
        print("\n🔍 Performing health check...")
        health_status = await db_manager.health_check()
        for db_name, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {db_name}: {'Healthy' if status else 'Unhealthy'}")

        # Demo operations
        print("\n📝 Running demo operations...")

        # User operations
        user_ops = db_manager.get_operations("users")

        # Insert sample user
        user_data = {
            "username": "john_doe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "hashed_password": "hashed_password_123",
            "phone": "+1234567890",
        }

        user = await user_ops.insert_record(user_data)
        print(f"✅ Inserted user: {user}")

        # Fetch all users
        users = await user_ops.fetch_all()
        print(f"📋 Total users: {len(users)}")

        # Product operations
        product_ops = db_manager.get_operations("products")

        # Insert sample product
        product_data = {
            "name": "Laptop",
            "description": "High-performance laptop",
            "price": 999.99,
            "category": "Electronics",
            "stock_quantity": 50,
            "sku": "LAP001",
            "brand": "TechBrand",
        }

        product = await product_ops.insert_record(product_data)
        print(f"✅ Inserted product: {product}")

        # Order operations
        order_ops = db_manager.get_operations("orders")

        # Insert sample order
        order_data = {
            "user_id": user.id,
            "product_id": product.id,
            "quantity": 2,
            "total_amount": 1999.98,
            "status": "pending",
            "shipping_address": "123 Main St, City, State",
        }

        order = await order_ops.insert_record(order_data)
        print(f"✅ Inserted order: {order}")

        # Count operations
        user_count = await user_ops.count_records()
        product_count = await product_ops.count_records()
        order_count = await order_ops.count_records()

        print(f"\n📊 Database Statistics:")
        print(f"   Users: {user_count}")
        print(f"   Products: {product_count}")
        print(f"   Orders: {order_count}")

        print("\n🎉 Pipeline completed successfully!")

    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise
    finally:
        # Cleanup
        print("\n🧹 Shutting down...")
        await db_manager.shutdown()
        print("✅ Shutdown completed!")


if __name__ == "__main__":
    # Run the pipeline
    asyncio.run(run_database_pipeline())
