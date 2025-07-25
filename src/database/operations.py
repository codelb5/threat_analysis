# database_operations.py
from sqlalchemy import select, insert, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional, Type, TypeVar
from abc import ABC, abstractmethod
import logging
import pandas as pd
import os
from datetime import datetime


from database.engine import DatabaseEngine
from models.db_models import User, Product, Order, Base
from constants.db_constants import DatabaseConstants

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
        chunk_size: int | None = None,
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
        super().__init__(engine, User)  # User -> Table name

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
