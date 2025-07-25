# database_manager.py
import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from sqlalchemy import select


from database.engine import MultiDatabaseManager
from database.operations import DatabaseOperations, ProductOperations, UserOperations, OrderOperations
from config.db_config import DatabaseConfig
from constants.db_constants import DatabaseConstants

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
            # await self.multi_db_manager.create_all_tables()

            # Setup operation classes
            self.operations = {
                "users": UserOperations(self.multi_db_manager.get_engine("users_db")),  # users_db -> databasename
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
