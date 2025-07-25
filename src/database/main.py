# main.py - Pipeline Runner
import argparse
import sys
import asyncio
from pathlib import Path
from typing import Optional
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from database.manager import DatabaseManager


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
