# db_models.py
from sqlalchemy import Column, Integer, String, DateTime, Numeric, Text, Boolean

# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func
from datetime import datetime


# Base class for all models
# Base = declarative_base()
class Base(DeclarativeBase):
    pass


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


# __all__ = ["User", "Product", "Order", "Base"]
