from dataclasses import dataclass


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
