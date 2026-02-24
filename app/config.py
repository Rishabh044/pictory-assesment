"""Configuration settings for the semantic search system."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str
    chroma_persist_directory: str = "./data/chroma_db"
    embedding_model: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(env_file=".env")


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
