"""Environment config, model selection, flags."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # LLM
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.0
    openai_api_key: str = ""
    
    # Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Redis Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_enabled: bool = True
    l2_enabled: bool = True
    l2_threshold: float = 0.92  # cosine similarity; raise if you see wrong hits
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
