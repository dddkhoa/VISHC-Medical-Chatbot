import logging
import os

from pydantic import BaseSettings


class Config(BaseSettings):
    ENVIRONMENT: str
    LOGGING_LEVEL: int = logging.INFO

    OPENAI_API_KEY: str
    HUGGINGFACEHUB_API_TOKEN: str
    WEAVIATE_ENDPOINT: str
    WEAVIATE_API_KEY: str

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"


environment = os.environ.get("ENVIRONMENT", "local")
config = Config(
    ENVIRONMENT=environment,
    # ".env.{environment}" takes priority over ".env"
    _env_file=[".env", f".env.{environment}"],
)
