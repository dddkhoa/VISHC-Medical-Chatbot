import logging
import os

from pydantic import BaseSettings


class Config(BaseSettings):
    ENVIRONMENT: str
    LOGGING_LEVEL: int = logging.INFO

    OPENAI_API_KEY: str

    HUGGINGFACE_API_KEY: str

    # Weaviate
    WEAVIATE_API_KEY: str
    WEAVIATE_CLUSTER_URL: str

    WEAVIATE_CLASS_NAME: str = "MedicalDocs"
    WEAVIATE_RETRIEVED_CLASS_PROPERTIES: list = ["en"]
    WEAVIATE_ANSWER_FORMAT: str = "_additional {answer {result}}"

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"


environment = os.environ.get("ENVIRONMENT", "local")
config = Config(
    ENVIRONMENT=environment,
    # ".env.{environment}" takes priority over ".env"
    _env_file=[".env", f".env.{environment}"],
)
