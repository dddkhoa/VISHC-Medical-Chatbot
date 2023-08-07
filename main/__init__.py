import weaviate

from main._config import config

weaviate_auth_config = weaviate.AuthApiKey(api_key=config.WEAVIATE_API_KEY)
weaviate_client = weaviate.Client(
    url=config.WEAVIATE_CLUSTER_URL,
    auth_client_secret=weaviate_auth_config,
    additional_headers={
        "X-OpenAI-Api-Key": config.OPENAI_API_KEY,
        "X-HuggingFace-Api-Key": config.HUGGINGFACE_API_KEY,
    },
)
