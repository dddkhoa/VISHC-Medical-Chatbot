import json

import weaviate

from main import config

with open("./docs/data_mini.json", "r") as file:
    data = json.load(file)

weaviate_auth_config = weaviate.AuthApiKey(api_key=config.WEAVIATE_API_KEY)
weaviate_client = weaviate.Client(
    url=config.WEAVIATE_CLUSTER_URL,
    auth_client_secret=weaviate_auth_config,
    additional_headers={
        "X-OpenAI-Api-Key": config.OPENAI_API_KEY,
        "X-HuggingFace-Api-Key": config.HUGGINGFACE_API_KEY,
    },
)

with weaviate_client.batch(batch_size=100) as batch:
    # Batch import all Questions
    for i, d in enumerate(data):
        print(f"importing entry: {i}")
        properties = {
            "url": d["url"],
            "crawl_date": d["crawl_date"],
            "en": d["en"],
            "vi": d["vi"],
        }

        weaviate_client.batch.add_data_object(
            properties,
            "MedicalDocs",
        )
