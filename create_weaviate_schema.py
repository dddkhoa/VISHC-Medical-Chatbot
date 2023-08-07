import argparse

import weaviate

from main import config


def create_weaviate_schema(is_recreate=False):
    weaviate_auth_config = weaviate.AuthApiKey(api_key=config.WEAVIATE_API_KEY)
    weaviate_client = weaviate.Client(
        url=config.WEAVIATE_CLUSTER_URL,
        auth_client_secret=weaviate_auth_config,
        additional_headers={
            "X-OpenAI-Api-Key": config.OPENAI_API_KEY,
            "X-HuggingFace-Api-Key": config.HUGGINGFACE_API_KEY,
        },
    )
    if is_recreate:
        weaviate_client.schema.delete_all()
    try:
        weaviate_client.schema.create_class("./schema/medical_docs.json")
    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", dest="is_recreate", action="store_true")

    args = parser.parse_args()
    print(args.is_recreate)
    create_weaviate_schema(args.is_recreate)


if __name__ == "__main__":
    main()
