import argparse

import weaviate

from main import config


def create_weaviate_schema(schema_path, is_recreate):
    weaviate_auth_config = weaviate.AuthApiKey(api_key=config.WEAVIATE_API_KEY)
    weaviate_client = weaviate.Client(
        url=config.WEAVIATE_CLUSTER_URL,
        auth_client_secret=weaviate_auth_config,
        additional_headers={
            "X-OpenAI-Api-Key": config.OPENAI_API_KEY,
            "X-HuggingFace-Api-Key": config.HUGGINGFACEHUB_API_TOKEN,
        },
    )
    if is_recreate:
        weaviate_client.schema.delete_all()
    try:
        weaviate_client.schema.create_class(schema_path)
    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema_path",
        type=str,
        required=True,
        default="./schema/vietnamese_corpus.json",
    )
    parser.add_argument("--is_recreate", type=bool, default=False)

    args = parser.parse_args()
    create_weaviate_schema(args.schema_path, args.is_recreate)


if __name__ == "__main__":
    main()
