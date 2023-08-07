import argparse
import logging
import time

import datasets
import weaviate

from main import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WeaviateDataManager:
    def __init__(self, url, auth_client_secret, huggingfacehub_api_token):
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=weaviate.AuthApiKey(api_key=auth_client_secret),
            additional_headers={
                "X-HuggingFace-Api-Key": huggingfacehub_api_token,
            },
        )

    @staticmethod
    def load_data(dataset_name="ms_marco", version="v2.1", split="validation"):
        # Default dataset is MS-Marco V2.1 Validation Split
        data = datasets.load_dataset(dataset_name, version, split=split)
        return data

    def create_class_obj(self, class_name):
        class_obj = {
            "class": class_name,
            "vectorizer": "text2vec-huggingface",
            "properties": [
                {
                    "dataType": ["text"],
                    "name": "answer",
                },
                {
                    "dataType": ["text"],
                    "name": "context",
                },
                {
                    "dataType": ["text"],
                    "name": "query",
                },
                {
                    "dataType": ["boolean"],
                    "name": "isSelected",
                },
            ],
            "moduleConfig": {
                "text2vec-huggingface": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "options": {"waitForModel": True},
                },
            },
        }

        logger.info(f"Creating class object: {class_name}")
        self.client.schema.create_class(class_obj)

    def delete_class_obj(self, class_name):
        logger.info(f"Deleting class object: {class_name}")
        self.client.schema.delete_class(class_name)

    def upload(self, data, max_upload_size=1000, batch_size=100, max_rate_limit=200):
        """
        In default `MS-MARCO` dataset, each query has 10 contexts (served as database for searching).
        For each query, create a separate class object. This class will have 10 objects/data points, one for each context.

        TODO: Update logic to handle rate limit reached
        """
        total_uploaded = 0
        with self.client.batch(batch_size=batch_size):
            count = 0
            for i, d in enumerate(data):
                if "No Answer Present." in d["answers"][0]:
                    continue

                if count < max_rate_limit:
                    class_name = f"MSMARCO_{i}"
                    self.create_class_obj(class_name)

                    answer = d["answers"][0]
                    query = d["query"]
                    contexts = d["passages"]["passage_text"]

                    for j, ctx in enumerate(contexts):
                        is_selected = bool(d["passages"]["is_selected"][j])
                        properties = {
                            "answer": answer,
                            "context": ctx,
                            "query": query,
                            "isSelected": is_selected,
                        }
                        count += 1
                        self.client.batch.add_data_object(
                            properties,
                            class_name,
                        )
                        total_uploaded += 1
                        logger.info(f"Uploaded entry: {i}_{j}")
                else:
                    count = 0
                    logger.warning("Rate limit reached. Sleeping for 60 seconds.")
                    time.sleep(60)

                logger.info(f"Uploaded {total_uploaded} entries.")
                if total_uploaded > max_upload_size:
                    break

    def query(self):
        # TODO: query by question with filtering
        pass


def upload_data():
    data_manager = WeaviateDataManager(
        url=config.WEAVIATE_CLUSTER_URL,
        auth_client_secret=config.WEAVIATE_API_KEY,
        huggingfacehub_api_token=config.HUGGINGFACE_API_KEY,
    )
    data = data_manager.load_data()
    data_manager.upload(data)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upload",
        type=bool,
        default=False,
        help="Upload MSMarco validation data to Weaviate",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.upload:
        upload_data()
    pass


if __name__ == "__main__":
    args = get_args()
    main(args)
