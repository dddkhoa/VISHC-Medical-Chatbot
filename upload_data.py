import json
import time

from main import weaviate_client


def callback():
    time.sleep(60)


with open("./docs/vihealthqa/corpus.jsonl", "r") as file:
    data = [json.loads(line) for line in file]


with weaviate_client.batch(
    batch_size=180, dynamic=False, timeout_retries=3, callback=callback
) as batch:
    for i, d in enumerate(data):
        print(f"importing entry: {i}")
        properties = {
            "text": d["text"],
            "doc_id": d["_id"][len("doc") :],
        }

        weaviate_client.batch.add_data_object(
            properties,
            "VietnameseCorpus",
        )
