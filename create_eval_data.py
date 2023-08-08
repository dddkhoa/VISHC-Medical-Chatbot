import json

import datasets

from main import config
from main.retriever import Retriever
from main.utils import Utils

utils = Utils()

chatbot = utils.setup_chatbot()
retriever = Retriever()
data = datasets.load_dataset("ms_marco", "v2.1", split="validation")


def create_data():
    # TODO: Refactor code + handle logic
    for i, d in enumerate(data):
        config.WEAVIATE_CLASS_NAME = f"MSMARCO_{i}"
        config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES = ["context", "isSelected"]

        query = d["query"]
        contexts = d["passages"]["passage_text"]
        ground_truth = d["answers"][0]

        with open("./docs/retrieval_preds.jsonl", "a") as f:
            results = retriever.search(query=query, search_type="hybrid")
            entry = {"query": query, "predictions": []}
            for r in results:
                tmp_pred = {
                    "context": r.page_content,
                    "isSelected": r.metadata["isSelected"],
                }
                entry["predictions"].append(tmp_pred)

            f.write(json.dumps(entry) + "\n")

        with open("./docs/retrieval_grounds.jsonl", "a") as f:
            entry = {"query": query, "ground_truths": []}
            for j, ctx in enumerate(contexts):
                is_selected = bool(d["passages"]["is_selected"][j])
                if is_selected:
                    entry["ground_truths"].append(ctx)

                f.write(json.dumps(entry) + "\n")

        with open("./docs/generator_preds.jsonl", "a") as f:
            answer = chatbot.chat_with_weaviate(query, search_type="hybrid")["result"]
            entry = {"query": query, "predictions": answer}
            f.write(json.dumps(entry) + "\n")

        with open("./docs/generator_grounds.jsonl", "a") as f:
            entry = {"query": query, "ground_truths": ground_truth}
            f.write(json.dumps(entry) + "\n")

        if i >= 8:
            break


if __name__ == "__main__":
    create_data()
