import json

from main import config
from main.retriever import Retriever
from main.utils import Utils

utils = Utils()

chatbot = utils.setup_chatbot()
retriever = Retriever()

with open("../docs/vihealthqa/en_vi_queries.jsonl", "r") as file:
    data = [json.loads(line) for line in file]


with open("../docs/vihealthqa/qrels/test.tsv", "r") as file:
    file.readline()
    test_data = {}
    for line in file.readlines():
        line = line.split("\t")
        test_data[line[0]] = line[1][len("doc") :]


def find_doc_with_id(doc_id):
    result = None
    with open("../docs/vihealthqa/corpus.jsonl", "r") as file:
        tmp_data = [json.loads(line) for line in file]
        for d in tmp_data:
            if d["_id"][len("doc") :] == doc_id:
                result = d["text"]

    return result


def create_data():
    # TODO: Refactor code + handle logi
    config.WEAVIATE_CLASS_NAME = "VietnameseCorpus"
    config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES = ["text", "doc_id"]
    count = 815
    for i, d in enumerate(data[825:]):
        query = d["en_text"]
        query_id = d["_id"]
        doc_id = test_data[query_id]
        ground_truth = find_doc_with_id(doc_id)
        if ground_truth:
            with open("../docs/vihealthqa/results/retrieval_preds.jsonl", "a") as f:
                results = retriever.search(query=query, search_type="hybrid")
                entry = {"query": query, "predictions": []}
                for r in results:
                    tmp_pred = {
                        "context": r.page_content,
                    }
                    entry["predictions"].append(tmp_pred)

                f.write(json.dumps(entry) + "\n")
            count += 1
            # with open("./docs/vihealthqa/results/retrieval_grounds.jsonl", "a") as f:
            #     entry = {"query": query, "ground_truths": []}
            #     entry["ground_truths"].append(ground_truth)
            #
            #     f.write(json.dumps(entry) + "\n")

        # with open("./docs/generator_preds.jsonl", "a") as f:
        #     answer = chatbot.chat_with_weaviate(query, search_type="hybrid")["result"]
        #     entry = {"query": query, "predictions": answer}
        #     f.write(json.dumps(entry) + "\n")
        #
        # with open("./docs/generator_grounds.jsonl", "a") as f:
        #     entry = {"query": query, "ground_truths": ground_truth}
        #     f.write(json.dumps(entry) + "\n")

        if count >= 1000:
            break


if __name__ == "__main__":
    create_data()
