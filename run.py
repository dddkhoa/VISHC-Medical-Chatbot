from main import config
from main.retriever import Retriever
from main.utils import Utils

utils = Utils()

config.WEAVIATE_CLASS_NAME = "VietnameseCorpus"
config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES = ["text", "doc_id"]

chatbot = utils.setup_chatbot()
query = """
viêm gan B"""
retriever = Retriever()
result = retriever.search(query=query, search_type="hybrid")
for r in result:
    print(r)

# answer = chatbot.chat_with_weaviate(query, search_type="hybrid")
# print(f"query:{query}")
# print(f"result:\n{answer['result']}\n")
# print("source_documents\n")
# for doc in answer["source_documents"]:
#     print(doc)
#     print()
