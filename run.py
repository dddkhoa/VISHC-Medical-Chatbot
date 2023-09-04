from main import config
from main.retriever import Retriever
from main.utils import Utils

utils = Utils()

config.WEAVIATE_CLASS_NAME = "MedicalDocs"
config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES = ["text"]

chatbot = utils.setup_chatbot()
query = """
Should 8-month-old baby with measles be abstinent and supplemented with calcium?"""
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
