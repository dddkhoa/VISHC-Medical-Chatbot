from main.utils import Utils

utils = Utils()

chatbot = utils.setup_chatbot()
query = """
What was the average copeptin concentration in the group of patients with acute myocardial infarction (AMI) and ST-segment elevation (NMCT ST elevation), compared to the group of patients with AMI but without ST-segment elevation (NMCT non-ST elevation) and the healthy control group?
"""
# retriever = Retriever()
# result = retriever.search(query=query, search_type="hybrid")
# print(result)
answer = chatbot.chat_with_weaviate(query, search_type="hybrid")
print(f"query:{query}")
print(f"result:\n{answer['result']}\n")
print("source_documents\n")
for doc in answer["source_documents"]:
    print(doc)
    print()
