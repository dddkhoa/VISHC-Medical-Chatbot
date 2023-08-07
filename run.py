from main.retriever import Retriever
from main.utils import Utils

utils = Utils()

chatbot = utils.setup_chatbot()
query = """
What was the average copeptin concentration in the group of patients with acute myocardial infarction (AMI) and ST-segment elevation (NMCT ST elevation), compared to the group of patients with AMI but without ST-segment elevation (NMCT non-ST elevation) and the healthy control group?
"""
retriever = Retriever(search_type="similarity")
result = retriever.search(query=query)
print(result)
# answer = chatbot.chat(query)
# print(f"query:{query}")
# print(f"\nen_answer:\n{answer}")
# print(f"\nvi_answer:\n{answer['vi_answer']}")
