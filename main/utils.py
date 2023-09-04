from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever

from main import config, weaviate_client
from main.chatbot import Chatbot


class Utils:
    weaviate_schema_class = None

    @staticmethod
    def create_vectors():
        vectorstore = WeaviateHybridSearchRetriever(
            client=weaviate_client,
            index_name=config.WEAVIATE_CLASS_NAME,
            text_key=config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES[0],
            alpha=0.5,
            k=4,
            create_schema_if_missing=False,
        )
        return vectorstore

    def setup_chatbot(
        self,
        model="gpt-3.5-turbo",
        temperature=0.0,
    ):
        vectors = self.create_vectors()
        chatbot = Chatbot(model_name=model, temperature=temperature, vectors=vectors)

        return chatbot
