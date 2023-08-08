from langchain.vectorstores import Weaviate

from main import config, weaviate_client
from main.chatbot import Chatbot


class Utils:
    weaviate_schema_class = None

    @staticmethod
    def create_vectors():
        vectorstore = Weaviate(
            weaviate_client,
            config.WEAVIATE_CLASS_NAME,
            config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES[0],
        )
        print(vectorstore)
        return vectorstore

    def setup_chatbot(
        self,
        model="gpt-3.5-turbo",
        temperature=0.0,
    ):
        vectors = self.create_vectors()
        chatbot = Chatbot(model_name=model, temperature=temperature, vectors=vectors)

        return chatbot
