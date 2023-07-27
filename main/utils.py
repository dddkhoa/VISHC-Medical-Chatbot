from langchain.document_loaders import JSONLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from main.chatbot import Chatbot


class Utils:
    @staticmethod
    def load_data():
        loader = JSONLoader(file_path="./docs/data_mini.json", jq_schema=".[].en")

        data = loader.load()

        return data

    @staticmethod
    def create_vectors(data):
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        embeddings = OpenAIEmbeddings()
        docs = splitter.split_documents(data)
        db = Chroma.from_documents(docs, embeddings)

        return db

    def setup_chatbot(self, model="gpt-3.5-turbo", temperature=0.0):
        data = self.load_data()
        vectors = self.create_vectors(data)

        chatbot = Chatbot(model_name=model, temperature=temperature, vectors=vectors)

        return chatbot
