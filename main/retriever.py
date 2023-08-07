from main import config, weaviate_client
from main.enums import SearchType


class Retriever:
    def __init__(self, search_type):
        self.search_type = search_type

    def search(self, query):
        if self.search_type == SearchType.SIMILARITY:
            return self.similarity_search(query)
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

    def similarity_search(self, query: str, k: int = 4):
        result = (
            weaviate_client.query.get(
                config.WEAVIATE_CLASS_NAME, config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES
            )
            .with_near_text({"concepts": [query]})
            .with_limit(k)
            .with_additional(["distance"])
            .do()
        )

        docs = []
        for doc in result["data"]["Get"][config.WEAVIATE_CLASS_NAME]:
            docs.append(doc)

        return docs
