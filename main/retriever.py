from main import config, weaviate_client
from main.enums import SearchType


class Retriever:
    def search(self, search_type, query):
        if search_type == SearchType.SIMILARITY:
            return self.similarity_search(query)
        elif search_type == SearchType.BM25:
            return self.BM25_search(query)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")

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

    def BM25_search(self, query: str, k: int = 4):
        result = (
            weaviate_client.query.get(
                config.WEAVIATE_CLASS_NAME, config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES
            )
            .with_bm25(query=query)
            .with_additional("score")
            .with_limit(k)
            .do()
        )

        docs = []
        for doc in result["data"]["Get"][config.WEAVIATE_CLASS_NAME]:
            docs.append(doc)

        return docs
