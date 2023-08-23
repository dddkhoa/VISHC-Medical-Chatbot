from langchain.schema import Document

from main import config, weaviate_client
from main.enums import SearchType


class Retriever:
    def search(self, search_type, query):
        if search_type == SearchType.SIMILARITY:
            result = self.similarity_search(query)
        elif search_type == SearchType.BM25:
            result = self.BM25_search(query)
        elif search_type == SearchType.HYBRID:
            result = self.hybrid_search(query)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")

        docs = []
        for doc in result["data"]["Get"][config.WEAVIATE_CLASS_NAME]:
            text = doc.pop(config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES[0])
            metadata = doc["_additional"]
            for p in config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES[1:]:
                metadata[p] = doc[p]
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

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

        return result

    def BM25_search(self, query: str, k: int = 4):
        result = (
            weaviate_client.query.get(
                config.WEAVIATE_CLASS_NAME, config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES
            )
            .with_bm25(
                query=query, properties=[config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES[0]]
            )
            .with_additional("score")
            .with_limit(k)
            .do()
        )

        return result

    def hybrid_search(self, query: str, k: int = 4):
        result = (
            weaviate_client.query.get(
                config.WEAVIATE_CLASS_NAME, config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES
            )
            .with_hybrid(
                query=query,
                properties=[config.WEAVIATE_RETRIEVED_CLASS_PROPERTIES[0]],
                alpha=0.5,
            )
            .with_additional("score")
            .with_limit(k)
            .do()
        )

        return result
