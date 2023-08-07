import os

import pytest

from main import config
from main.weaviate_manager import WeaviateDataManager
from tests.helper import inject_test_data

os.environ["HUGGINGFACEHUB_API_TOKEN"] = config.HUGGINGFACE_API_KEY

manager = WeaviateDataManager(
    url=config.WEAVIATE_CLUSTER_URL,
    auth_client_secret=config.WEAVIATE_API_KEY,
    huggingfacehub_api_token=config.HUGGINGFACE_API_KEY,
)


class TestWeaviate:
    test_data = inject_test_data(file_path="../docs/data_test.json")

    @pytest.mark.parametrize("test_case", test_data)
    def test_successful_retrieve(self, test_case):
        response = (
            manager.client.query.get(test_case["class_name"], ["context"])
            .with_near_text({"concepts": [test_case["query"]]})
            .with_limit(1)
            .with_additional(["distance"])
            .do()
        )

        assert response.status_code == 200
        assert (
            response["data"]["Get"][test_case["class_name"]]["context"]
            == test_case["context"]
        )
