from enum import Enum, auto, unique


@unique
class BaseEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(name: str, *_):
        """
        Automatically generate values for enum.
        Enum values are lower-cased enum member names.
        """
        return name.lower()

    @classmethod
    def get_values(cls) -> list[str]:
        # noinspection PyUnresolvedReferences
        return [m.value for m in cls]


class SearchType(BaseEnum):
    SIMILARITY = auto()


WEAVIATE_SCHEMA_CLASS = "MSMarco_Toy"
WEAVIATE_TEXT_KEY = "context"
