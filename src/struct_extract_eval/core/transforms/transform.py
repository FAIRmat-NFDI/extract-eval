from typing import Any, Protocol


class Transform(Protocol):
    """Interface for transform functions.

    A transform preprocesses a value before comparison.
    Applied to both gold and extracted values.
    """

    def __call__(self, value: Any, params: dict[str, Any]) -> Any: ...
