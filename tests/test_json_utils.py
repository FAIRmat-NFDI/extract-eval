from __future__ import annotations

from struct_extract_eval.core.json_utils import (
    get_children,
    get_leaf_paths,
    is_leaf,
    iter_schema,
    resolve_type,
    walk_schema,
)


# --- Shared fixtures ---

FLAT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
}

NESTED_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "experiment": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "temp": {"type": "number"},
            },
        },
    },
}

ARRAY_OF_PRIMITIVES: dict[str, object] = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

ARRAY_OF_OBJECTS: dict[str, object] = {
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "value": {"type": "number"},
                },
            },
        },
    },
}

DEEPLY_NESTED: dict[str, object] = {
    "type": "object",
    "properties": {
        "experiment": {
            "type": "object",
            "properties": {
                "samples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "measurements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "property": {"type": "string"},
                                        "value": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


# --- resolve_type ---


class TestResolveType:
    def test_string_type(self) -> None:
        assert resolve_type({"type": "string"}) == "string"

    def test_object_type(self) -> None:
        assert resolve_type({"type": "object"}) == "object"

    def test_missing_type(self) -> None:
        assert resolve_type({}) is None

    def test_non_string_type(self) -> None:
        assert resolve_type({"type": 42}) is None  # type: ignore[dict-item]


# --- is_leaf ---


class TestIsLeaf:
    def test_string_field(self) -> None:
        assert is_leaf({"type": "string"}) is True

    def test_number_field(self) -> None:
        assert is_leaf({"type": "number"}) is True

    def test_object_with_properties(self) -> None:
        assert is_leaf({"type": "object", "properties": {"x": {"type": "string"}}}) is False

    def test_object_without_properties(self) -> None:
        assert is_leaf({"type": "object"}) is True

    def test_array_with_items(self) -> None:
        assert is_leaf({"type": "array", "items": {"type": "string"}}) is False

    def test_array_without_items(self) -> None:
        assert is_leaf({"type": "array"}) is True

    def test_empty_schema(self) -> None:
        assert is_leaf({}) is True


# --- get_children ---


class TestGetChildren:
    def test_flat_object(self) -> None:
        children = get_children(FLAT_SCHEMA)
        paths = [path for _, path in children]
        assert set(paths) == {"name", "age", "active"}

    def test_child_paths_flat(self) -> None:
        children = get_children(FLAT_SCHEMA)
        paths = [path for _, path in children]
        assert "name" in paths
        assert "age" in paths

    def test_child_paths_with_parent(self) -> None:
        inner = NESTED_SCHEMA["properties"]["experiment"]  # type: ignore[index]
        children = get_children(inner, path="experiment")
        paths = [path for _, path in children]
        assert "experiment.name" in paths
        assert "experiment.temp" in paths

    def test_array_items(self) -> None:
        tags = ARRAY_OF_PRIMITIVES["properties"]["tags"]  # type: ignore[index]
        children = get_children(tags, path="tags")
        assert len(children) == 1
        schema, path = children[0]
        assert path == "tags[]"
        assert schema == {"type": "string"}

    def test_leaf_returns_empty(self) -> None:
        assert get_children({"type": "string"}) == []

    def test_empty_path(self) -> None:
        children = get_children(FLAT_SCHEMA, path="")
        paths = [path for _, path in children]
        assert "name" in paths  # no leading dot


# --- walk_schema ---


class TestWalkSchema:
    def test_flat_visits_all(self) -> None:
        visited: list[str] = []
        walk_schema(FLAT_SCHEMA, lambda _s, p: visited.append(p))
        assert "" in visited  # root
        assert "name" in visited
        assert "age" in visited
        assert "active" in visited

    def test_nested_visits_all(self) -> None:
        visited: list[str] = []
        walk_schema(NESTED_SCHEMA, lambda _s, p: visited.append(p))
        assert "" in visited
        assert "experiment" in visited
        assert "experiment.name" in visited
        assert "experiment.temp" in visited

    def test_array_of_objects(self) -> None:
        visited: list[str] = []
        walk_schema(ARRAY_OF_OBJECTS, lambda _s, p: visited.append(p))
        assert "samples" in visited
        assert "samples[]" in visited
        assert "samples[].id" in visited
        assert "samples[].value" in visited

    def test_deeply_nested(self) -> None:
        visited: list[str] = []
        walk_schema(DEEPLY_NESTED, lambda _s, p: visited.append(p))
        assert "experiment.samples[].measurements[].property" in visited
        assert "experiment.samples[].measurements[].value" in visited

    def test_pre_order(self) -> None:
        """Parent is visited before children."""
        visited: list[str] = []
        walk_schema(NESTED_SCHEMA, lambda _s, p: visited.append(p))
        assert visited.index("experiment") < visited.index("experiment.name")

    def test_can_mutate(self) -> None:
        """walk_schema passes the actual dict, so visitors can mutate."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "x": {"type": "string"},
            },
        }

        def add_marker(node: dict[str, object], path: str) -> None:
            node["_visited"] = True

        walk_schema(schema, add_marker)
        assert schema["_visited"] is True
        x_schema = schema["properties"]["x"]  # type: ignore[index]
        assert x_schema["_visited"] is True


# --- iter_schema ---


class TestIterSchema:
    def test_flat_yields_all(self) -> None:
        paths = [path for _, path in iter_schema(FLAT_SCHEMA)]
        assert set(paths) == {"", "name", "age", "active"}

    def test_deeply_nested_yields_leaves(self) -> None:
        paths = [path for _, path in iter_schema(DEEPLY_NESTED)]
        assert "experiment.samples[].measurements[].property" in paths
        assert "experiment.samples[].measurements[].value" in paths

    def test_consistent_with_walk(self) -> None:
        walk_paths: list[str] = []
        walk_schema(FLAT_SCHEMA, lambda _s, p: walk_paths.append(p))
        iter_paths = [path for _, path in iter_schema(FLAT_SCHEMA)]
        assert walk_paths == iter_paths


# --- get_leaf_paths ---


class TestGetLeafPaths:
    def test_flat(self) -> None:
        leaves = get_leaf_paths(FLAT_SCHEMA)
        assert set(leaves) == {"name", "age", "active"}

    def test_nested(self) -> None:
        leaves = get_leaf_paths(NESTED_SCHEMA)
        assert set(leaves) == {"experiment.name", "experiment.temp"}

    def test_array_of_primitives(self) -> None:
        leaves = get_leaf_paths(ARRAY_OF_PRIMITIVES)
        assert leaves == ["tags[]"]

    def test_array_of_objects(self) -> None:
        leaves = get_leaf_paths(ARRAY_OF_OBJECTS)
        assert set(leaves) == {"samples[].id", "samples[].value"}

    def test_deeply_nested(self) -> None:
        leaves = get_leaf_paths(DEEPLY_NESTED)
        assert set(leaves) == {
            "experiment.samples[].measurements[].property",
            "experiment.samples[].measurements[].value",
        }

    def test_bare_leaf(self) -> None:
        leaves = get_leaf_paths({"type": "string"})
        assert leaves == [""]
