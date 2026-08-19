"""
Persistent storage for TreeIndex objects.

Tree indexes are generated offline and saved as JSON.

The storage layer knows nothing about:
- Ollama
- retrieval
- prompts
- Records
- Flask

Its only responsibility is:

TreeIndex -> JSON file
JSON file -> TreeIndex
"""

import json
from pathlib import Path

from config.settings import PROJECT_ROOT
from indexing.tree_schema import TreeIndex


INDEX_DIR = PROJECT_ROOT / "data" / "indexes"


class TreeStorage:
    """
    Save and load persistent TreeIndex objects.
    """

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _get_path(self, source: str) -> Path:
        if not source.strip():
            raise ValueError("Source cannot be empty.")

        return self.index_dir / f"{source}_tree.json"

    def save(self, tree: TreeIndex) -> Path:

        tree.validate()

        path = self._get_path(tree.source)

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                tree.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

        return path

    def load(self, source: str) -> TreeIndex:

        path = self._get_path(source)

        if not path.exists():
            raise FileNotFoundError(
                f"No tree index exists for source '{source}'. "
                f"Expected file: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        tree = TreeIndex.from_dict(data)

        tree.validate()

        return tree

    def exists(self, source: str) -> bool:
        return self._get_path(source).exists()