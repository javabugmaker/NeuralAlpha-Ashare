from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..data.store import ParquetStore


class ModelRegistry:
    """Explicit champion/challenger registry; never silently replaces a model."""

    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self.path = self.models_dir / "registry.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"champion": None, "challengers": [], "history": []}
        with self.path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def register(
        self, artifact: str | Path, metadata: dict[str, Any], metrics: dict[str, Any]
    ) -> str:
        registry = self.read()
        entry = {
            "artifact": str(Path(artifact).resolve()),
            "metadata": metadata,
            "metrics": metrics,
            "registered_at": datetime.now(UTC).isoformat(),
        }
        if registry["champion"] is None:
            registry["champion"] = entry
            role = "champion"
        else:
            registry["challengers"].append(entry)
            role = "challenger"
        registry["history"].append({"action": "register", "role": role, **entry})
        ParquetStore.atomic_json(registry, self.path)
        return role

    def promote(self, artifact: str | Path, reason: str) -> None:
        registry = self.read()
        resolved = str(Path(artifact).resolve())
        match = next(
            (item for item in registry["challengers"] if item["artifact"] == resolved), None
        )
        if match is None:
            raise KeyError(f"challenger not found: {resolved}")
        old = registry["champion"]
        registry["challengers"] = [
            item for item in registry["challengers"] if item["artifact"] != resolved
        ]
        if old is not None:
            registry["challengers"].append(old)
        registry["champion"] = match
        registry["history"].append(
            {
                "action": "promote",
                "artifact": resolved,
                "reason": reason,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        ParquetStore.atomic_json(registry, self.path)

    def champion_artifact(self) -> Path:
        champion = self.read()["champion"]
        if champion is None:
            raise FileNotFoundError("no champion model; run alpha-ashare train")
        return Path(champion["artifact"])
