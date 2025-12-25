"""
Ontology loader and helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import CONFIG


class Ontology:
    """Represents ontology schema and helpers."""

    def __init__(self, path: str | Path | None = None) -> None:
        path = path or CONFIG.ontology_path
        data = json.loads(Path(path).read_text())
        self.entity_types: Dict[str, Dict[str, object]] = data.get("entity_types", {})
        self.relation_types: Dict[str, Dict[str, object]] = data.get("relation_types", {})
        self.constraints = data.get("constraints", {})
        self.intent_bias = data.get("intent_bias", {})
        self.entity_synonyms = self._build_synonym_map(self.entity_types)
        self.relation_synonyms = self._build_synonym_map(self.relation_types)

    @staticmethod
    def _build_synonym_map(items: Dict[str, Dict[str, object]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for name, info in items.items():
            mapping[name.lower()] = name
            for syn in info.get("synonyms", []):
                mapping[str(syn).lower()] = name
        return mapping

    def normalize_entity_type(self, raw: str) -> Optional[str]:
        return self.entity_synonyms.get(raw.lower())

    def normalize_relation_type(self, raw: str) -> Optional[str]:
        return self.relation_synonyms.get(raw.lower())

    def allowed_relation(self, relation: str, domain: str, rng: str) -> bool:
        rel = self.relation_types.get(relation)
        if not rel:
            return False
        domains = rel.get("domain", [])
        ranges = rel.get("range", [])
        return domain in domains and rng in ranges

    def required_attrs(self, entity_type: str) -> List[str]:
        return self.constraints.get("required_attributes", {}).get(entity_type, [])

    def co_occurrence_penalty(self) -> float:
        return float(self.constraints.get("validation", {}).get("co_occurrence_penalty", 0.0))

    def edge_confidence_penalty(self) -> float:
        return float(self.constraints.get("validation", {}).get("edge_confidence_penalty", 0.0))

    def bias_entities_for_intent(self, intent: str) -> List[str]:
        return self.intent_bias.get(intent, self.intent_bias.get("default", []))
