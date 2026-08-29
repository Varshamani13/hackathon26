"""Load and validate the knowledge-base CSV into the domain model.

Public API:
    load_dataset(path, *, strict=True) -> Dataset

``Dataset`` is a thin, queryable container over the facts: it indexes by
``fact_id``, ``group_id``, ``entity``, and ``(attribute, value)`` so downstream
phases (probe generation, neighbor graph, erasure-request resolution) never have
to re-scan the list.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from retrace.exceptions import CSVSchemaError, DataValidationError
from retrace.knowledge.schema import (
    REQUIRED_COLUMNS,
    Attribute,
    EntityType,
    Fact,
    FactGroup,
)

logger = logging.getLogger("retrace.dataset")


@dataclass(slots=True)
class Dataset:
    """In-memory, indexed view of the knowledge base.

    Prefer the accessor methods over touching ``facts`` directly - they are O(1)
    where the raw list would be O(n).
    """

    facts: tuple[Fact, ...]
    groups: tuple[FactGroup, ...]

    # private indexes (built in __post_init__)
    _by_fact_id: dict[str, Fact]
    _by_group_id: dict[str, FactGroup]
    _by_entity: dict[str, FactGroup]
    _entities_by_value: dict[tuple[Attribute, str], list[str]]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_facts(cls, facts: Iterable[Fact]) -> "Dataset":
        facts = tuple(facts)
        if not facts:
            raise DataValidationError("dataset is empty")

        by_fact_id: dict[str, Fact] = {}
        for f in facts:
            if f.fact_id in by_fact_id:
                raise DataValidationError("duplicate fact_id", fact_id=f.fact_id)
            by_fact_id[f.fact_id] = f

        grouped: dict[str, list[Fact]] = defaultdict(list)
        for f in facts:
            grouped[f.group_id].append(f)
        groups = tuple(
            FactGroup.from_facts(members)
            for _, members in sorted(grouped.items())
        )

        by_group_id = {g.group_id: g for g in groups}

        by_entity: dict[str, FactGroup] = {}
        for g in groups:
            if g.entity in by_entity:
                # Two groups with the same surface name would make
                # entity-name erasure requests ambiguous. The challenging
                # dataset avoids this, but we assert it rather than assume it.
                raise DataValidationError(
                    "two fact groups share an entity name",
                    entity=g.entity,
                    group_ids=[by_entity[g.entity].group_id, g.group_id],
                )
            by_entity[g.entity] = g

        entities_by_value: dict[tuple[Attribute, str], list[str]] = defaultdict(list)
        for f in facts:
            entities_by_value[(f.attribute, f.value)].append(f.entity)

        return cls(
            facts=facts,
            groups=groups,
            _by_fact_id=by_fact_id,
            _by_group_id=by_group_id,
            _by_entity=by_entity,
            _entities_by_value={k: sorted(set(v)) for k, v in entities_by_value.items()},
        )

    def __post_init__(self) -> None:  # pragma: no cover - dataclass plumbing
        pass

    # ---- sizes -------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.facts)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    # ---- lookups ---------------------------------------------------- #
    def fact(self, fact_id: str) -> Fact:
        try:
            return self._by_fact_id[fact_id]
        except KeyError as exc:
            raise DataValidationError("unknown fact_id", fact_id=fact_id) from exc

    def group(self, group_id: str) -> FactGroup:
        try:
            return self._by_group_id[group_id]
        except KeyError as exc:
            raise DataValidationError("unknown group_id", group_id=group_id) from exc

    def group_for_entity(self, entity: str) -> FactGroup:
        """Resolve a surface entity name (case-insensitive) to its group."""
        hit = self._by_entity.get(entity)
        if hit is not None:
            return hit
        lowered = entity.strip().lower()
        for name, grp in self._by_entity.items():
            if name.lower() == lowered:
                return grp
        raise DataValidationError("unknown entity", entity=entity)

    def entities_with_value(self, attribute: Attribute, value: str) -> list[str]:
        """All entity names asserting ``attribute == value`` (sorted, unique)."""
        return list(self._entities_by_value.get((attribute, value), []))

    def iter_groups(self) -> Iterator[FactGroup]:
        return iter(self.groups)

    # ---- stats ---------------------------------------------------- #
    def summary(self) -> dict[str, object]:
        by_type: dict[str, int] = defaultdict(int)
        by_attr: dict[str, int] = defaultdict(int)
        for f in self.facts:
            by_type[f.entity_type.value] += 1
            by_attr[f.attribute.value] += 1
        collisions = {
            f"{attr.value}={val}": ents
            for (attr, val), ents in self._entities_by_value.items()
            if len(ents) > 1
        }
        return {
            "n_facts": len(self.facts),
            "n_groups": self.n_groups,
            "facts_by_entity_type": dict(by_type),
            "facts_by_attribute": dict(by_attr),
            "shared_value_count": len(collisions),
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _validate_header(header: Sequence[str] | None, *, path: Path) -> None:
    if header is None:
        raise CSVSchemaError("CSV has no header row", path=str(path))
    normalized = [h.strip() for h in header]
    if tuple(normalized) != REQUIRED_COLUMNS:
        raise CSVSchemaError(
            "CSV header does not match expected schema",
            path=str(path),
            expected=list(REQUIRED_COLUMNS),
            found=normalized,
        )


def load_dataset(
    path: str | Path = "data/knowledge_challenging_500.csv",
    *,
    strict: bool = True,
    encoding: str = "utf-8",
) -> Dataset:
    """Read the knowledge-base CSV and return an indexed :class:`Dataset`.

    Args:
        path: Path to the CSV file.
        strict: If ``True`` (default), the first invalid row raises. If
            ``False``, invalid rows are logged at WARNING and skipped.
        encoding: Text encoding of the file.

    Returns:
        A fully-indexed :class:`Dataset`.

    Raises:
        ArtifactError: wrapped file-not-found / OS errors.
        CSVSchemaError: header mismatch or empty file.
        DataValidationError: a row failed validation (only in ``strict`` mode),
            or the resulting dataset is structurally inconsistent.
    """
    from retrace.exceptions import ArtifactError  # local import avoids cycle at top

    path = Path(path)
    try:
        handle = path.open("r", newline="", encoding=encoding)
    except OSError as exc:
        raise ArtifactError("cannot open dataset CSV", path=str(path)) from exc

    facts: list[Fact] = []
    skipped: list[tuple[int, str]] = []
    with handle:
        reader = csv.DictReader(handle)
        _validate_header(reader.fieldnames, path=path)

        for line_number, row in enumerate(reader, start=2):  # row 1 is the header
            try:
                facts.append(Fact.from_row(row, row_number=line_number))
            except DataValidationError as exc:
                if strict:
                    raise
                skipped.append((line_number, str(exc)))
                logger.warning("skipping row %d: %s", line_number, exc)

    if not facts:
        raise CSVSchemaError("no data rows found in CSV", path=str(path))

    if skipped:
        logger.warning("loaded %d facts, skipped %d rows", len(facts), len(skipped))

    dataset = Dataset.from_facts(facts)
    logger.info(
        "loaded dataset: %d facts / %d groups from %s",
        len(dataset),
        dataset.n_groups,
        path,
    )
    return dataset
