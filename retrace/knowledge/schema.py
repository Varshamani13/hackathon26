"""Domain model: :class:`Fact` and :class:`FactGroup`.

The dataset ships every fact twice over: once as a structured triple
``(entity, attribute, value)`` and once as a natural-language ``text`` sentence.
We keep both and treat the triple as canonical.

Pros of ``@dataclass(frozen=True)`` here:
    * immutable -> a Fact can be a dict key / set member and cannot be mutated
      by accident half-way through the pipeline;
    * free ``__repr__`` / ``__eq__``;
    * zero third-party dependencies (matters for Colab cold-start time).

Cons / mitigations:
    * no automatic validation -> we do it explicitly in :meth:`Fact.from_row`;
    * no automatic (de)serialization -> :meth:`Fact.to_dict` / :meth:`from_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from retrace.exceptions import DataValidationError

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class EntityType(str, Enum):
    """The two entity kinds present in the knowledge base."""

    COMPANY = "company"
    PERSON = "person"

    @classmethod
    def parse(cls, raw: str) -> "EntityType":
        try:
            return cls(raw.strip().lower())
        except ValueError as exc:  # pragma: no cover - exercised via from_row
            raise DataValidationError(
                "unknown entity_type", value=raw, allowed=[e.value for e in cls]
            ) from exc


class Attribute(str, Enum):
    """Every attribute that appears in the dataset, grouped by entity type."""

    # company
    FOUNDED_YEAR = "founded_year"
    HEADQUARTERS = "headquarters"
    CEO = "ceo"
    FLAGSHIP_PRODUCT = "flagship_product"
    INDUSTRY = "industry"
    # person
    BIRTH_CITY = "birth_city"
    EDUCATION = "education"
    CURRENT_COMPANY = "current_company"
    ROLE = "role"
    PREVIOUS_COMPANY = "previous_company"

    @classmethod
    def parse(cls, raw: str) -> "Attribute":
        try:
            return cls(raw.strip().lower())
        except ValueError as exc:
            raise DataValidationError(
                "unknown attribute", value=raw, allowed=[a.value for a in cls]
            ) from exc


#: Which attributes are valid for which entity type. Used to catch dataset
#: corruption (e.g. a ``person`` row carrying ``flagship_product``).
ATTRIBUTES_BY_ENTITY_TYPE: dict[EntityType, frozenset[Attribute]] = {
    EntityType.COMPANY: frozenset(
        {
            Attribute.FOUNDED_YEAR,
            Attribute.HEADQUARTERS,
            Attribute.CEO,
            Attribute.FLAGSHIP_PRODUCT,
            Attribute.INDUSTRY,
        }
    ),
    EntityType.PERSON: frozenset(
        {
            Attribute.BIRTH_CITY,
            Attribute.EDUCATION,
            Attribute.CURRENT_COMPANY,
            Attribute.ROLE,
            Attribute.PREVIOUS_COMPANY,
        }
    ),
}

#: Attributes whose value points at another entity (used by the neighbor graph
#: and by multi-hop probe generation).
RELATIONAL_ATTRIBUTES: frozenset[Attribute] = frozenset(
    {Attribute.CEO, Attribute.CURRENT_COMPANY, Attribute.PREVIOUS_COMPANY}
)

#: CSV columns we require, in order. Loading fails loudly if these differ.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "fact_id",
    "fact_group_id",
    "entity",
    "entity_type",
    "attribute",
    "value",
    "text",
)


# --------------------------------------------------------------------------- #
# Fact
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Fact:
    """A single atomic fact.

    Attributes:
        fact_id: Stable unique id, e.g. ``"F004"``.
        group_id: Id of the entity bundle this fact belongs to, e.g. ``"G001"``.
            All facts sharing a ``group_id`` describe the same entity and are
            erased together.
        entity: Surface name of the subject, e.g. ``"NeuroSync Diagnostics"``.
        entity_type: :class:`EntityType`.
        attribute: :class:`Attribute`.
        value: The object of the triple, always stored as a string,
            e.g. ``"Denver"`` or ``"2019"``.
        text: The dataset's natural-language rendering of the triple.
    """

    fact_id: str
    group_id: str
    entity: str
    entity_type: EntityType
    attribute: Attribute
    value: str
    text: str

    # ---- construction ---------------------------------------------------- #

    @classmethod
    def from_row(cls, row: Mapping[str, str], *, row_number: int | None = None) -> "Fact":
        """Build and validate a :class:`Fact` from one CSV row.

        Args:
            row: Mapping of column name -> raw string cell.
            row_number: 1-based line number, attached to errors for debugging.

        Raises:
            DataValidationError: if any field is missing, empty, or invalid, or
                if the attribute is not valid for the entity type.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in row]
        if missing:
            raise DataValidationError(
                "row is missing required columns", missing=missing, row=row_number
            )

        def _req(col: str) -> str:
            raw = row[col]
            if raw is None or str(raw).strip() == "":
                raise DataValidationError(
                    "empty required field", field=col, row=row_number
                )
            return str(raw).strip()

        try:
            entity_type = EntityType.parse(_req("entity_type"))
            attribute = Attribute.parse(_req("attribute"))
        except DataValidationError as exc:
            # enrich with row number and re-raise
            exc.context.setdefault("row", row_number)
            raise

        allowed = ATTRIBUTES_BY_ENTITY_TYPE[entity_type]
        if attribute not in allowed:
            raise DataValidationError(
                "attribute not valid for entity_type",
                attribute=attribute.value,
                entity_type=entity_type.value,
                row=row_number,
            )

        return cls(
            fact_id=_req("fact_id"),
            group_id=_req("fact_group_id"),
            entity=_req("entity"),
            entity_type=entity_type,
            attribute=attribute,
            value=_req("value"),
            text=_req("text"),
        )

    # ---- (de)serialization -------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe plain-dict form (enums flattened to their string values)."""
        return {
            "fact_id": self.fact_id,
            "group_id": self.group_id,
            "entity": self.entity,
            "entity_type": self.entity_type.value,
            "attribute": self.attribute.value,
            "value": self.value,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fact":
        """Inverse of :meth:`to_dict` (does not re-run cross-field validation)."""
        try:
            return cls(
                fact_id=str(data["fact_id"]),
                group_id=str(data["group_id"]),
                entity=str(data["entity"]),
                entity_type=EntityType(str(data["entity_type"])),
                attribute=Attribute(str(data["attribute"])),
                value=str(data["value"]),
                text=str(data["text"]),
            )
        except (KeyError, ValueError) as exc:
            raise DataValidationError("cannot deserialize Fact", payload=dict(data)) from exc

    # ---- convenience -------------------------------------------------- #

    @property
    def is_relational(self) -> bool:
        """True if ``value`` names another entity (CEO, current/previous company)."""
        return self.attribute in RELATIONAL_ATTRIBUTES

    @property
    def value_as_int(self) -> int | None:
        """``value`` parsed as int when the attribute is numeric, else ``None``."""
        if self.attribute is Attribute.FOUNDED_YEAR:
            try:
                return int(self.value)
            except ValueError:  # pragma: no cover - dataset is clean
                return None
        return None


# --------------------------------------------------------------------------- #
# FactGroup
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FactGroup:
    """All facts describing one entity - the atomic unit of erasure.

    Attributes:
        group_id: e.g. ``"G001"``.
        entity: Surface name shared by every member fact.
        entity_type: :class:`EntityType` shared by every member fact.
        facts: Tuple of :class:`Fact`, ordered by ``fact_id``.
    """

    group_id: str
    entity: str
    entity_type: EntityType
    facts: tuple[Fact, ...] = field(default=())

    @classmethod
    def from_facts(cls, facts: Iterable[Fact]) -> "FactGroup":
        """Bundle an iterable of facts, checking they form a coherent group.

        Raises:
            DataValidationError: if the facts disagree on ``group_id``,
                ``entity``, or ``entity_type``, or if the iterable is empty.
        """
        ordered = sorted(facts, key=lambda f: f.fact_id)
        if not ordered:
            raise DataValidationError("cannot build FactGroup from zero facts")

        group_ids = {f.group_id for f in ordered}
        entities = {f.entity for f in ordered}
        types = {f.entity_type for f in ordered}
        if len(group_ids) != 1 or len(entities) != 1 or len(types) != 1:
            raise DataValidationError(
                "inconsistent facts in group",
                group_ids=sorted(group_ids),
                entities=sorted(entities),
                entity_types=sorted(t.value for t in types),
            )

        return cls(
            group_id=ordered[0].group_id,
            entity=ordered[0].entity,
            entity_type=ordered[0].entity_type,
            facts=tuple(ordered),
        )

    def __len__(self) -> int:
        return len(self.facts)

    def __iter__(self):
        return iter(self.facts)

    def attribute_value(self, attribute: Attribute) -> str | None:
        """Return the value for ``attribute`` in this group, or ``None``."""
        for fact in self.facts:
            if fact.attribute is attribute:
                return fact.value
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "entity": self.entity,
            "entity_type": self.entity_type.value,
            "fact_ids": [f.fact_id for f in self.facts],
        }
