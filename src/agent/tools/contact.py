"""Contact writes: the only side effects of Flow 1, performed by the graph.

The model never reaches these (rule 3). `ContactStore` is the seam the
Postgres implementation will fill; `InMemoryContactStore` serves tests and
the text demo, and records every write so a test can assert on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class ContactStore(Protocol):
    async def update_contact(self, contact_id: str, field: str, value: str) -> None:
        """Write a value the caller confirmed and mark the field validated."""

    async def mark_unvalidated(self, contact_id: str, field: str) -> None:
        """Flag a field for asynchronous human review; the stored value is kept."""


@dataclass
class InMemoryContactStore:
    contacts: dict[str, dict[str, str]] = field(default_factory=dict)
    writes: list[tuple[str, str, str, str | None]] = field(default_factory=list)

    async def update_contact(self, contact_id: str, field: str, value: str) -> None:
        self.contacts.setdefault(contact_id, {})[field] = value
        self.writes.append(("update_contact", contact_id, field, value))

    async def mark_unvalidated(self, contact_id: str, field: str) -> None:
        self.writes.append(("mark_unvalidated", contact_id, field, None))
