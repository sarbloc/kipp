"""Export entities to JSON/markdown and import from JSON."""

from __future__ import annotations

import json
from typing import TextIO

from kipp.client import DOMAINS
from kipp.models import Entity, Fact


def export_json(by_domain: dict[str, list[Entity]], out: TextIO) -> None:
    """Export entities as one JSON array (no embeddings).

    Every record carries its ``domain`` so ``import_json`` can put it back where
    it came from (issue #65).
    """
    data = [_entity_record(e, domain) for domain, entities in by_domain.items() for e in entities]
    json.dump(data, out, indent=2)
    out.write("\n")


def _entity_record(e: Entity, domain: str) -> dict:
    return {
        "domain": domain,
        "id": e.id,
        "type": e.type,
        "last_updated": e.last_updated,
        "parent": e.parent,
        "facts": [
            {
                "text": f.text,
                "added": f.added,
                "source": f.source,
                "expires": f.expires,
                "last_seen": f.last_seen,
                "hit_count": f.hit_count,
                # Bi-temporal fields (issue #21): without these, a backup
                # round-trip would silently turn superseded history back into
                # current facts and lose valid-time provenance.
                "valid_from": f.valid_from,
                "superseded_at": f.superseded_at,
                "superseded_by": f.superseded_by,
            }
            for f in e.facts
        ],
    }


def export_markdown(by_domain: dict[str, list[Entity]], out: TextIO) -> None:
    """Export entities as human-readable markdown, under a heading per domain when there are several."""
    headed = len(by_domain) > 1
    for domain, entities in by_domain.items():
        if headed:
            out.write(f"# {domain}\n\n")
        for e in entities:
            out.write(f"## {e.id}\n")
            for f in e.facts:
                expires = f" [expires {f.expires}]" if f.expires else ""
                superseded = f" [superseded {f.superseded_at}]" if f.superseded_at else ""
                out.write(
                    f"- {f.text} (x{f.hit_count}, since {f.added}){expires}{superseded}\n"
                )
            out.write("\n")


def reject_future_dated_facts(entities: list[Entity], today: str) -> None:
    """Raise ``ValueError`` if any fact carries a future valid-time date.

    Future-effective dating isn't supported yet (issue #24): the default
    ``is_current`` view assumes a fact is valid exactly when it isn't superseded,
    which only holds when neither end of its valid-time window is in the future.
    A future ``valid_from`` (start) would surface a not-yet-true fact as live; a
    future ``superseded_at`` (end) would hide a still-true fact. Import enforces
    the same invariant the store path guards at write time, so a hand-edited
    backup can't corrupt the current view.
    """
    for e in entities:
        for f in e.facts:
            for label, value in (
                ("valid_from", f.valid_from),
                ("superseded_at", f.superseded_at),
            ):
                if value is not None and value[:10] > today:
                    raise ValueError(
                        f"backup has a future {label} {value!r} on {e.id} (today "
                        f"is {today}); future-effective dating is not supported "
                        f"yet (issue #24)"
                    )


def import_json(data: list[dict]) -> list[tuple[str, Entity]]:
    """Parse a JSON export back into ``(domain, Entity)`` pairs.

    A record without a ``domain`` field (an export from before issue #65) is
    ``shared``, the only domain those exports covered. An unknown domain raises
    ``ValueError`` before anything is returned, so a bad file can't half-import.
    """
    entities = []
    for item in data:
        domain = item.get("domain", "shared")
        if domain not in DOMAINS:
            raise ValueError(
                f"record {item.get('id')!r} has unknown domain {domain!r}; "
                f"expected one of {DOMAINS}"
            )
        facts = [
            Fact(
                text=f["text"],
                added=f["added"],
                source=f["source"],
                expires=f.get("expires"),
                last_seen=f.get("last_seen"),
                hit_count=f.get("hit_count", 1),
                # .get() so pre-#21 backups (without these keys) still import.
                valid_from=f.get("valid_from"),
                superseded_at=f.get("superseded_at"),
                superseded_by=f.get("superseded_by"),
            )
            for f in item.get("facts", [])
        ]
        entities.append((domain, Entity(
            id=item["id"],
            type=item["type"],
            facts=facts,
            last_updated=item.get("last_updated", ""),
            parent=item.get("parent"),
        )))
    return entities
