"""Notes that persist across runs in memory/.

Each ticker has memory/<TICKER>.json; lessons that don't depend on the
ticker (mostly about tools) go to memory/general.json. A record is
{"date", "note", "category"}, where category is one of:

    finding  what the last run concluded (rating, thesis, key numbers)
    lesson   what the last report lacked and how to fix it next time
    tool     a tool error and what worked instead

The date lives in the record only. load_notes() returns the note text
without dates so the planner's prompt (and its cache key) stays the
same when the same notes are replayed.
"""

import json
from datetime import datetime
from pathlib import Path

from finagent import config

# None means config.MEMORY_DIR (repo root). Tests and the notebook can
# point this, or the memory_dir argument, at another folder.
MEMORY_DIR: Path | None = None

CATEGORIES = ("finding", "lesson", "tool")
GENERAL = "general"


def _dir(memory_dir=None) -> Path:
    path = Path(memory_dir or MEMORY_DIR or config.MEMORY_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dedupe_notes(notes: list[str]) -> list[str]:
    """Keep the first occurrence of each note and discard duplicates."""
    seen: set[str] = set()
    ordered: list[str] = []
    for note in notes:
        normalized = str(note).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [r for r in json.load(f) if r.get("note")]


def load_records(ticker: str, memory_dir=None) -> list[dict]:
    """Ticker records then general ones, deduplicated by note text.

    Each record gets a "scope" of "ticker" or "general".
    """
    folder = _dir(memory_dir)
    records = [
        {**r, "scope": "ticker"}
        for r in _read(folder / f"{ticker.upper()}.json")
    ] + [
        {**r, "scope": "general"}
        for r in _read(folder / f"{GENERAL}.json")
    ]
    keep = set(_dedupe_notes([r["note"] for r in records]))
    out = []
    for r in records:
        note = str(r["note"]).strip()
        if note in keep:
            keep.discard(note)
            out.append({**r, "note": note,
                        "category": r.get("category", "finding")})
    return out


def load_notes(ticker: str, memory_dir=None) -> list[str]:
    """Load notes for this ticker + general lessons."""
    return [r["note"] for r in load_records(ticker, memory_dir)]


def _append(path: Path, notes, category: str) -> list[dict]:
    """Append notes to one file, skipping ones already stored. Returns
    the records actually added."""
    existing = _read(path)
    seen = set(_dedupe_notes([r["note"] for r in existing]))
    # Normalize stored records (older files may have duplicates).
    kept, done = [], set()
    for r in existing:
        note = str(r["note"]).strip()
        if note in done:
            continue
        done.add(note)
        kept.append({
            "date": r.get("date", datetime.now().isoformat()),
            "note": note,
            "category": r.get("category", "finding"),
        })

    added = []
    items = [n if isinstance(n, dict) else {"note": n} for n in notes]
    for item in items:
        note = str(item.get("note", "")).strip()
        if not note or note in seen:
            continue
        record = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "note": note,
            "category": item.get("category", category),
        }
        kept.append(record)
        added.append(record)
        seen.add(note)

    with open(path, "w") as f:
        json.dump(kept, f, indent=2)
    return added


def save_notes(ticker: str, notes: list, category: str = "finding",
               memory_dir=None) -> list[dict]:
    """Append notes from this run, preserving order and removing
    duplicates.

    notes are strings (stored under category) or dicts with "note" and
    optionally "category". Returns the records that were new.
    """
    path = _dir(memory_dir) / f"{ticker.upper()}.json"
    return _append(path, notes, category)


def save_general(notes: list, category: str = "tool",
                 memory_dir=None) -> list[dict]:
    """Like save_notes, for lessons that apply to every ticker."""
    return _append(_dir(memory_dir) / f"{GENERAL}.json", notes, category)
