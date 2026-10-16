"""Notes that persist across runs in memory/."""

import json
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path("memory")


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


def load_notes(ticker: str) -> list[str]:
    """Load notes for this ticker + general lessons."""
    MEMORY_DIR.mkdir(exist_ok=True)

    notes = []

    # Load ticker-specific notes
    ticker_file = MEMORY_DIR / f"{ticker}.json"
    if ticker_file.exists():
        with open(ticker_file) as f:
            ticker_notes = json.load(f)
            notes.extend([n["note"] for n in ticker_notes if "note" in n])

    # Load general lessons
    general_file = MEMORY_DIR / "general.json"
    if general_file.exists():
        with open(general_file) as f:
            general_notes = json.load(f)
            notes.extend([n["note"] for n in general_notes if "note" in n])

    return _dedupe_notes(notes)


def save_notes(ticker: str, notes: list[str]):
    """Append notes from this run, preserving order and removing duplicates."""
    MEMORY_DIR.mkdir(exist_ok=True)

    ticker_file = MEMORY_DIR / f"{ticker}.json"

    existing = []
    if ticker_file.exists():
        with open(ticker_file) as f:
            existing = json.load(f)

    # Deduplicate and normalize existing notes first
    existing_notes = _dedupe_notes([
        str(item.get("note", "")) for item in existing if item.get("note")
    ])

    # Build seen set from normalized existing notes
    seen = set(existing_notes)

    # Re-create existing list with deduplicated notes
    existing = [
        {
            "date": existing[i].get("date", datetime.now().isoformat()),
            "note": note,
            "category": existing[i].get("category", "research"),
        }
        for i, note in enumerate(existing_notes)
    ]

    for note in _dedupe_notes(notes):
        if note in seen:
            continue
        existing.append({
            "date": datetime.now().isoformat(),
            "note": note,
            "category": "research",
        })
        seen.add(note)

    with open(ticker_file, 'w') as f:
        json.dump(existing, f, indent=2)
