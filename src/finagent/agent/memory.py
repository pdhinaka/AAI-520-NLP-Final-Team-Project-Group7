"""Notes that persist across runs in memory/."""

from pathlib import Path
import json
from datetime import datetime

MEMORY_DIR = Path("memory")

def load_notes(ticker: str) -> list[str]:
    """Load notes for this ticker + general lessons."""
    MEMORY_DIR.mkdir(exist_ok=True)
    
    notes = []
    
    # Load ticker-specific notes
    ticker_file = MEMORY_DIR / f"{ticker}.json"
    if ticker_file.exists():
        with open(ticker_file) as f:
            ticker_notes = json.load(f)
            notes.extend([n["note"] for n in ticker_notes])
    
    # Load general lessons
    general_file = MEMORY_DIR / "general.json"
    if general_file.exists():
        with open(general_file) as f:
            general_notes = json.load(f)
            notes.extend([n["note"] for n in general_notes])
    
    return notes


def save_notes(ticker: str, notes: list[str]):
    """Append notes from this run."""
    MEMORY_DIR.mkdir(exist_ok=True)
    
    ticker_file = MEMORY_DIR / f"{ticker}.json"
    
    # Load existing or start fresh
    existing = []
    if ticker_file.exists():
        with open(ticker_file) as f:
            existing = json.load(f)
    
    # Append new notes with timestamp
    for note in notes:
        existing.append({
            "date": datetime.now().isoformat(),
            "note": note,
            "category": "research"  # optional
        })
    
    # Write back
    with open(ticker_file, 'w') as f:
        json.dump(existing, f, indent=2)
