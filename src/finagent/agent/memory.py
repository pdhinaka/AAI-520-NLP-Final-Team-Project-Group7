"""Notes that persist across runs in memory/."""



def load_notes(ticker: str):
    """Return saved notes for this ticker plus general lessons."""
    raise NotImplementedError


def save_notes(ticker: str, notes: list[str]):
    """Append notes from this run."""
    raise NotImplementedError
