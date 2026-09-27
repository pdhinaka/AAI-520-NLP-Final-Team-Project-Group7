"""Single entry point for LLM calls. Backend is a team decision."""



def complete(prompt: str, system: str | None = None, **kwargs) -> str:
    """Send a prompt to the chosen LLM and return the text reply."""
    raise NotImplementedError("Pick an LLM backend (see README)")
