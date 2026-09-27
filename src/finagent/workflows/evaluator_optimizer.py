"""Evaluator-optimizer: generate, evaluate, refine.

Owner: Workstream C."""



def generate(context: dict):
    """Draft an analysis."""
    raise NotImplementedError


def evaluate(analysis: str):
    """Score the analysis against a rubric and return feedback."""
    raise NotImplementedError


def refine(analysis: str, feedback: str):
    """Revise the analysis using the feedback."""
    raise NotImplementedError


def run_loop(context: dict, max_iters: int = 3, threshold: float = 0.8):
    """Loop until the score passes; return final text and score history."""
    raise NotImplementedError
