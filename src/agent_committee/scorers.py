"""Plain-code scorers named by prereg.yaml. No LLM may sit behind one."""


def exact_label(pred: str | None, gold: str) -> bool:
    """Right only on an exact label match. No answer (None) is wrong."""
    return pred == gold
