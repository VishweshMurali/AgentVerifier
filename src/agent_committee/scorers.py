"""Plain-code scorers named by prereg.yaml. No LLM may sit behind one.

A scenario names its scorer in its prereg (`scorer.target`, e.g.
`agent_committee.scorers:exact_label`).
Only functions registered in SCORERS can be resolved, so a prereg cannot point at arbitrary code.
"""

from collections.abc import Callable

SCORER_MODULE = "agent_committee.scorers"
Scorer = Callable[[str | None, str], bool]


def exact_label(pred: str | None, gold: str) -> bool:
    """Right only on an exact label match. No answer (None) is wrong."""
    return pred == gold


SCORERS: dict[str, Scorer] = {"exact_label": exact_label}


def get_scorer(name: str) -> Scorer:
    """A registered scorer by name; anything else is refused."""
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; registered: {sorted(SCORERS)}")
    return SCORERS[name]


def resolve_scorer(target: str) -> Scorer:
    """The prereg's scorer from its `module:function` target. Only registered plain-code scorers
    in this module are accepted."""
    module, _, func = target.partition(":")
    if module != SCORER_MODULE:
        raise ValueError(f"scorer {target!r} is not in {SCORER_MODULE}")
    return get_scorer(func)
