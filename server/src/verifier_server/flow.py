"""The "test your agent" flow: what the page may choose from, and how a quick test is launched.

Every value comes from the project's own code: the dataset menu (catalog), the labels of a built
DEV split, test sizes from the exact McNemar power code (quick.plan_size), measured seconds per
case, and the prereg's price tables. A quick test runs as a child process (`agent_committee.quick`
imports the runner, so this server never imports it) and only on DEV; it never opens TEST.

Nothing here builds a split: that stays a command for the owner until they decide otherwise.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# the runner's RUNNABLE arms (copied: the server never imports the runner; a test compares them)
ARMS = (
    "majority",
    "linear",
    "naive_bayes",
    "linear_svm",
    "random_forest",
    "boosting",
    "prompt",
    "agent",
)
LLM_ARMS = ("prompt", "agent")
AGENT = "agent"
GAPS = (0.02, 0.05, 0.10, 0.15, 0.20)
READY = ("built in", "split built")
QUICK_TIMEOUT_S = 3600.0


class FlowRefused(ValueError):
    pass


def _ready(entry: dict[str, Any]) -> bool:
    return entry.get("status") in READY


def flow_data(root: Path) -> dict[str, Any]:
    """The page's choices: datasets, labels of the ready ones, test sizes, timings, prices."""
    from agent_committee import catalog, quick
    from agent_committee.data import load_dev
    from agent_committee.prereg import load_prereg
    from agent_committee.scenarios import get_scenario

    prereg = load_prereg(root / "prereg.yaml")
    datasets, labels = [], {}
    for entry in catalog.menu(root):
        if entry["option"] == "own":
            continue
        datasets.append(
            {k: entry.get(k) for k in ("option", "title", "description", "status", "labels")}
        )
        if _ready(entry):
            s = get_scenario(entry["option"], root)
            labels[entry["option"]] = sorted({c.label for c in load_dev(root / s.data_dir)})
    # the reader's own datasets (added from the page or the MCP tools), once their split is built
    from agent_committee.scenarios import file_scenarios

    presets = {d["option"] for d in datasets}
    for name in file_scenarios(root):
        if name in presets:
            continue
        s = get_scenario(name, root)
        if not (root / s.data_dir / "dev.jsonl").is_file():
            continue
        from agent_committee.scenarios import read_scenario_file

        title = read_scenario_file(name, root).description or name
        datasets.append(
            {
                "option": name,
                "title": title,
                "description": "",
                "status": "split built",
                "own": True,
            }
        )
        labels[name] = sorted({c.label for c in load_dev(root / s.data_dir)})
    return {
        "datasets": datasets,
        "labels": labels,
        "plan_sizes": {f"{g:.2f}": quick.plan_size(g, prereg.alpha, prereg.power) for g in GAPS},
        "seconds_per_case": dict(quick.SECONDS_PER_CASE),
        "price_tables": [t.model_dump() for t in prereg.price_tables],
        "eval_sizes": {name: eval_size(root, name) for name in labels},
        "speeds": {name: speeds(root, name) for name in labels},
        "saved_run": None,
    }


def speeds(root: Path, scenario: str) -> dict[str, Any]:
    """Measured times for the page's estimate: whether the free models are saved for the current
    code, how long fitting them took, and our test agent's seconds per answer (None = unknown)."""
    from agent_committee.model_cache import read_summary

    summary = read_summary(root, scenario) or {}
    fits = summary.get("fit_seconds") or {}
    return {
        "models_saved": bool(fits),
        "fit_seconds": round(sum(fits.values()), 1) if fits else None,
        "agent_seconds_per_case": summary.get("agent_seconds_per_case"),
    }


def quick_command(choice: dict[str, Any], data: dict[str, Any]) -> list[str]:
    """The child command for a checked choice. Every value is checked against an allow-list and
    passed as its own argument (never through a shell)."""
    scenario = choice.get("scenario")
    if not isinstance(scenario, str) or scenario not in data["labels"]:
        raise FlowRefused(f"dataset {scenario!r} is not ready to test")
    gap = choice.get("gap")
    if not isinstance(gap, (int, float)) or round(float(gap), 2) not in GAPS:
        raise FlowRefused(f"gap must be one of {list(GAPS)}")
    arms = choice.get("arms")
    if not isinstance(arms, list) or not arms or any(a not in ARMS for a in arms):
        raise FlowRefused(f"arms must be among {list(ARMS)}")
    if AGENT not in arms or all(a in LLM_ARMS for a in arms):
        raise FlowRefused("include the agent and at least one free model")
    scope = choice.get("scope")
    known = data["labels"][scenario]
    if (
        not isinstance(scope, list)
        or not scope
        or any(not isinstance(x, str) or x not in known for x in scope)
    ):
        raise FlowRefused("scope must be labels of this dataset")
    cmd = [
        sys.executable, "-u", "-m", "agent_committee.quick",
        "--scenario", scenario,
        "--gap", f"{float(gap):.2f}",
        "--arms", ",".join(a for a in ARMS if a in arms),
        "--yes",
    ]  # fmt: skip
    if set(scope) != set(known):  # every label = the full test, no scoping
        cmd += ["--scope", ",".join(sorted(set(scope)))]
    one_route = choice.get("one_route", False)
    if not isinstance(one_route, bool):
        raise FlowRefused("one_route must be true or false")
    if one_route:
        if len(set(scope)) < 2 or set(scope) == set(known):
            raise FlowRefused("“same place” needs two or more of your kinds, not all of them")
        cmd += ["--one-route"]
    agent = choice.get("agent")
    if agent is not None:  # the user's own agent over HTTP (agent_committee.external)
        url, cost = agent_choice(agent)
        if "prompt" in arms:
            raise FlowRefused(
                "the single-prompt model runs on Ollama; leave it out with your agent"
            )
        cmd += ["--agent-url", url, "--agent-cost", repr(cost)]
    return cmd


AGENT_MAX_COST = 10.0  # USD per message: anything above is a typo, not a price


def agent_choice(agent: Any) -> tuple[str, float]:
    """A checked {url, usd_per_message} for the user's agent."""
    from agent_committee.external import check_url

    if not isinstance(agent, dict):
        raise FlowRefused("agent must be {url, usd_per_message}")
    try:
        url = check_url(str(agent.get("url", "")))
    except ValueError as exc:
        raise FlowRefused(str(exc)) from exc
    cost = agent.get("usd_per_message")
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not 0 <= cost <= AGENT_MAX_COST
    ):
        raise FlowRefused(
            f"your agent's price per message must be between $0 and ${AGENT_MAX_COST:g}"
        )
    return url, float(cost)


def unreachable(exc: BaseException) -> str:
    """Why a call to the user's agent failed, in plain words."""
    import socket
    import urllib.error

    reason = getattr(exc, "reason", exc)
    if isinstance(exc, urllib.error.HTTPError):
        return f"Your agent answered with an error (HTTP {exc.code}). Check its address and path."
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "Your agent didn’t answer within 15 seconds."
    text = str(reason).lower()
    if isinstance(reason, ConnectionRefusedError) or "refused" in text or "10061" in text:
        return "Nothing is answering at that address. Is your agent running, and is the port right?"
    if isinstance(reason, socket.gaierror):
        return "That host name can’t be found. Check the address."
    if isinstance(exc, ValueError):
        return 'Your agent answered, but not with JSON like {"label": "..."}.'
    return "Couldn’t reach your agent at that address."


def probe_agent(root: Path, scenario: str, url: str, n: int = 3) -> dict[str, Any]:
    """Send a few practice messages (DEV) to the user's agent and show what came back, beside the
    labels it may answer. Nothing is scored; this only checks the connection and the format."""
    from agent_committee.external import NOT_MINE, ask, check_url

    try:
        url = check_url(url)
    except ValueError as exc:
        raise FlowRefused(str(exc)) from exc
    from agent_committee.arms.splits import make_arms_split
    from agent_committee.data import load_dev
    from agent_committee.scenarios import get_scenario

    # training-pool messages only: a test message never reaches the agent before its test
    pool = make_arms_split(load_dev(root / get_scenario(scenario, root).data_dir)).pool
    labels = sorted({c.label for c in pool}, key=str.lower)
    valid = set(labels) | {NOT_MINE}
    ranked = sorted(pool, key=lambda c: c.id)
    picks = [{"text": c.text} for c in ranked[:: max(1, len(ranked) // n)][:n]]
    tries = []
    for m in picks:
        try:
            answer, took = ask(url, m["text"], labels, timeout=15)
            tries.append(
                {
                    "text": m["text"],
                    "answer": answer,
                    "valid": answer in valid,
                    "seconds": round(took, 2),
                }
            )
        except Exception as exc:  # noqa: BLE001 (shown to the user in plain words)
            return {"ok": False, "error": unreachable(exc), "tries": tries, "labels": labels}
    ok = all(t["valid"] for t in tries)
    return {
        "ok": ok,
        "error": None if ok else "some answers are not one of this dataset's kinds (or not_mine)",
        "tries": tries,
        "labels": labels,
    }


_LOOK = re.compile(
    r"look at (\d+) cases: gain ([-+]?\d*\.\d+), z\(agent ahead\) ([-\d.]+), "
    r"z\(gain ruled out\) ([-\d.]+), boundary ([\d.]+)"
)
_PLAN = re.compile(r"^gap ([\d.]+): (\d+) paired cases planned, (\d+) drawn")
_SIMPLER = re.compile(r"^simpler model to beat \(best free model on tune\): (\w+)")
_STOP = re.compile(r"^stopped at (\d+) of (\d+) cases: (.+)$")
_VERDICT = re.compile(r"^verdict: (\w+) -> (.+)$")
_TRAINING = re.compile(r"^training (\w+) \.\.\.$")
_TRAINED = re.compile(r"^trained (\w+) in (\d+) s$")
_SCORED = re.compile(r"^scored (\d+) of (\d+) cases$")


def read_progress(progress: dict[str, Any], line: str) -> None:
    """Update a quick test's progress from one line of its output (plain parsing, no guessing)."""
    s = line.strip()
    if m := _LOOK.search(s):
        progress.setdefault("looks", []).append(
            {
                "n": int(m[1]),
                "gain": float(m[2]),
                "z_agent": float(m[3]),
                "z_ruled_out": float(m[4]),
                "bound": float(m[5]),
            }
        )
        progress["phase"] = "racing"
    elif m := _SCORED.match(s):
        progress.update(scored=int(m[1]), phase="racing")
    elif m := _TRAINING.match(s):
        progress.update(training=m[1], phase="training")
    elif m := _TRAINED.match(s):
        progress.setdefault("trained", []).append(m[1])
        progress.pop("training", None)
    elif m := _PLAN.match(s):
        progress.update(planned=int(m[2]), drawn=int(m[3]), phase="preparing")
    elif m := _SIMPLER.match(s):
        progress.update(simpler=m[1], phase="racing")
    elif m := _STOP.match(s):
        progress.update(stopped_at=int(m[1]), of=int(m[2]), phase="finishing")
    elif m := _VERDICT.match(s):
        progress.update(verdict=m[1], result=m[2].strip(), phase="done")


def read_result(root: Path, verdict_path: str) -> dict[str, Any]:
    """plan.json, scope.json (when scoped) and verdict.json of a finished quick test. The folder
    must be inside results/ (the child's own output names it)."""
    results = (root / "results").resolve()
    path = Path(verdict_path)
    folder = (path if path.is_absolute() else root / path).resolve().parent
    if results not in folder.parents or folder.name == "":
        raise FlowRefused("result folder is outside results/")
    out: dict[str, Any] = {}
    for name in ("plan", "scope", "verdict"):
        f = folder / f"{name}.json"
        out[name] = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
    if out["plan"] is None or out["verdict"] is None:
        raise FlowRefused("the quick test left no plan or verdict")
    out["messages"] = run_messages(root, folder, out["plan"])
    return out


LIBRARY_PER_LABEL = 6


def library(root: Path, scenario: str, per_label: int = LIBRARY_PER_LABEL) -> list[dict[str, str]]:
    """A fixed sample of DEV messages per label (the page's message graph and its label
    suggestions). DEV only, chosen by a stable hash of the case id, so it is the same every time."""
    import hashlib

    from agent_committee.data import load_dev
    from agent_committee.scenarios import get_scenario

    cases = load_dev(root / get_scenario(scenario, root).data_dir)
    key = lambda c: hashlib.sha256(c.id.encode("utf-8")).hexdigest()  # noqa: E731
    out: list[dict[str, str]] = []
    by_label: dict[str, list[Any]] = {}
    for c in sorted(cases, key=key):
        bucket = by_label.setdefault(c.label, [])
        if len(bucket) < per_label:
            bucket.append(c)
    for label in sorted(by_label):
        out += [{"text": c.text, "label": c.label} for c in by_label[label]]
    return out


def neighbours(texts: list[str], k: int = 3) -> list[list[int]]:
    """Each message's k most similar messages (character TF-IDF cosine; plain code)."""
    if len(texts) < 2:
        return [[] for _ in texts]
    from sklearn.feature_extraction.text import TfidfVectorizer

    m = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts)
    sim = (m @ m.T).toarray()
    out = []
    for i in range(len(texts)):
        sim[i, i] = -1.0
        out.append([int(j) for j in sim[i].argsort()[::-1][: min(k, len(texts) - 1)]])
    return out


def run_messages(root: Path, folder: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-message results of a finished quick test, in the order the agent scored them: the text,
    its true kind, the scoped right answer, both answers (agent and the simpler model) and
    whether each was right, and its 3 most similar messages. Read from the run's own files."""
    from agent_committee.data import load_dev
    from agent_committee.scenarios import get_scenario

    simpler = plan.get("comparator")
    scope = plan.get("scope") or {}
    labels = set(scope.get("labels") or [])
    near = set(scope.get("near_misses") or [])
    mine = scope.get("not_mine_label", "not_mine")
    route = scope.get("route_label") if scope.get("one_route") else None
    pred_file = folder / "predictions.jsonl"
    if not pred_file.is_file() or not plan.get("scenario"):
        return []
    cases = {c.id: c for c in load_dev(root / get_scenario(plan["scenario"], root).data_dir)}
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in pred_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        if p["arm"] not in (AGENT, simpler) or p["case_id"] not in cases:
            continue
        if p["arm"] == AGENT:
            order.append(p["case_id"])
        rows.setdefault(p["case_id"], {})[p["arm"]] = p
    out = []
    for cid in order:
        r = rows[cid]
        a, s = r.get(AGENT), r.get(simpler)
        if a is None or s is None:
            continue
        true = cases[cid].label
        out.append(
            {
                "text": cases[cid].text,
                "true": true,
                "gold": (route or true) if not labels or true in labels else mine,
                "near": true in near,
                "agent": a.get("pred"),
                "agent_ok": bool(a.get("correct")),
                "simpler": s.get("pred"),
                "simpler_ok": bool(s.get("correct")),
                "agent_calls": int(a.get("own_tool_calls") or 0),
                "agent_latency": float(a.get("latency_s") or 0.0),
                "agent_tokens": int(a.get("prompt_tokens") or 0)
                + int(a.get("completion_tokens") or 0),
            }
        )
    for m, nb in zip(out, neighbours([m["text"] for m in out]), strict=True):
        m["nb"] = nb
    return out


def eval_size(root: Path, scenario: str) -> int:
    """How many DEV cases a quick test can score at most (the arms' eval split)."""
    from agent_committee.arms.splits import make_arms_split
    from agent_committee.data import load_dev
    from agent_committee.scenarios import get_scenario

    return len(make_arms_split(load_dev(root / get_scenario(scenario, root).data_dir)).eval)


def detectable(n_eval: int, sizes: dict[str, int]) -> float | None:
    """The smallest planned gap whose test fits in n_eval scored cases (None: none fits)."""
    fits = [float(g) for g, n in sizes.items() if n <= n_eval]
    return min(fits) if fits else None


OWN_MAX_BYTES = 20 * 1024 * 1024
OWN_DIR = "sources/own"


def _own_rows(raw: bytes, fmt: str) -> list[dict[str, str]]:
    """Rows of an uploaded .csv or .jsonl with text and label columns (id optional)."""
    import csv
    import io

    text = raw.decode("utf-8-sig")
    if fmt == "jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif fmt == "csv":
        rows = list(csv.DictReader(io.StringIO(text)))
    else:
        raise FlowRefused("upload a .csv or .jsonl file")
    out = []
    for n, row in enumerate(rows, 1):
        ok = isinstance(row, dict) and all(
            isinstance(row.get(k), str) and row[k].strip() for k in ("text", "label")
        )
        if not ok:
            raise FlowRefused(f"row {n}: needs text and label columns, both filled")
        out.append(row)
    if not out:
        raise FlowRefused("the file has no rows")
    return out


def own_cases(raw: bytes, fmt: str) -> list[Any]:
    """Cases from an upload. Without an id column, ids come from each row's own text and label
    (plus a counter for repeats), so the split never depends on the file's row order."""
    import hashlib

    from agent_committee.schemas import Case

    seen: dict[str, int] = {}
    cases = []
    for row in _own_rows(raw, fmt):
        rid = row.get("id")
        if not (isinstance(rid, str) and rid.strip()):
            key = f"{row['text']}\x1f{row['label']}".encode()
            base = hashlib.sha256(key).hexdigest()[:16]
            seen[base] = seen.get(base, 0) + 1
            rid = f"{base}-{seen[base]}"
        tie = row.get("group") or None
        group = tie if isinstance(tie, str) else None
        cases.append(Case(id=rid, text=row["text"], label=row["label"].strip(), group=group))
    if len({c.id for c in cases}) != len(cases):
        raise FlowRefused("ids must be unique")
    return cases


def add_own(
    root: Path, name: str, fmt: str, raw: bytes, phrase: dict[str, str], title: str = ""
) -> dict[str, Any]:
    """The reader's own data, end to end: check it in memory with the same split checks a build
    runs, and only then write sources/own/<name>.jsonl, its scenario, its prereg (Banking77's
    assumed costs, marked as copied) and its one-time DEV/TEST split. Refuses to overwrite."""
    from agent_committee import catalog, quick
    from agent_committee.data import build_split, planned_split
    from agent_committee.prereg import load_prereg
    from agent_committee.scenarios import (
        LocalSource,
        ScenarioFile,
        TaskPhraseSpec,
        add_scenario,
        check_name,
        definition_path,
        get_scenario,
    )

    try:
        check_name(name)
    except ValueError as exc:
        raise FlowRefused(str(exc)) from exc
    taken = (
        name in catalog.PRESETS
        or (root / definition_path(name)).exists()
        or (root / OWN_DIR / f"{name}.jsonl").exists()
    )
    if taken:
        raise FlowRefused(f"a dataset called {name!r} already exists; pick another name")
    if len(raw) > OWN_MAX_BYTES:
        raise FlowRefused("the file is larger than 20 MB")
    if not all(
        isinstance(phrase.get(k), str) and phrase[k].strip()
        for k in ("item", "items", "label_kind")
    ):
        raise FlowRefused("say what one message is, what many are called, and what a label is")
    cases = own_cases(raw, fmt)
    prereg = load_prereg(root / "prereg.yaml")
    try:
        planned_split(cases, prereg.split)  # every check a build runs; writes nothing
    except ValueError as exc:
        raise FlowRefused(f"this data can't be split fairly yet: {exc}") from exc

    src = root / OWN_DIR / f"{name}.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for c in cases:
        row = {"id": c.id, "text": c.text, "label": c.label}
        if c.group:
            row["group"] = c.group
        lines.append(json.dumps(row) + "\n")
    src.write_text("".join(lines), encoding="utf-8")
    spec = ScenarioFile(
        name=name,
        description=title.strip()[:80] or name,
        source=LocalSource(kind="local", path=f"{OWN_DIR}/{name}.jsonl"),
        task_phrase=TaskPhraseSpec(
            item=phrase["item"].strip(),
            items=phrase["items"].strip(),
            label_kind=phrase["label_kind"].strip(),
        ),
    )
    add_scenario(spec, root)
    body = (root / "prereg.yaml").read_text(encoding="utf-8")
    body = re.sub(r"^task: \w+", f"task: {name}", body, count=1, flags=re.M)
    head = (
        f"# Pre-registration for your own dataset {name}: COPIED from Banking77 "
        "(every cost, price and gap is its ASSUMPTION).\n"
    )
    (root / "scenarios" / name / "prereg.yaml").write_text(head + body, encoding="utf-8")
    manifest = build_split(get_scenario(name, root), root)
    n_eval = eval_size(root, name)
    sizes = {f"{g:.2f}": quick.plan_size(g, prereg.alpha, prereg.power) for g in GAPS}
    return {
        "name": name,
        "cases": len(cases),
        "labels": len({c.label for c in cases}),
        "dev": manifest.get("n_dev"),
        "eval": n_eval,
        "detectable_gap": detectable(n_eval, sizes),
    }
