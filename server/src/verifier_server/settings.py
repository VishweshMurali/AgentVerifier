"""How this server is deployed, read once from the environment.

Local (the default): one operator on 127.0.0.1, the Ollama reference agent, every model.

Public (VERIFIER_PUBLIC=1, owner approved 2026-11-14): strangers on a small shared host, so
- the agent is the hosted one (AGENT_PROVIDER=groq, see agent_committee.hosted), not pinned;
- only the light free models run (the tree models need gigabytes);
- only the larger gaps are offered and a test must fit in what is left of the day's free
  tokens (the hosted agent's limit is shared by every visitor);
- rehearsals and "bring your agent" are off (the server would call any address a visitor names);
- uploads stay on, smaller, and each is listed only for the visitor who added it;
- the guide answers in plain code only (no model call spends the shared tokens).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from agent_committee import hosted

LOCAL_HOSTS = ("127.0.0.1", "localhost")
PUBLIC_ARMS = ("majority", "linear", "naive_bayes", "linear_svm", "agent")
PUBLIC_GAPS = (0.10, 0.20)
DAILY_TOKENS = 180_000  # under Groq's free 200K a day
TOKENS_PER_CASE = 900  # the hosted agent, measured on Banking77 (about 860)
PUBLIC_OWN_MAX_BYTES = 1024 * 1024
LOCAL_OWN_MAX_BYTES = 20 * 1024 * 1024
PUBLIC_QUEUE = 3  # tests allowed to wait in line behind the running one (public site)
UPLOAD_TTL_S = 2 * 3600.0  # a public upload is deleted this long after it was added


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    public: bool = False
    hosts: tuple[str, ...] = LOCAL_HOSTS  # Host names answered; ("*",) is any
    provider: str | None = None  # None = local Ollama; "groq" = hosted
    agent_name: str = "Local Ollama agent"
    agent_model: str = "llama3.1:8b via Ollama"
    arms: tuple[str, ...] | None = None  # None = every arm
    gaps: tuple[float, ...] | None = None  # None = every gap
    daily_tokens: int | None = None  # None = no cap
    own_max_bytes: int = LOCAL_OWN_MAX_BYTES
    workers: int | None = None  # self-check processes for a quick test (None = by CPU count)
    queue: int = 0  # tests allowed to wait behind the running one (0 = others are refused)
    upload_ttl_s: float | None = None  # public: how long an upload is kept (None = kept)

    def view(self) -> dict[str, object]:
        """What the page is told."""
        return {
            "public": self.public,
            "agent": {
                "provider": self.provider or "ollama",
                "name": self.agent_name,
                "model": self.agent_model,
                "pinned": self.provider is None,
            },
            "own_max_bytes": self.own_max_bytes,
        }


def load(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    public = _flag(env.get("VERIFIER_PUBLIC"))
    provider = (env.get("AGENT_PROVIDER") or "").strip().lower() or None
    if provider == "ollama":
        provider = None
    if provider not in (None, "groq"):
        raise ValueError(f"AGENT_PROVIDER must be 'groq' or 'ollama', not {provider!r}")
    if public and provider is None:
        raise ValueError("VERIFIER_PUBLIC needs a hosted agent: set AGENT_PROVIDER=groq")
    if provider and hosted.proxy(env) is None and not (env.get("GROQ_API_KEY") or "").strip():
        raise ValueError("AGENT_PROVIDER=groq needs GROQ_API_KEY (or GROQ_PROXY_URL + token)")
    named = [h.strip() for h in (env.get("VERIFIER_ALLOWED_HOSTS") or "").split(",") if h.strip()]
    if named:
        hosts = tuple(dict.fromkeys([*LOCAL_HOSTS, *named]))
    else:
        # a public host has nothing local to protect from DNS rebinding, and its proxy and
        # health checks pick the Host name: any name is answered unless names are listed
        hosts = ("*",) if public else LOCAL_HOSTS
    names = {}
    if provider:
        model = (env.get("GROQ_MODEL") or "").strip() or "openai/gpt-oss-20b"
        names = {
            "agent_name": "Hosted test agent",
            "agent_model": f"{model} via Groq (hosted, not pinned)",
        }
    workers = env.get("FALSE_ALARM_WORKERS")
    return Settings(
        public=public,
        hosts=hosts,
        provider=provider,
        arms=PUBLIC_ARMS if public else None,
        gaps=PUBLIC_GAPS if public else None,
        daily_tokens=int(env.get("VERIFIER_DAILY_TOKENS") or DAILY_TOKENS) if public else None,
        own_max_bytes=PUBLIC_OWN_MAX_BYTES if public else LOCAL_OWN_MAX_BYTES,
        workers=int(workers) if workers else None,
        queue=int(env.get("VERIFIER_QUEUE") or PUBLIC_QUEUE) if public else 0,
        upload_ttl_s=float(env.get("VERIFIER_UPLOAD_TTL_S") or UPLOAD_TTL_S) if public else None,
        **names,
    )
