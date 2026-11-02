"""Shared constants for the arms. Values measured by the C1 probe (see COORDINATION.md)."""

TFIDF_SEED = 0
TEMPERATURE = 0.0
MODEL = "qwen2.5:7b"
MODEL_DIGEST = "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"

# Ollama silently truncates the head of any prompt over this many tokens and still reports a
# smaller prompt_tokens, so truncation cannot be detected afterwards. Reject before sending.
CONTEXT_CAP_TOKENS = 4096
MAX_PROMPT_CHARS = 14_000  # about 3,250 tokens at 4.3 chars/token; sum all messages per call

K_SHOTS = 2  # few-shot examples per label in the single-prompt arm
RETRIEVAL_K = 5  # neighbours returned by the agent's retrieve tool
AGENT_MAX_TOOL_TURNS = 3  # at most 4 model calls per case
