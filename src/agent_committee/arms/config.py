"""Shared constants for the arms. Values measured by the C1 probe (see COORDINATION.md)."""

TEMPERATURE = 0.0
MODEL = "llama3.1:8b"
MODEL_DIGEST = "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"

# Ollama silently truncates the head of any prompt over this many tokens and still reports a
# smaller prompt_tokens, so truncation cannot be detected afterwards. Reject before sending.
CONTEXT_CAP_TOKENS = 4096
MAX_PROMPT_CHARS = 14_000  # about 3,250 tokens at 4.3 chars/token; sum all messages per call

K_SHOTS = 2  # few-shot examples per label in the single-prompt arm
RETRIEVAL_K = 5  # neighbours returned by the agent's retrieve tool
AGENT_MAX_TOOL_TURNS = 3  # at most 4 model calls per case

SHORTLIST_K = 5  # labels the shared linear layer passes to the prompt and agent arms
# 10th percentile of the layer's top score on the 488 tune cases, computed once. A flag only.
NULL_THRESHOLD = 0.486288
