# Hosted agent (Groq gpt-oss-20b): how the tool use fails, and why (2026-11-17)

**Symptom.** On the live demo, a quick test on Banking77 (5 kinds, gap 0.20) came back `INVALID`:
the agent passed 95.7% of cases but failed the pre-registered quality floor ("own tool use 0.826,
send-backs 0.174", floor 0.90). A local run with the same settings reproduced it exactly.

**What was NOT wrong.** The tool connection. Sending the agent's real first-call request (its
system prompt and both tool definitions) for 8 sample messages under three `tool_choice` settings
(free, `required`, named `search_examples`) gave 24 of 24 valid tool calls, with correct arguments.

**Why the floor fails when the first call is free.** On about 1 case in 6 the model answers with the
label straight away instead of searching. The agent rule sends that back once ("use your tools
first"); the model then searches and answers correctly. Accuracy is fine; the floor counts the
send-back as "not its own tool use", so 0.826 < 0.90.

**Why forcing the first tool call (`tool_choice: required`, or naming the tool) made it worse.**
Measured locally on the same 39-message sample: free 95.7% (23 cases scored), required 68.8%,
named 75.0% (16 cases scored each, test stopped early). Captured replies for the failing cases:
Groq returned HTTP 400 `tool_use_failed` with the model's raw output being just the label
(for example `direct_debit_payment_not_recognised`). On a message it is sure of, the model still
writes the label as text; under a forced tool choice Groq tries to parse that as a tool call,
refuses it, and the agent scores the case malformed (locked rule: a malformed tool call scores
wrong). Free mode turns the same reply into a send-back, which works. So forcing converts a
harmless skipped tool into a hard failure on about 1 case in 4, and the floor still failed
(`own tool use 0.75`).

**Decision.** The forcing code stays as an off-by-default experiment (`GROQ_FIRST_TOOL=required|named`,
recorded in plan.json, part of the cache key). Nothing else in the agent was changed.

**Options not yet tried (need the owner's approval, they change the agent for the hosted model):**
1. Put the search results in the first message (examples in context) so no tool call is needed.
2. A hosted-only line in the system prompt: "never answer with a label before calling search_examples".
3. Keep the rule as is and explain INVALID plainly on the page.

**Monitoring.** A tracing tool (LangSmith or Langfuse) would show each call's request, reply, tool
calls and failures in one place; the ledger today keeps counts and tokens only.
