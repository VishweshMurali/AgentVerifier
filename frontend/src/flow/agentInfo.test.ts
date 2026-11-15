import assert from 'node:assert/strict';
import { test } from 'node:test';
import { agentWords, maskedNote, OLLAMA_AGENT, runAgent } from './agentInfo.ts';

test('the local agent is named as Ollama on this computer', () => {
  for (const info of [undefined, null, OLLAMA_AGENT]) {
    const w = agentWords(info);
    assert.equal(w.name, 'Local Ollama agent');
    assert.equal(w.hosted, false);
    assert.match(w.who, /this computer/);
  }
});

test('a hosted agent is never called local and says a re-run may differ', () => {
  const w = agentWords({ provider: 'groq', name: 'Hosted test agent', model: 'openai/gpt-oss-20b via Groq (hosted, not pinned)', pinned: false });
  assert.equal(w.name, 'Hosted test agent');
  assert.equal(w.hosted, true);
  for (const text of [w.name, w.lower, w.who, w.standIn, w.tag]) assert.doesNotMatch(text, /Ollama|this computer/);
  assert.match(w.who, /not pinned/);
  assert.match(w.who, /re-run may differ/);
});

test('a finished run is named by its own plan', () => {
  assert.equal(runAgent({ kind: 'reference', model: 'llama3.1:8b via Ollama' }), OLLAMA_AGENT);
  assert.equal(runAgent(undefined), OLLAMA_AGENT);
  const hosted = runAgent({ kind: 'hosted', model: 'm via Groq (hosted, not pinned)' });
  assert.equal(hosted.pinned, false);
  assert.equal(agentWords(hosted).tag, 'm via Groq (hosted, not pinned)');
});

test('the upload note says what was masked and what cannot be', () => {
  assert.match(maskedNote({ kinds: { EMAIL: 3, PHONE: 1 }, messages: 3 }), /^Masked 4 personal details \(3 email addresses, 1 phone number\) in 3 messages before saving anything\. Names/);
  assert.match(maskedNote({ kinds: { CARD: 1 }, messages: 1 }), /^Masked 1 personal detail \(1 card number\) in 1 message /);
  assert.match(maskedNote({ kinds: {}, messages: 0 }), /^No personal details .* were found to mask\. Names/);
  assert.match(maskedNote(undefined), /^No personal details/);
});
