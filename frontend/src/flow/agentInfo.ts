// How the page names the agent that races when the reader brings none: the local Ollama
// reference (pinned), or on a public deployment a hosted model that is not pinned. Plain code.

export interface AgentInfo {
  provider: string;
  name: string;
  model: string;
  pinned: boolean;
  /** the hosted agent is told to search before it answers (its plan records it) */
  search_first?: boolean;
}

export const OLLAMA_AGENT: AgentInfo = { provider: 'ollama', name: 'Local Ollama agent', model: 'llama3.1:8b via Ollama', pinned: true };

export interface AgentWords {
  /** "Local Ollama agent" */
  name: string;
  /** "the local Ollama agent" */
  lower: string;
  /** one line saying exactly which model it is and where it runs */
  who: string;
  /** "a local Ollama model (llama3.1:8b on this computer)" */
  standIn: string;
  /** the short model tag beside the name */
  tag: string;
  hosted: boolean;
}

export function agentWords(info: AgentInfo | null | undefined): AgentWords {
  if (info && info.provider !== 'ollama') {
    return {
      name: 'Hosted test agent',
      lower: 'the hosted test agent',
      who: `${info.model}${info.search_first ? '; told to search before it answers' : ''}; the provider can change the model, so a re-run may differ`,
      standIn: `a hosted model, ${info.model}${info.search_first ? ' (told to search before it answers)' : ''}`,
      tag: info.model,
      hosted: true,
    };
  }
  return {
    name: 'Local Ollama agent',
    lower: 'the local Ollama agent',
    who: 'llama3.1:8b, running on this computer through Ollama',
    standIn: 'a local Ollama model (llama3.1:8b on this computer)',
    tag: 'llama3.1:8b via Ollama',
    hosted: false,
  };
}

/** The agent a finished run raced, read from its plan (never from the page's current settings). */
export function runAgent(agent?: { kind?: string; model?: string; search_first?: string } | null): AgentInfo {
  if (agent?.kind === 'hosted') {
    const told = typeof agent.search_first === 'string' && agent.search_first !== 'off';
    return { provider: 'hosted', name: 'Hosted test agent', model: agent.model ?? 'a hosted model (not pinned)', pinned: false, search_first: told };
  }
  return OLLAMA_AGENT;
}

/** "Masked 4 personal details (3 emails, 1 phone number) in 3 messages." or '' when none. */
const KIND_WORDS: Record<string, [string, string]> = {
  EMAIL: ['email address', 'email addresses'],
  CARD: ['card number', 'card numbers'],
  IBAN: ['IBAN', 'IBANs'],
  ID_NUMBER: ['ID number', 'ID numbers'],
  IP_ADDRESS: ['IP address', 'IP addresses'],
  PHONE: ['phone number', 'phone numbers'],
  ADDRESS: ['street address', 'street addresses'],
  LONG_NUMBER: ['long number', 'long numbers'],
};

export function maskedNote(masked?: { kinds?: Record<string, number>; messages?: number } | null): string {
  const kinds = Object.entries(masked?.kinds ?? {}).filter(([, n]) => n > 0);
  const total = kinds.reduce((a, [, n]) => a + n, 0);
  const caveat = 'Names and details written in words can’t be detected, so check your file for those.';
  if (total === 0) return `No personal details (emails, phone, card or ID numbers, addresses) were found to mask. ${caveat}`;
  const parts = kinds.map(([k, n]) => `${n} ${(KIND_WORDS[k] ?? [k, k])[n === 1 ? 0 : 1]}`);
  const messages = masked?.messages ?? 0;
  return `Masked ${total} personal ${total === 1 ? 'detail' : 'details'} (${parts.join(', ')}) in ${messages} ${messages === 1 ? 'message' : 'messages'} before saving anything. ${caveat}`;
}
