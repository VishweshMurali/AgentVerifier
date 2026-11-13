// Screen 1's use-case picker: the four ready datasets as cards (none preselected), the reader's own
// uploaded datasets, "Bring your own data" (upload, checked and split once by the server) and
// "Bring your agent" (an HTTP address, checked with three practice messages, plus its price per
// message). Without one, our test agent (llama3.1:8b on Ollama) races. Plain code only.
import { useState, type CSSProperties } from 'react';
import { Icon } from './Icons.tsx';
import { Reveal } from './ScrollText.tsx';

/** The reader's own agent: where it answers and what one answer costs them (their figure). */
export interface MyAgent {
  url: string;
  usd: number;
  /** seconds per answer seen in the connection check, for the time estimate */
  secs?: number;
}
export interface AgentCheck {
  ok: boolean;
  error: string | null;
  tries: { text: string; answer: unknown; valid: boolean; seconds: number }[];
  labels: string[];
}

export interface UseCase {
  option: string;
  title: string;
  line: string;
  labels: number;
  own?: boolean;
  /** largest gap test that fits this dataset, from the server (null: unknown) */
  evalSize: number | null;
}

const PRESET_LINES: Record<string, [string, string]> = {
  banking77: ['Bank support routing', 'Is it a lost card or a late transfer?'],
  clinc150: ['General assistant requests', 'Set an alarm, book a table, check my balance…'],
  emotion: ['Customer mood', 'Is this customer angry, happy or worried?'],
  massive_en: ['Voice assistant commands', 'Play music, turn off the lights…'],
};

// words that point at a use case even when no practice message uses them (plain code)
export const CASE_KEYWORDS: Record<string, string[]> = {
  banking77: ['bank', 'card', 'transfer', 'refund', 'payment', 'account', 'atm', 'balance', 'money', 'top', 'cash', 'pin', 'loan'],
  clinc150: ['assistant', 'alarm', 'calendar', 'reminder', 'book', 'restaurant', 'travel', 'flight', 'weather', 'timer', 'translate', 'recipe'],
  emotion: ['customer', 'mood', 'feel', 'emotion', 'tone', 'anger', 'joy', 'sadness', 'fear', 'love', 'surprise', 'complaint', 'sentiment'],
  massive_en: ['voice', 'speaker', 'alexa', 'music', 'play', 'lights', 'iot', 'command', 'smart', 'home', 'podcast', 'radio'],
};

export function caseCard(option: string, title: string, labels: number, evalSize: number | null, own = false): UseCase {
  const p = PRESET_LINES[option];
  return { option, title: p ? p[0] : title, line: p ? p[1] : 'Your own data', labels, own, evalSize };
}

export interface OwnUpload {
  name: string;
  title: string;
  file: File;
  item: string;
  items: string;
  labelKind: string;
}

export function UseCases({
  cases,
  chosen,
  best,
  onChoose,
  canUpload,
  onUpload,
  uploadState,
  agent = null,
  setAgent,
  checkAgent,
}: {
  cases: UseCase[];
  chosen: string | null;
  best: string | null;
  onChoose: (option: string) => void;
  canUpload: boolean;
  onUpload: (u: OwnUpload) => void;
  uploadState: { busy: boolean; error: string | null; note: string | null };
  agent?: MyAgent | null;
  setAgent?: (a: MyAgent | null) => void;
  checkAgent?: (url: string) => Promise<AgentCheck>;
}) {
  const [url, setUrl] = useState(agent?.url ?? '');
  const [usd, setUsd] = useState(agent ? String(agent.usd) : '');
  const [probe, setProbe] = useState<{ busy: boolean; out: AgentCheck | null; error: string | null; url: string }>({ busy: false, out: null, error: null, url: '' });
  const price = Number(usd);
  const priceOk = usd.trim() !== '' && Number.isFinite(price) && price >= 0 && price <= 10;
  const checkedOk = !!probe.out?.ok && probe.url === url.trim();
  const runCheck = async () => {
    if (!checkAgent || !url.trim()) return;
    setProbe({ busy: true, out: null, error: null, url: url.trim() });
    try {
      setProbe({ busy: false, out: await checkAgent(url.trim()), error: null, url: url.trim() });
    } catch (e) {
      setProbe({ busy: false, out: null, error: e instanceof Error ? e.message : String(e), url: url.trim() });
    }
  };
  const [open, setOpen] = useState<'own' | 'agent' | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [item, setItem] = useState('a customer message');
  const [items, setItems] = useState('customer messages');
  const [labelKind, setLabelKind] = useState('topic');
  const name = (title || file?.name.replace(/\.(csv|jsonl)$/i, '') || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 40);

  return (
    <div className="cases">
      <p className="cases-q mono">Which use case do you want to test?</p>
      <Reveal className="case-grid" items=".case">
        {cases.map((c, i) => (
          <button
            key={c.option}
            type="button"
            style={{ '--i': i } as CSSProperties}
            className={`case glass${chosen === c.option ? ' on' : ''}${best === c.option && chosen !== c.option ? ' best' : ''}`}
            onClick={() => onChoose(c.option)}
            aria-pressed={chosen === c.option}
          >
            {best === c.option && <span className="case-badge mono">best match</span>}
            <Icon name={PRESET_LINES[c.option] ? c.option : 'agent'} />
            <b>{c.title}</b>
            <small>{c.line}</small>
            <span className="case-meta mono">
              {c.labels} kinds{c.own ? ' · yours' : ''}
            </span>
          </button>
        ))}
        <button type="button" className={`case glass extra${open === 'own' ? ' on' : ''}`} onClick={() => setOpen(open === 'own' ? null : 'own')}>
          <span className="case-plus">+</span>
          <b>Bring your own data</b>
          <small>A .csv or .jsonl with text and label columns</small>
        </button>
        <button type="button" className={`case glass extra${open === 'agent' ? ' on' : ''}`} onClick={() => setOpen(open === 'agent' ? null : 'agent')}>
          <span className="case-plus">⇄</span>
          <b>Bring your agent</b>
          <small>{agent ? 'Your agent is plugged in' : 'Test your own agent instead of ours'}</small>
          <span className="case-meta mono">{agent ? 'connected' : 'local Ollama by default'}</span>
        </button>
      </Reveal>

      {open === 'own' && (
        <form
          className="own glass"
          onSubmit={(e) => {
            e.preventDefault();
            if (file && name) onUpload({ name, title, file, item, items, labelKind });
          }}
        >
          <label className="own-file">
            <input type="file" accept=".csv,.jsonl" onChange={(e) => setFile(e.target.files?.[0] ?? null)} disabled={!canUpload || uploadState.busy} />
            <span>{file ? file.name : 'Choose a .csv or .jsonl file'}</span>
          </label>
          <small className="mono muted">
            Columns: <b>text</b> and <b>label</b> (optional <b>id</b>, <b>group</b>). At least 30 rows per label. Up to 20 MB.
          </small>
          <label className="own-row">
            <span>Name it</span>
            <input value={title} placeholder="e.g. My bank bot" onChange={(e) => setTitle(e.target.value)} />
          </label>
          <div className="own-phrase">
            <span className="mono muted">So the AI agent knows the task: it classifies</span>
            <input value={item} onChange={(e) => setItem(e.target.value)} aria-label="One message is" />
            <span className="mono muted">into one</span>
            <input value={labelKind} onChange={(e) => setLabelKind(e.target.value)} aria-label="What a label is" />
            <span className="mono muted">label; examples are</span>
            <input value={items} onChange={(e) => setItems(e.target.value)} aria-label="Many messages are called" />
          </div>
          <div className="own-go">
            <button type="submit" className="chip primary" disabled={!canUpload || !file || !name || uploadState.busy}>
              {uploadState.busy ? 'Checking and splitting…' : 'Add my data'}
            </button>
            {!canUpload && <small className="mono muted">Needs the local server.</small>}
          </div>
          {uploadState.error && <small className="run-error">{uploadState.error}</small>}
          {uploadState.note && <small className="own-note">{uploadState.note}</small>}
          <small className="mono muted">
            We split it once into practice and a sealed test (never opened here). Results on your own data are exploratory.
          </small>
        </form>
      )}
      {open === 'agent' && (
        <div className="own glass agent-form">
          <b>Plug in your own agent</b>
          <p className="muted">
            Without one, we race <b>a local Ollama model</b> (llama3.1:8b, running on this computer). To race yours instead, give its address: it gets one message and the list
            of kinds, and must answer one of those kinds (or <code>not_mine</code>). Anything else counts as wrong, the same rule our agent follows.
          </p>
          <code className="mono">{'POST your-url  { "text": "…", "labels": ["…"] }  →  { "label": "…" }'}</code>
          <label className="own-row">
            <span>Your agent’s address</span>
            <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="http://127.0.0.1:8091/classify" spellCheck={false} />
          </label>
          <label className="own-row">
            <span>What one answer costs you (USD)</span>
            <input value={usd} onChange={(e) => setUsd(e.target.value)} placeholder="e.g. 0.002" inputMode="decimal" />
          </label>
          {!chosen && <small className="mono muted">Pick a use case first: the check sends three of its practice messages.</small>}
          {chosen && !url.trim() && <small className="mono muted">Enter your agent’s address to check it.</small>}
          <div className="chips">
            <button type="button" className="chip small" disabled={!chosen || !url.trim() || probe.busy || !checkAgent} onClick={runCheck}>
              {probe.busy ? 'Checking…' : 'Check connection'}
            </button>
            <button type="button" className="chip small primary" disabled={!checkedOk || !priceOk || !setAgent} onClick={() =>
                setAgent?.({
                  url: url.trim(),
                  usd: price,
                  secs: probe.out?.tries.length ? probe.out.tries.reduce((a, t) => a + t.seconds, 0) / probe.out.tries.length : undefined,
                })
              }>
              Use my agent
            </button>
            {agent && (
              <button type="button" className="chip small" onClick={() => setAgent?.(null)}>
                Use the local Ollama agent instead
              </button>
            )}
          </div>
          {probe.error && <small className="run-error">{probe.error}</small>}
          {probe.out && (
            <div className="agent-tries mono">
              {probe.out.tries.map((t, i) => (
                <div key={i} className={t.valid ? 'ok' : 'no'}>
                  <span>{t.valid ? '✓ valid answer' : '✗ not a kind'}</span> “{t.text.slice(0, 60)}{t.text.length > 60 ? '…' : ''}” → <b>{JSON.stringify(t.answer)}</b> · {t.seconds}s
                </div>
              ))}
              {probe.out.ok ? (
                <small>Connected: every answer is one of this dataset’s kinds. (This checks the format only; right or wrong is decided in the test.)</small>
              ) : (
                <small className="run-error">
                  {probe.out.error}
                  {probe.out.tries.length > 0 && probe.out.tries.some((t) => !t.valid)
                    ? ` Your agent must answer one of the ${probe.out.labels.length} kinds (e.g. ${probe.out.labels.slice(0, 3).join(', ')}), or not_mine.`
                    : ''}
                </small>
              )}
            </div>
          )}
          {!priceOk && usd.trim() !== '' && <small className="run-error">Give a price between $0 and $10 per answer.</small>}
          {agent && (
            <small className="mono muted">
              Racing your agent at {agent.url} · ${agent.usd} per answer (your figure; it replaces our assumed prices). Not pinned: a re-run may differ.
            </small>
          )}
        </div>
      )}
    </div>
  );
}
