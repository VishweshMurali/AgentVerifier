// What the robot says while a quick test runs: plain steps read from the test's own output
// (server/flow.py read_progress). No guessing: a step shows only once the test has printed it.

export type RunProgress = {
  phase?: string;
  training?: string;
  trained?: string[];
  scored?: number;
  planned?: number;
  drawn?: number;
  simpler?: string;
  stopped_at?: number;
  of?: number;
  looks?: { n: number }[];
  verdict?: string;
};

export type RunStep = { text: string; now: boolean };

export type RunLog = {
  /** One short sentence that changes only when the stage changes (read out to screen readers). */
  headline: string;
  /** The steps so far, the last one still running unless the test is done. */
  steps: RunStep[];
};

const FREE_MODELS = 6; // majority, linear, naive bayes, linear SVM, random forest, boosting

export function runLog(p: RunProgress | null | undefined, name: (arm: string) => string = (a) => a): RunLog {
  const g = p ?? {};
  const A = name('agent') === 'agent' ? 'Your agent' : name('agent');
  const a = A === 'Your agent' ? 'your agent' : `the ${A.charAt(0).toLowerCase()}${A.slice(1)}`;
  const steps: RunStep[] = [];
  const total = g.drawn ?? g.planned;
  const done = (text: string) => steps.push({ text, now: false });
  const now = (text: string) => steps.push({ text, now: true });

  if (!total) {
    now('Starting the test and picking messages…');
    return { headline: 'Starting your test…', steps };
  }
  done(`Picked ${total} real messages to test`);

  const trained = g.trained ?? [];
  if (g.phase === 'training' || (!g.simpler && !g.scored)) {
    if (trained.length >= FREE_MODELS && !g.training) {
      done(`Trained ${trained.length} free models`);
      now(`Picking the free model ${a} must beat…`);
      return { headline: `Picking the model ${a} must beat…`, steps };
    }
    if (trained.length) done(`Trained ${trained.length} of ${FREE_MODELS} free models`);
    now(g.training ? `Training ${name(g.training)}…` : 'Training the free models…');
    return { headline: `Training the free models to race ${a}…`, steps };
  }
  done(`Trained ${trained.length || FREE_MODELS} free models`);
  if (g.simpler) done(`${name(g.simpler)} is the one to beat`);

  if (g.phase === 'done') {
    done(g.stopped_at && g.of && g.stopped_at < g.of ? `Stopped early at ${g.stopped_at} of ${g.of}: clear enough` : `All ${total} messages answered`);
    done('Your answer is ready');
    return { headline: 'Done! Your result is ready.', steps };
  }
  if (g.phase === 'finishing') {
    done(g.stopped_at && g.of && g.stopped_at < g.of ? `Stopped early at ${g.stopped_at} of ${g.of}: clear enough` : `${A} answered all ${total}`);
    now('Scoring the other models and writing your answer…');
    return { headline: 'Almost done: writing your answer…', steps };
  }
  const looks = g.looks?.length ?? 0;
  if (looks) done(`Compared them ${looks} time${looks > 1 ? 's' : ''}: too close to stop yet`);
  now(`${A} has answered ${g.scored ?? 0} of ${total}…`);
  return { headline: `${A} is answering real messages…`, steps };
}
