// Minimal types for the Node built-ins the *.test.ts files use, so `tsc --noEmit` checks the
// tests without adding @types/node (not an approved dependency). Tests run with `node --test`.

interface ImportMeta {
  readonly url: string;
}

declare module 'node:test' {
  type Fn = () => void | Promise<void>;
  export function test(name: string, fn: Fn): Promise<void>;
  export function describe(name: string, fn: Fn): Promise<void>;
  export function it(name: string, fn: Fn): Promise<void>;
  export default test;
}

declare module 'node:assert/strict' {
  interface Assert {
    (value: unknown, message?: string): asserts value;
    ok(value: unknown, message?: string): asserts value;
    fail(message?: string): never;
    equal(actual: unknown, expected: unknown, message?: string): void;
    notEqual(actual: unknown, expected: unknown, message?: string): void;
    deepEqual(actual: unknown, expected: unknown, message?: string): void;
    match(value: string, pattern: RegExp, message?: string): void;
    doesNotMatch(value: string, pattern: RegExp, message?: string): void;
    throws(fn: () => unknown, expected?: RegExp | ((err: unknown) => boolean), message?: string): void;
    rejects(
      promise: Promise<unknown> | (() => Promise<unknown>),
      expected?: RegExp | ((err: unknown) => boolean),
      message?: string,
    ): Promise<void>;
  }
  const assert: Assert;
  export default assert;
}

declare module 'node:fs' {
  export function readFileSync(path: URL | string, encoding: 'utf8'): string;
}
