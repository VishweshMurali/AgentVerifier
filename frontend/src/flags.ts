// Page switches read from the address (?classic, ?tune, ?motion) or, in a packaged single file where
// there is no address to carry them, from window.__flags set before the page starts.
export function flag(name: string): boolean {
  if (typeof window === 'undefined') return false;
  const packed = (window as unknown as { __flags?: Record<string, boolean> }).__flags;
  return new URLSearchParams(window.location.search).has(name) || !!packed?.[name];
}
