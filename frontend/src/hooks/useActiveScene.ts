import { useEffect, useState } from 'react';

/**
 * The scene currently in the middle of the window, tracked with an IntersectionObserver.
 * A thin band across the middle of the viewport decides, so it also works when a scene is
 * taller than the screen (phones).
 */
export function useActiveScene(ids: readonly string[]): string {
  const [active, setActive] = useState(ids[0]);

  useEffect(() => {
    const sections = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);
    if (sections.length === 0 || typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id);
      },
      { rootMargin: '-45% 0px -45% 0px', threshold: 0 },
    );
    sections.forEach((s) => observer.observe(s));
    return () => observer.disconnect();
  }, [ids]);

  return active;
}

/**
 * Scenes currently on screen (at least a fifth visible). A scene's content rises in when it
 * enters and resets when it leaves, so the reveals replay on every visit, on every browser.
 */
export function useScenesInView(ids: readonly string[]): ReadonlySet<string> {
  const [seen, setSeen] = useState<ReadonlySet<string>>(() => new Set());

  useEffect(() => {
    const sections = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);
    if (typeof IntersectionObserver === 'undefined') {
      setSeen(new Set(ids));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        setSeen((prev) => {
          const next = new Set(prev);
          for (const e of entries) {
            if (e.intersectionRatio >= 0.2) next.add(e.target.id);
            else if (!e.isIntersecting) next.delete(e.target.id);
          }
          return next;
        });
      },
      { threshold: [0, 0.2] },
    );
    sections.forEach((s) => observer.observe(s));
    return () => observer.disconnect();
  }, [ids]);

  return seen;
}
