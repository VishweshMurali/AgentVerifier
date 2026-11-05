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
