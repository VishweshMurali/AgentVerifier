import { SCENES, type SceneDef } from '../story/scenes.ts';

export function Timeline({ active, scenes = SCENES, counted = false }: { active: string; scenes?: readonly SceneDef[]; counted?: boolean }) {
  const current = scenes.find((s) => s.id === active) ?? scenes[0];
  const step = scenes.indexOf(current) + 1;
  return (
    <nav className="timeline" aria-label="Scenes">
      <div>
        {scenes.map((sc) => (
          <a
            key={sc.id}
            href={`#${sc.id}`}
            aria-label={sc.name}
            title={sc.name}
            className={sc.id === active ? 'on' : undefined}
            aria-current={sc.id === active ? 'step' : undefined}
          >
            <span />
          </a>
        ))}
        <span className="scene-name">
          {/* opt-in (mockup): "2 of 6 · The race", so the reader knows where they are */}
          {counted && <b className="step-no">{step} of {scenes.length} · </b>}
          {current.name}
        </span>
      </div>
    </nav>
  );
}
