import { SCENES, sceneById } from '../story/scenes.ts';

export function Timeline({ active }: { active: string }) {
  return (
    <nav className="timeline" aria-label="Scenes">
      <div>
        {SCENES.map((sc) => (
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
        <span className="scene-name">{sceneById(active).name}</span>
      </div>
    </nav>
  );
}
