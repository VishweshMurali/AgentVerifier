import type { ReactNode } from 'react';
import type { Mood, Side } from '../story/scenes.ts';

interface RobotProps {
  mood: Mood;
  side: Side;
  /** The single speech bubble's content; it travels with the robot. */
  children: ReactNode;
}

/** The original white flying guide. Flight, moods and idle life are all CSS (see styles.css). */
export function Robot({ mood, side, children }: RobotProps) {
  return (
    <div className={`robot mood-${mood} side-${side}`}>
      <div className="r-ground" aria-hidden="true" />
      <div className="r-fly-y">
        <div className="say" role="status" aria-live="polite">
          {children}
        </div>
        <div className="r-body" aria-hidden="true">
          <svg viewBox="0 0 200 260" focusable="false" style={{ width: '100%', height: 'auto', overflow: 'visible' }}>
            <defs>
              <radialGradient id="pearl" cx="36%" cy="28%" r="85%">
                <stop offset="0" stopColor="#FFFFFF" />
                <stop offset=".55" stopColor="#EEF0F5" />
                <stop offset="1" stopColor="#AEB4C4" />
              </radialGradient>
              <linearGradient id="visor" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#1D2030" />
                <stop offset="1" stopColor="#05060B" />
              </linearGradient>
              <radialGradient id="jet-glow" cx="50%" cy="50%" r="50%">
                <stop offset="0" stopColor="#6FB6FF" stopOpacity=".9" />
                <stop offset="1" stopColor="#6FB6FF" stopOpacity="0" />
              </radialGradient>
              <filter id="soft-glow" x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur stdDeviation="3" result="b" />
                <feMerge>
                  <feMergeNode in="b" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <g className="r-bank">
              <g className="r-hover">
                <ellipse className="r-jet" cx="100" cy="236" rx="34" ry="14" fill="url(#jet-glow)" />
                <g className="r-sway">
                  <g className="r-arm">
                    <path d="M44 140 C 26 150, 22 176, 32 196 C 38 204, 48 200, 48 190 C 46 172, 50 156, 58 146 Z" fill="url(#pearl)" />
                  </g>
                  <g className="r-wave">
                    <path d="M156 140 C 174 150, 178 176, 168 196 C 162 204, 152 200, 152 190 C 154 172, 150 156, 142 146 Z" fill="url(#pearl)" />
                  </g>
                  <path d="M58 132 C 58 118, 142 118, 142 132 L 136 200 C 132 226, 68 226, 64 200 Z" fill="url(#pearl)" />
                  <ellipse cx="82" cy="146" rx="14" ry="6" fill="#FFFFFF" opacity=".8" />
                  <circle className="r-lamp" cx="100" cy="176" r="5" fill="#2242E8" />
                  <g className="r-head">
                    <g className="r-antenna">
                      <line x1="100" y1="8" x2="100" y2="26" stroke="#C7CBD6" strokeWidth="3.5" strokeLinecap="round" />
                      <g className="r-lamp-on">
                        <circle className="r-lamp" cx="100" cy="7" r="5.5" fill="#D6620A" />
                      </g>
                    </g>
                    <ellipse cx="100" cy="72" rx="74" ry="50" fill="url(#pearl)" />
                    <ellipse cx="72" cy="40" rx="24" ry="9" fill="#FFFFFF" opacity=".9" />
                    <ellipse cx="100" cy="78" rx="56" ry="32" fill="url(#visor)" />
                    <g className="r-look">
                      <g className="r-wake" filter="url(#soft-glow)">
                        <g className="eyes eyes-open">
                          <ellipse className="r-eye" cx="80" cy="78" rx="10" ry="12" fill="#5CE1E6" />
                          <ellipse className="r-eye" cx="120" cy="78" rx="10" ry="12" fill="#5CE1E6" />
                        </g>
                        <g className="eyes eyes-wide">
                          <ellipse className="r-eye" cx="80" cy="77" rx="12.5" ry="15" fill="#5CE1E6" />
                          <ellipse className="r-eye" cx="120" cy="77" rx="12.5" ry="15" fill="#5CE1E6" />
                          <circle cx="84" cy="72" r="3.5" fill="#FFFFFF" />
                          <circle cx="124" cy="72" r="3.5" fill="#FFFFFF" />
                        </g>
                        <g className="eyes eyes-smile">
                          <path d="M70 82 Q 80 70, 90 82" fill="none" stroke="#5CE1E6" strokeWidth="5" strokeLinecap="round" />
                          <path d="M110 82 Q 120 70, 130 82" fill="none" stroke="#5CE1E6" strokeWidth="5" strokeLinecap="round" />
                        </g>
                        <g className="eyes eyes-sad">
                          <ellipse cx="80" cy="84" rx="10" ry="5" fill="#5CE1E6" transform="rotate(-16 80 84)" />
                          <ellipse cx="120" cy="84" rx="10" ry="5" fill="#5CE1E6" transform="rotate(16 120 84)" />
                        </g>
                      </g>
                    </g>
                  </g>
                </g>
              </g>
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
}
