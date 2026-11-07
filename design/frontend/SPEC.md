# Front end: approved design (locked 2026-11-04, revised 2026-11-05)

The approved look and behaviour is `story-reference.dc.html` in this folder, a working prototype
from the design canvas (final version: teal on light, dotted 3D ribbon background, night toggle).
Build the front end to match it. Changes to anything below need the user's approval.

## Rules carried over from the project
- The page only shows results. Every number, verdict and robot line comes from deterministic code
  reading the pipeline's outputs. No LLM writes or chooses any text on the page.
- The robot's lines are fixed scripts chosen by the results and the scene, never generated.
- Numbers in the prototype are DEV preview values. The build reads them from the outputs below.

## Structure
One scrolling story of 9 scenes. Each scene fills exactly one screen height, so the robot's
evenly spaced stops land on each scene. No scroll snapping.

| # | Scene | Robot side | Robot mood | Content |
|---|---|---|---|---|
| 1 | The question | right | hello: wakes up, smiles, waves | Title, one-line promise |
| 2 | Contenders | left | curious: head tilt, wide eyes | The three arms; asks "which one wins?" with 3 answer buttons |
| 3 | Fair rules | right | thinking: tilt, eyes up | Rules locked before the test |
| 4 | Scores | left | thinking | Accuracy bars per arm; pays off the user's guess |
| 5 | Integrity | right | nod | Checks passed count; "Show all checks" expands the list (see Checks list) |
| 6 | The reveal | left | sad: droop, sad eyes, dim jet | Accuracy gap, fades to a dark scene; "Yes, show me" jumps to scene 7 |
| 7 | Your call | right | point: arm toward the slider | Error-cost slider with live expected cost per arm |
| 8 | One case | left | curious | One wrong case; "Replay it" steps through the trace |
| 9 | Recap | right | proud: bounce, smile, wave | Result summary, recalls the guess, audit bundle download |

## The robot
- Original white guide (not a copy of any film character): pearl body, dark visor, cyan eyes,
  teal antenna light and glow, small arms. Flight is a spring that follows a path through the
  scene stops; it launches up from below on load.
- Idle life: hover, sway, blink, antenna sway, light pulse. Keeps moving with reduced motion on,
  only slower.
- Flight is tied to scroll and scales with the window: wind-up before leaving, an arc of a
  different height each time, overshoot past the stop, then settle. Banks into turns, shrinks when
  far, ground shadow shrinks as it rises, eyes look toward the content.
- One speech bubble with a tail, attached to the robot. It changes text per scene and hides while
  the robot is flying, but never while one of its buttons has keyboard focus. Only its line is
  announced to screen readers; its buttons sit outside the announced part.
- Keyboard path (approved 2026-11-05): the robot comes after the scenes in the page order (it is
  fixed on screen, so it looks the same). Each scene whose bubble has buttons ends with a
  "Go to the robot’s buttons" button that shows only on keyboard focus and moves focus to them.
- Phones: the robot docks small in the bottom-right corner and stops flying.

## Page motion
- Text rises in as each scene enters, score bars fill, the reveal number lands with a blur-in.
- Headlines reveal word by word; numbers count up when their scene comes into view.
- Background: a canvas of dots forming one diagonal ribbon twisted in 3D, cut into three lanes
  (Linear, Prompt, Your agent). Each scene has its own palette, blended while scrolling. Lanes split
  at scene 2, sit at their score heights at scene 4, the agent lane drops at the reveal, all rejoin
  at the recap. Dots fly in on load, part for the cursor and the robot, thin out around text, and
  burst and re-form on each scene change. A ring opens around the reveal number.
- Corner brackets and a scan line only in the reveal. A small readout: source, case count, scene.
- Wheel scrolling is eased. All motion is driven by JavaScript (works on every browser, phones
  included). Reduced motion slows the motion instead of stopping it; eased wheel scrolling stays
  off under reduced motion (confirmed by the user 2026-11-05).
- A fixed timeline at the bottom highlights the current scene and shows its name; each mark jumps
  to its scene. Scene links move focus and the address to the scene.
- Header: brand, night toggle, and a "Try your error cost" button that jumps to scene 7
  (renamed from "Verify an agent", approved 2026-11-05).

## Checks list (scene 5, approved 2026-11-05)
- Each check shows a plain, short name (for example "Copies of DEV text", "Per-label bias",
  "No overfitting to DEV: Linear"), its value and its pre-registered limit. The internal name
  (`leakage`, `dev_gap[linear]`, …) and the engine's detail line sit in the detail layer under it.
- Pill words: "Passed"; "Noted" in a neutral colour for a failed check that is only reported;
  "Failed" (ink) for a failed check that binds the verdict.

## Look
- Type: Space Grotesk (UI), JetBrains Mono (labels, hashes).
- Colours: ink #253031, teal #1F6E73 (linear, primary), bright teal #2FB5BB, grey #8FA3A4
  (prompt), the agent in ink. No orange. Robot eyes #5CE1E6. Night mode: dark ground, glowing dots.
- Frosted glass panels with a thin glowing edge. Headlines 96/60px, body 22px.

## Data each scene needs (from the pipeline outputs)
| Scene | Values |
|---|---|
| 2, 4 | Arm names, accuracy per arm, case count |
| 3 | Number of checks and price tables from `prereg.yaml` |
| 5 | Each check's name, value and pass state from the checks results |
| 6 | Accuracy gap agent minus best simpler arm, with its CI once the sealed run exists |
| 7 | Error rate and tokens per case per arm, price table; expected cost = inference cost + error rate × error cost |
| 8 | One disagreement case and its agent trace from the run cache |
| 9 | Verdict, evidence level, commit, model digest, layer fingerprint, audit bundle |

All of it comes from `frontend/public/data/story.json` (scripts/export_frontend_data.py): the DEV
rehearsal verdict, labelled as a rehearsal, until the sealed `results/verdict.json` exists. The
numbers typed into the prototype are placeholders and are never used.

## Stack (approved by the user 2026-11-04)
React + Vite + TypeScript. Any further front-end dependency (chart, animation or state library)
still needs the user's approval first.

## Not decided yet (needs the user's approval)
- How the page gets the data (read the JSON files at build time, or a small API).
- Whether the robot is upgraded to a 3D (Spline) or Rive animation later.
