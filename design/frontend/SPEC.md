# Front end: approved design (locked 2026-11-04)

The approved look and behaviour is `story-reference.dc.html` in this folder, a working prototype
from the design canvas (Agent Verifier dashboard ideas, artboard "Story · white flying guide").
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
| 5 | Integrity | right | nod | Checks passed count; "Show all checks" expands the list |
| 6 | The reveal | left | sad: droop, sad eyes, dim jet | Accuracy gap, fades to a dark scene; "Yes, show me" jumps to scene 7 |
| 7 | Your call | right | point: arm toward the slider | Error-cost slider with live expected cost per arm |
| 8 | One case | left | curious | One wrong case; "Replay it" steps through the trace |
| 9 | Recap | right | proud: bounce, smile, wave | Result summary, recalls the guess, audit bundle download |

## The robot
- Original white guide (not a copy of any film character): pearl body, dark visor, cyan eyes,
  orange antenna light, small arms, hover glow.
- Idle life: hover, sway, blink, antenna sway, light pulse. Keeps moving with reduced motion on,
  only slower.
- Flight is tied to scroll and scales with the window: wind-up before leaving, an arc of a
  different height each time, overshoot past the stop, then settle. Banks into turns, shrinks when
  far, ground shadow shrinks as it rises, eyes look toward the content.
- One speech bubble with a tail, attached to the robot. It changes text per scene and hides while
  the robot is flying.
- Phones: the robot docks small in the bottom-right corner and stops flying.

## Page motion
- Text rises in as each scene enters, score bars fill, the reveal number lands with a blur-in.
- Scene backgrounds blend: white, soft blue tint (scenes 2, 4, 8), dark for the reveal.
- A fixed timeline at the bottom highlights the current scene and shows its name; each mark jumps
  to its scene.

## Look
- Type: Space Grotesk (UI), JetBrains Mono (labels, hashes).
- Colours: ink #0B0C10, blue #2242E8 (linear, primary), amber #D6620A / text #A84C08 (the agent),
  grey #8A8FA0 (prompt), tint #F1F3FF, robot eyes #5CE1E6.

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

Before the sealed run these come from `src/agent_committee/arms/dev_report.json` and are labelled
"DEV preview"; afterwards from `results/verdict.json`.

## Stack (approved by the user 2026-11-04)
React + Vite + TypeScript. Any further front-end dependency (chart, animation or state library)
still needs the user's approval first.

## Not decided yet (needs the user's approval)
- How the page gets the data (read the JSON files at build time, or a small API).
- Whether the robot is upgraded to a 3D (Spline) or Rive animation later.
