# rubik engine (C++)

The image-processing core of `rubik_detector` as a C++17 library with no dependencies (no OpenCV), for native mobile
apps. The Python package stays the reference; `tools/parity.py` checks the engine against it on the labelled samples.

## Layout

	include/rubik/engine.h     C++ API (the primary one)
	include/rubik/rubik_c.h    flat C ABI over it (ctypes, JNI, Swift without C++ interop)
	src/                       stages — imgproc/lsd/optimize are the ports verified bit-exact in klstudio's hexiamond
	                           recognizer (INTER_AREA, OpenCV 4.9 LSD, scipy Nelder–Mead)
	tests/tests.cpp            unit tests without data (tables, turns, rotations, legal search)
	python/rubik_engine.py     ctypes binding
	tools/parity.py            engine vs the Python reference over all samples
	tools/api_check.py         tracking / renamed frames / turn scoring on the samples
	tools/focal_check.py       focal length estimation vs the samples' true K
	tools/hud_video.py         a detection HUD on a video, with the engine's real latency (× a phone factor)

Build: `cmake -S engine -B engine/build && cmake --build engine/build -j` (static `librubik_engine.a`, shared
`librubik_engine.so`, `rubik_engine_tests`). `-ffp-contract=off` is set on purpose: ARM64 would otherwise fuse a·b+c
and the iterative searches amplify last-bit differences.

## API in one page

| stage | call | cost (desktop, 1024 px) |
|---|---|---|
| one photo / frame → pose + 27 facelet colours | `Detector::detect(image, camera, {Search::Wide})` | ~0.56 s |
| unknown camera: focal length, once per camera | `Detector::estimateFocal(images)` | ~1.7–3 s for one image, ~5–9 s for three |
| tracked frame from a prior pose | `Detector::detect(image, camera, {Search::Local, prior})` | ~0.22 s (position 58 ms) |
| colours at a known pose | `Detector::sample(image, camera, pose)` | ~11 ms (Lab only inside the 27 patches) |
| views → one cube frame | `Evidence::addView(samples, rotation = -1)`, `Evidence::fuse()` | ms |
| state, confidence, colour model | `estimateState(evidence)` | ~0.1 s |
| score candidate states for a view | `scoreStates(model, samples, rotation, states)` | < 1 ms each |
| frame renaming / chaining | `relativeRotation`, `composeRotations`, `inverseRotation` | — |
| moves | `parseTurn`, `applyTurn`, `applyMoves`, `faceTurnNames`, `compareStates` | — |

`Detector` keeps scratch buffers: one per thread. Most of a detect is the four LSD passes (~160 ms) and the position
search; the three views of a photo set are independent and can run on three threads.

### Usage patterns the API is cut for

**Three photos (the current protocol).** `detect` each (Wide), `addView` each without a rotation, `fuse()`,
`estimateState`. Matches the Python pipeline group for group.

**Video: accumulate until confident.** Detect the first frame Wide, then every frame Local with the previous pose as
the prior (fall back to Wide when `latticeCost` jumps). Chain the frame-to-frame renaming:
`g = composeRotations(g, relativeRotation(prevPose, pose))` and `addView(samples, g)` — fusion by colour is then not
needed (`fuse` places only views added without a rotation). `estimateState` whenever a new face has been seen;
accept when `minPosterior` (or the per-facelet `posterior`) passes the app's threshold and the state stays stable
over a few estimates. `removeView` drops blurred or doubtful frames.

**Turns on a known state.** Keep the state and the `ColourModel` from the estimate. Per new frame (tracked pose,
`sample`), `scoreStates(model, samples, g, {state, applyMoves(state, m) for m in faceTurnNames()})`: the lowest wins;
the margin is how sure. On the samples the true state beats all 18 single turns in every view, margin ≥ 12.7 nll
(median 60). Mid-turn frames (a layer rotated part way) are not modelled yet: the lattice fit assumes an intact cube,
so an app should score frames where the turn has settled (lattice cost back to normal).

## Verification (2026-10-28, 140 scenarios: renders g/n × 50, seedream edits g/n × bg/hand × 10)

- `tools/parity.py --mode working` (same working image and K as the Python pipeline): engine exact 139/140, Python
  139/140, the same scenarios (n45 render fails in both: glare + two close blues, colour stage); per-view sticker
  centre error vs truth mean 0.026–0.059 cubie in both. `--mode file` (original 2304-px edits, the engine resizes):
  identical.
- `tools/api_check.py`: local search from priors perturbed by 2–4° and 0.2–0.4 cubie, 420/420 views located; priors
  given in any of the 24 renamed frames, 10080/10080; `sample()` at the detected pose = `detect()`'s colours, 420/420;
  turn scoring, 417/417 views.
- Not bit-identical to Python, by design: RANSAC uses its own generator (Python draws from numpy's PCG64), and
  BGR→Lab uses the exact formula rather than OpenCV's 16-bit LUT interpolation (≤ 0.5 ΔLab). Results agree at the
  level of states and sticker positions.
- Focal length (`Detector::estimateFocal`, `tools/focal_check.py`, 80 groups with K hidden): a separate step before
  detection — `detect` refuses `fx <= 0`. Criterion: the share of *all* segment weight the lattice explains (the
  lattice cost alone, averaged over the segments assigned to the axes, picked fx 2–5× too long, where few segments fit
  and fit well: 9/40 render groups). Coarse: 8 log-spaced fx / long side over 0.45–2.6 at 512 px, each a quick wide
  solve (3 starts, 1 joint refinement), narrows the range; fine: 6 golden-section steps at 1024 px, each a *local*
  solve from the coarse pose (at k·f the same projection is the centre (x, y, k·z)), ~70 ms each. Error: one view
  p50 2–3 %, p95 7–10 %, max 15 %; three views together max 9–16 %. With the joint estimate all 240 views located and
  all 80 groups exact. 384 px coarse gave occasional outliers (max 99 % log error on one hand view): 512 is the
  default. Real photo without EXIF (`klstudio/footages.local/rubik-samples/IMG_20190901_150650.jpg`): fx ≈ 1.1 × long
  side, support 0.67; the coarse curve peaks clearly at 1.23.
- Speed: ~0.56 s per view vs ~4.6 s in Python on the same machine (single thread each).
