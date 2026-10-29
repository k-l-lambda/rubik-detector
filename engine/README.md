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
	tools/presence_check.py    cube-presence score on positives (video, photos) and generated negatives
	tools/real_video.py        per-frame detection on a real (unlabelled) video, overlays and cost / extent statistics
	tools/hud_video.py         a detection HUD on a video, with the engine's real latency (× a phone factor)
	tools/state_tracker.py     app-layer prototype on a camera stream: FOV → evidence of every frame → Bayesian colour
	                           posteriors → legal state (from five faces), panel top right
	tools/follow_tracker.py    app-layer prototype: acquire the state, lock it, then follow face turns (nearest states)
	../android/                Android prototype app (Kotlin, Compose, CameraX) running that pipeline on the camera

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
| views → one cube frame | `Evidence::addView(samples, rotation = -1, weight = 1)`, `Evidence::fuse()` | ms |
| state, confidence, colour model | `estimateState(evidence)` | ~0.1 s |
| video: per facelet colour posteriors, state from five faces | `estimateState(evidence, {.partial, .outlier})` | ~20 ms (60 frames) |
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

**Video: continuous accumulation (`tools/state_tracker.py`).** Every good frame is evidence: frames linked to the
previous one (small rotation up to a renaming, agreeing colours) form a chain placed through the renamings; a chain
is placed by colour. `estimateState` with `partial` and `reliability` gives per facelet `probability`, the Bayesian
posterior over the six colour clusters with a uniform prior, where each frame (or each `group` of views) is right
with an unknown probability r, uniform on [0, 1], and otherwise says nothing: P(c) ∝ ∫ Π_frames (r·L_f(c) + (1 − r)/6)
dr, L_f the frame's own colour posterior (exact by Gauss–Legendre: the integrand is a polynomial in r). However sharp
one frame's colour, it gives 7/12 = 58 %; two agreeing 90 %, three 98 %, four 99.7 % (the rule of succession for six
colours; the plain product of the likelihoods says 99.99 % from one sample a cluster's width away from the rest).
With `legal` once five faces are seen, the most probable legal state under these: the sixth face completed by the pieces' colours and parities, per facelet
`confidence` = the chosen colour's posterior under the legal constraint (uniform prior over legal states,
approximated by the best legal state without that colour there). On v102 at 30 fps (`?videoFps=30`, 4 s holds,
camera 30 ms): the first frame 58 %, the fifth face 58 % on colour alone but 98 % under the legal constraint; the state
exact from two holds at ~6 s (three holds ~11.5 s); at 3× latency still exact from five faces
(14 s). Where two completions tie (a 3-cycle of the hidden face's edges) their facelets say 0.5.

**Following turns (`tools/follow_tracker.py`).** Acquire as above; lock when the legal state is observed on all 54
facelets, every confidence ≥ 0.99, the same in 3 estimates in a row (a five-face completion can be confidently wrong
while the colour clusters are unsettled: on v103 it flipped between states, once at confidence 1.00 with 4 wrong).
Then each well-fitted frame (lattice cost ≤ 1.5 × the recent median: mid-turn fits fail it) is matched against the
states at most two face turns away — S, 18 single and 243 double turns, cached per S — over the 24 view renamings
with `rk_score_model` under the locked model (prior 10 nll per turn). The best is a match if at most 2 of the frame's
samples conflict with it (`conflicts` / `rk_conflicts`: the colour there costs > 4 nll above the sample's best — a
test of the stickers, not of the view's overall fit level, which drifts by 2 nll per sample between views). A new
state replaces S when it wins by ≥ 15 on 2 consecutive frames (or ≥ 80 on one); no match on 3 frames in a row: global
search (acquire again). v103 (klstudio `?videoTwists=8`, 30 fps): all 8 turns followed, no false switch, 0.3–1.2 s
after each turn ends at 1× latency (1.3–2.3 s at 2×); with frames cut so two turns fall in one gap (v103-gaps2) the
pairs are followed as one switch ("L' U'", "U F"); three in a gap (v103-gaps) start a global search.

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
- Cube presence (`ViewResult::presence`, `tools/presence_check.py`): mean lattice-line coverage of the best two faces
  (of a face's 8 lines, those with a fitting segment) − lattice cost, 0 for an implausible projected size. Negatives:
  5 seedream text-to-image scenes with no cube but many orthogonal edges (grid notebook, carton, keyboard, tiles,
  bookshelf; `~/data/rubik-renderer/tools/gen_negatives.py`), 30 views with flips / crops / ±8° rotations: presence
  ≤ 0.31 (≤ 0.45 with fx off by up to 25 %); positives: synthetic photos ≥ 0.70 (renders ≥ 0.83), real video 69 % of
  frames ≥ 0.5. Threshold 0.5. Cost alone separates less (a negative crop at 0.136); a flat grid covers one face at
  most, a cube two or three.
- Real hand-held video (`klstudio/footages.local/VID_20261028_225115.mp4`, 720×1280, 30 fps, GAN stickerless cube on
  a desk, hand-held, frames extracted to `~/data/rubik-renderer/real/vid225115`), every 2nd frame on its own (wide),
  fx 1070 (estimateFocal over 4 frames): before 69 % of frames fit (cost ≤ 0.15), 586 ms, but 11 of the accepted
  fits (all 1082 frames) were oversized lattices (projected cube larger than the frame, depth ~3 instead of ~10: a
  few long segments fit a pitch of several cubies at cost ~0.04) — the refinement left the framing prior's depth
  range. Now the position search is bounded to its cells' depth range ±16 %: 1 such fit left. Frames are never
  upscaled (a 720-wide frame at workWidth 1024 was 2× the pixels): 367 ms, 67 %. `starts = 16, joint = 5`
  (DetectOptions, opt-in): 71 % at 393 ms; on the synthetic photos it costs +15 % position time, hence not default.
  Failures: blur, the hand over the cube, and clear views whose position search lands one or two cubies off (a
  better pose exists at a lower cost: found from shifted starts); even the right pose often costs 0.15–0.3 here
  (rounded tiles, perspective) — the 0.15 acceptance threshold was set on renders. Tried and dropped: reversing
  the sign of a nearly edge-on axis (improved 27 of 337 failures, none under 0.15). Debug calls added:
  `rk_detect_segments` (segments with the result), `rk_pose_cost` / `Detector::poseCost` (cost of a given pose).
