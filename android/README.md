# Rubik Detector — Android prototype

Kotlin + Jetpack Compose + CameraX app over the C++ engine (`../engine`, built by the NDK through
`app/src/main/cpp/CMakeLists.txt`), running the application layer of `engine/tools/state_tracker.py` and
`follow_tracker.py` on the live camera, with debug information on screen.

## Build

Open `android/` in Android Studio (Koala or newer: AGP 8.5, Gradle 8.7, Kotlin 2.0, JDK 17) with the NDK and CMake
3.22.1 installed from the SDK manager; run the `app` configuration on a device (arm64-v8a; x86_64 for emulators).
The engine is always compiled -O3 (a debug engine is ~10× slower). No Gradle wrapper scripts are committed: Android
Studio uses `gradle/wrapper/gradle-wrapper.properties`; for the command line, `gradle wrapper` once.

## Layout

	app/src/main/cpp/rubik_jni.cpp     JNI bridge (flat double arrays, parsed in Engine.kt); turns frames upright
	.../Engine.kt                      engine calls, Detection / Estimate / Focal types
	.../Cube.kt                        rotation tables (from the engine), renaming / chaining, rotation log / exp
	.../Pipeline.kt                    focal → acquire (evidence, Bayesian posteriors, legal state, lock) → follow
	                                   (≤ 2 turns, conflicts, switch) → global search; immutable Snapshot per frame
	.../CameraSession.kt               CameraX preview + RGBA analysis (keep only latest) on one worker thread
	.../MainActivity.kt                UI
	hostcheck/                         the platform-free sources above run on the desktop against a rendered video

Engine.kt, Cube.kt, Colors.kt and Pipeline.kt have no Android imports, so `hostcheck/run.sh VIDEO_DIR [scale]`
compiles them unchanged with a host build of the JNI library and runs the pipeline on a klstudio 🎬 video with the
latency timeline of the Python tools (needs g++, a JDK's `jni.h` via `JNI_INCLUDE`, Kotlin compiler jars from the
Gradle cache). On v103 (8 turns): locks exact at ~12 s, all 8 turns followed with the state equal to the labels, at
1× and 2× latency — the same as `follow_tracker.py`.

## Tracking every frame, detecting in the background

With "track" on (default) the analysis thread runs the Coordinator on every camera frame: the frame goes into a ring
(24 frames), the engine's FrameTracker follows the visible lattice vertices into it (pyramidal Lucas–Kanade, robust
pose; ~20–25 ms on the desktop at 720×1280) and the HUD draws the tracked pose — its facelet quads in the last
detection's colours, the tracked points (magenta), the axes. A second thread runs the detection pipeline whenever free
on the newest frame, with the tracker's pose at that frame as the local search's prior (in a crop around it). A
finished detection is for an older frame k: compared with the tracker's own pose at k, agreeing it pulls the tracked
pose 30 % of the way towards it (moved on to the newest frame by the tracker's own motion, re-seeding the points);
disagreeing with a healthy track it is only taken once a second one agrees; with no track (or a lost one) the tracker
is seeded at k (the detected frame is pinned, it can outlast the ring) and caught up to the newest frame in at most
three steps. Two detections in a row without a cube end the track. Without a track the last detection is drawn dimmed.

Host check (`MAIN=CoordCheckKt hostcheck/run.sh VIDEO [scale] [HUD video dir] [fx]`, the two threads on a simulated
timeline, each frame shown with the newest HUD ready by its arrival):

| video | tracked HUD on | its age | error vs the shown frame | detection-only HUD: age, error |
|---|---|---|---|---|
| v103 (render, truth) | 92 % of frames | 33 ms | median 0.03, > 0.3 cubie on 7 % | 367 ms, median 0.05, > 0.3 on 27 % |
| real video, 1× | ~56–63 % | 33 ms | median 0.06–0.08 | 300–470 ms, median 0.31–0.42 |
| real video, 2× (phone-like) | ~20 % | 100 ms | median 0.15–0.2 | 1.2 s, median 0.5 |

On the real video the track is lost under the hand and in blur (~10 times in 36 s), and at 2× latency a detection
(~0.8 s) often comes too late to catch up from; the tracked HUD is then far better where it is shown, the detection
alone everywhere stale. Not verified on a device: CameraSession.kt / MainActivity.kt changes are not compiled here.

## Cube presence

Every frame is first asked whether there is a cube at all: the engine's presence score of the detection (line
coverage of the best two faces − lattice cost) must reach 0.5. Below it nothing is evidence, followed or drawn (the
headline says "no cube", only the debug segments show), and no focal estimate is made: before the focal length is
known, the check runs with fx = 1.0 × long side. Host check: 5 generated scenes without a cube shown for a second each
— no frame present, the focal stage never left; the real video — 71/98 processed frames present; v103 — unchanged
(locks at 11.9 s, all 8 turns followed).

## Screen

- Preview with the latest detection: sampled sticker quads in their measured colours (white outline; red when the
  fit failed the cost test), optional LSD segments by axis family (red / green / blue, grey unassigned) and the cube
  axes from the pose.
- Top left: stage (FOCAL / ACQUIRE n/54, confidence / FOLLOW n turns), detection mode and lattice cost.
- Top right: cube net — colour per facelet, best component of its colour posterior in percent, border = confidence of
  the state's colour (green ≥ 99 %, yellow ≥ 90 %, red), white corner = completed from the legal search, not observed.
  While following: the followed state.
- Debug text: processing time (pipeline, frame rotation), processed fps, camera frames per processed frame, frame age;
  focal estimate (coarse curve, support) and the lens data from Camera2 (`LENS_INTRINSIC_CALIBRATION` or focal length /
  sensor width); detection timings, cost, support, segments, pose; link to the previous frame (angle up to renaming,
  colour ΔE); evidence frames / chains / coverage; estimate (legal, min posterior / confidence, ms, stable count);
  follow (state, turns, last decision with margin, candidates, conflicts, scoring ms); recent events.
- Controls: reset (keep focal), FOV (estimate again), freeze, lens fx (use the camera's intrinsics instead of
  estimating), dump (upright PNG + the debug text to `Android/data/com.klstudio.rubikdetector/files/dumps/`, for
  the Python tools), stickers / segments / axes overlays.

Not verified on a device yet: the Android-only files (CameraSession.kt, MainActivity.kt, Gradle) have not been
compiled here (no SDK on the development machine). The JNI bridge and the engine were cross-compiled for arm64 with
NDK r21b.
