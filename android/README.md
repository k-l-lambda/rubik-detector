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
