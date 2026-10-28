#!/bin/bash
# Host check of the Android app's pipeline: builds the JNI library for the host (g++) and the app's platform-free
# Kotlin sources (Engine.kt, Cube.kt, Colors.kt, Pipeline.kt) with HostCheck.kt, then runs it on a rendered video.
#	hostcheck/run.sh VIDEO_DIR [latency scale]
# Needs: g++, a JDK with JNI headers (JNI_INCLUDE, default: the first jni.h found under JAVA_HOME or /usr/lib/jvm),
# Kotlin compiler jars (KOTLINC_CP: a classpath with kotlin-compiler-embeddable and its dependencies; default from
# ~/.gradle caches) and the Kotlin stdlib (KOTLIN_STDLIB).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${OUT:-/tmp/rubik-hostcheck}
mkdir -p "$OUT"
C=~/.gradle/caches/modules-2/files-2.1
if [ -z "$JNI_INCLUDE" ]; then
	JNI_H=$(find "${JAVA_HOME:-/usr/lib/jvm}" -name jni.h 2>/dev/null | head -1)
	[ -n "$JNI_H" ] || { echo "no jni.h: set JNI_INCLUDE"; exit 1; }
	JNI_INCLUDE=$(dirname "$JNI_H")
fi
KOTLIN_STDLIB=${KOTLIN_STDLIB:-$(find $C/org.jetbrains.kotlin/kotlin-stdlib -name "kotlin-stdlib-*.jar" ! -name "*sources*" | sort | tail -1)}
KOTLINC_CP=${KOTLINC_CP:-$(find $C/org.jetbrains.kotlin/kotlin-compiler-embeddable $C/org.jetbrains.kotlin/kotlin-daemon-embeddable \
	$C/org.jetbrains.intellij.deps/trove4j $C/org.jetbrains.kotlinx/kotlinx-coroutines-core-jvm $C/org.jetbrains/annotations \
	-name "*.jar" ! -name "*sources*" | tr '\n' ':')$KOTLIN_STDLIB:$(find $C/org.jetbrains.kotlin/kotlin-reflect -name "*.jar" ! -name "*sources*" | head -1)}

if [ ! -f "$OUT/librubik_jni.so" ] || [ -n "$(find "$ROOT/engine/src" "$ROOT/engine/include" "$ROOT/android/app/src/main/cpp" -newer "$OUT/librubik_jni.so" -name "*.*pp")" ]; then
	echo "building librubik_jni.so (host)"
	g++ -std=c++17 -O2 -fPIC -shared -ffp-contract=off -I"$JNI_INCLUDE" -I"$JNI_INCLUDE/linux" -I"$ROOT/engine/include" -I"$ROOT/engine/src" \
		"$ROOT"/engine/src/*.cpp "$ROOT/android/app/src/main/cpp/rubik_jni.cpp" -o "$OUT/librubik_jni.so"
fi
SRC="$ROOT/android/app/src/main/java/com/klstudio/rubikdetector"
java -cp "$KOTLINC_CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -nowarn -no-reflect -no-stdlib -jvm-target 11 -cp "$KOTLIN_STDLIB" \
	-d "$OUT/classes" "$SRC/Engine.kt" "$SRC/Cube.kt" "$SRC/Colors.kt" "$SRC/Pipeline.kt" "$HERE/HostCheck.kt" 2>&1 | grep -v "^warning: unable to find kotlin" || true
java -Djava.library.path="$OUT" -cp "$OUT/classes:$KOTLIN_STDLIB" HostCheckKt "$@"
