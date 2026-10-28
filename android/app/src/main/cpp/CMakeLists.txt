# The detection engine (rubik-detector/engine, only the static library: its tests and shared library are excluded)
# and the JNI bridge. Always optimized: a Debug build of the engine is ~10× slower, too slow to use on a camera.
cmake_minimum_required(VERSION 3.22.1)
project(rubik_jni CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../engine engine EXCLUDE_FROM_ALL)
target_compile_options(rubik_engine PRIVATE -O3)

add_library(rubik_jni SHARED rubik_jni.cpp)
target_compile_options(rubik_jni PRIVATE -O3 -ffp-contract=off)
target_link_libraries(rubik_jni rubik_engine log)
