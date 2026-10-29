cmake_minimum_required(VERSION 3.16)
project(rubik_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# No fused multiply-add: ARM64 (and x86 with -march=native) would otherwise contract a*b+c into one rounding, and the
# iterative searches amplify last-bit differences from the Python reference.
add_compile_options(-ffp-contract=off)

add_library(rubik_engine STATIC
	src/capi.cpp src/classify.cpp src/cube.cpp src/engine.cpp src/geometry.cpp src/imaging.cpp
	src/imgproc.cpp src/legal.cpp src/lsd.cpp src/state.cpp src/track.cpp)
target_include_directories(rubik_engine PUBLIC include PRIVATE src)

# shared library for ctypes (python/rubik_engine.py) and FFI
add_library(rubik_engine_shared SHARED $<TARGET_OBJECTS:rubik_engine>)
set_target_properties(rubik_engine_shared PROPERTIES OUTPUT_NAME rubik_engine)
target_include_directories(rubik_engine_shared PUBLIC include)

add_executable(rubik_engine_tests tests/tests.cpp)
target_link_libraries(rubik_engine_tests rubik_engine)
target_include_directories(rubik_engine_tests PRIVATE src)
