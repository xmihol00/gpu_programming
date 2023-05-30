cmake_minimum_required(VERSION 3.16)
project(ass04-INGP LANGUAGES CXX CUDA VERSION 1.1)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CC "75" CACHE STRING "Which compute capability to build for? E.g. 61, 70, 75, 86")


####################################################################
# Executable
add_executable(ingp
	src/main.cpp
	include/DataLoader.h
	src/DataLoader.cpp
	src/fpng.cpp
	include/Camera.h
	include/RayGeneration.h
	include/FrequencyEncoder.h
	include/HashEncoder.h
	include/SampleEncoder.h
	include/SampleGenerator.h
	include/NetworkEvaluator.h
	include/OutputAccumulator.h
	include/CpuINGPEngine.h
	include/GpuINGPEngine.h
	src/CpuINGPEngine.cpp
	src/GpuINGPEngine.cpp
	src/GpuINGPEngine.cu
)
target_include_directories(ingp PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(ingp PROPERTIES CUDA_ARCHITECTURES ${CC})
set_target_properties(ingp PROPERTIES CXX_STANDARD 17)

if (UNIX)
	set(SSE 1)
endif()

if (SSE)
	message("With SSE")
	set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DFPNG_NO_SSE=0 -msse4.1 -mpclmul")
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DFPNG_NO_SSE=0 -msse4.1 -mpclmul")
else()
	message("Without SSE")
	set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DFPNG_NO_SSE=1")
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DFPNG_NO_SSE=1")
endif()

if (MSVC)
   target_compile_options(ingp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:/O2> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:/Od>)
else()
   target_compile_options(ingp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-O3> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-O0>)
endif()