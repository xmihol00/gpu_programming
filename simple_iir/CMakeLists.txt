cmake_minimum_required(VERSION 3.16)
project(ass01-02-IIR LANGUAGES CXX CUDA VERSION 1.1)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CC "75" CACHE STRING "Which compute capability to build for? E.g. 61, 70, 75, 86, 89, 90")


####################################################################
# Executable
add_executable(iir
	src/main.cpp
	src/ConfigLoader.cpp
    src/IirFilterEngine.cpp
	src/CPUIirFilterEngine.cpp
	src/IirFilterEngine.cu
	src/filter.cpp
	include/IirFilterEngine.cuh
	include/IirFilterEngine.h
	include/CPUIirFilterEngine.h
	include/filter.h
	include/ConfigLoader.h
)
target_include_directories(iir PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(iir PROPERTIES CUDA_ARCHITECTURES ${CC})
set_target_properties(iir PROPERTIES CXX_STANDARD 17)

if (MSVC)
   target_compile_options(iir PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:/O2> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:/Od>)
else()
   target_compile_options(iir PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-O3> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-O0>)
endif()