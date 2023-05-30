cmake_minimum_required(VERSION 3.16)
project(ass03-MLP LANGUAGES CXX CUDA VERSION 1.1)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CC "75" CACHE STRING "Which compute capability to build for? E.g. 61, 70, 75, 86")


####################################################################
# Executable
add_executable(mlp
	src/main.cpp
	include/GpuMlpEngine.h
	include/CpuMlpEngine.h
	include/DataLoader.h
	src/GpuMlpEngine.cpp
	src/CpuMlpEngine.cpp
	src/DataLoader.cpp
	src/GpuMlpEngine.cu
	
)
target_include_directories(mlp PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(mlp PROPERTIES CUDA_ARCHITECTURES ${CC})
set_target_properties(mlp PROPERTIES CXX_STANDARD 17)

if (MSVC)
   target_compile_options(mlp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:/O2> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:/Od>)
else()
   target_compile_options(mlp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-O3> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-O0>)
endif()