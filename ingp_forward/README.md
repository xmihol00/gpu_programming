# Assignment 03 of GPU Programming SS 2023

In the root directory you can find a CMake (version 3.16 or newer) script to generate a
build environment. A modern C++ toolchain is required to compile the framework.
## Setup
* Clone repository to directory of your choice
* Create build folder `build`, change to build folder and call `cmake .. -DCC=*`
(replace * with the compute capability of your GPU, e.g. `61`, `75`, `86`, `89`, `90`  etc., alternatively use GUI to setup)
* Build (Visual Studio or `make`)
* Run: `./mlp <test_folder> [runs] [runCPU] [number_inputs]"`
## Supported toolchains:
* gcc 9.x or newer depending on your Linux distribution
* Visual Studio 2019 or 2022 on Windows
## Launch parameters:
* ` <test_folder>` : Pass a test folder to the executeable. e.g. tests/5x5.


You also need to install the CUDA toolkit on your machine. In order to profile and debug GPU code, we recommend NVIDIA NSight Compute. The exercise requires an NVIDIA GPU with compute capability 3.5 (Kepler) or better. If you don’t have a graphics card fulfilling these requirements, contact the lecturer and you will be provided with a card for this term. If you are experiencing build problems, make sure the compiler you are using matches one of the versions recommended above, you have installed the NVIDIA CUDA toolkit and CMake is able to find the corresponding CUDA libraries and headers. You can change the build to different compute capabilities by toggling the CMake options `CCxx` using the CMake GUI, the CMakeCache file, or command line arguments to CMake.

## Helpful Links
* [CMake](http://www.cmake.org/)
* [CUDA Installation on Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* [CUDA NSight Compute](https://developer.nvidia.com/nsight-compute)


##Bugs
If you encounter any bugs in the framework please do share them with us([Markus Steinberger](mailto:steinberger@icg.tugraz.at?subject=[Ass03]%20Bug%20Report)), such that we can adapt the framework.
