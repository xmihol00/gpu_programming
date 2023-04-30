# Assignment 01 and 02 of GPU Programming SS 2023

In the root directory you can find a CMake (version 3.16 or newer) script to generate a
build environment. A modern C++ toolchain is required to compile the framework.
## Setup
* Clone repository to directory of your choice
* Create build folder `build`, change to build folder and call `cmake .. -DCC=*`
(replace * with the compute capability of your GPU, e.g. `61`, `75`, `86`, `89`, etc., alternatively use GUI to setup)
* Build (Visual Studio or `make`)
* Run: `./iir <test_file.json>`
## Supported toolchains:
* gcc 8.x, 9.x, 10.x depending on your Linux distribution
* Visual Studio 2019 or 2022 on Windows
## Launch parameters:
* `<test_file.json>` : Pass a test case to the program. tc1-tc4 are for Assignment 1, for Assignment 2 tc1-tc13 should be considered. Note that performance measurements for Assignment 2 will use `executeWithDevicePointers`, this is already activated for tc5-tc13 per default. To activate them  for tc1-tc4, change the parameter in the config file. Also note, that you need to provide the device memory allocation and pointer setup in `main.cpp` for these measurements to work (copy them over from your Assignment 1).

Note that you can freely alter the config files.
- `CPU` allows you to switch between different CPU implementations
- `GPU` whether to run the GPU version or not
- `runWithDevicePointers` execute with device pointers (like the measurement for Ass 02). Note that you have to implement the device data management yourself in `main.cpp`
- `runs` alters the number of runs to be averaged over
- `compare` compares the results again the CPU double precision ground truth
- `output` gives you the options to write out the signals (e.g., setting it to `"output/"`) you can then look at the signal with the provided python script. Note that writing outputs may take a significant amount of time.

You also need to install the CUDA toolkit on your machine. In order to profile and debug GPU code, we recommend NVIDIA NSight Compute. The exercise requires an NVIDIA GPU with compute capability 3.0 (Kepler) or better. If you don’t have a graphics card fulfilling these requirements, contact the lecturer and you will be provided with a card for this term. If you are experiencing build problems, make sure the compiler you are using matches one of the versions recommended above, you have installed the NVIDIA CUDA toolkit version 10.x or 11.x and CMake is able to find the corresponding CUDA libraries and headers. You can change the build to different compute capabilities by toggling the CMake options `CCxx` using the CMake GUI, the CMakeCache file, or command line arguments to CMake.

## Helpful Links
* [CMake](http://www.cmake.org/)
* [CUDA Installation on Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* [CUDA NSight Compute](https://developer.nvidia.com/nsight-compute)


##Bugs
If you encounter any bugs in the framework please do share them with us([Markus Steinberger](mailto:steinberger@icg.tugraz.at?subject=[Ass01_02]%20Bug%20Report)), such
that we can adapt the framework.
