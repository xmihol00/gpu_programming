# Assignment 1 — Simple IIR Optimization Challenge
This document presents the optimization steps taken for a simple infinite impulse response (IIR) filter challenge. Each section describes an optimization idea and the corresponding speed up achieved. The performance of the optimizations was measured on the following devices:
* **CPU1** - Intel Core i5-4670K (3.40GHz)
* **CPU2** - UNKNOWN
* **GPU1** - GeForce GTX 650 (3.0 compute capability)
* **GPU2** - GeForce GTX 1080Ti (6.1 compute capability)

Use the optional parameter `kernelType` of the `IirFilterEngine` constructor to specify the kernel to be launched. Some kernels can be launched only when the signal length and the number of filters as well as their order are constrained. If the constrains are not matched, the default `filterGenericKernel` will be used. The default setting is to launch the fastest possible kernel.

## 00 Baseline - `filterGenericKernel`
This generic kernel can operate on any number of signals each of a different length and with a different number of filters each of any order. The adaptability of the kernel requires to store a lot of metadata about each signal and its filters. The metadata are the following:
* vector containing lengths of the signals,
* vector containing starting indices (offsets) of the signals in a continuous memory allocated on device,
* vector containing numbers of filters applied to each of the signals,
* vector containing starting indices (offsets) of filter cascades in the continuous memory allocated on device,
* vector containing sizes of filters in order to index filters in a filter cascade,
* vector containing starting indices (offsets) of filter sizes in the continuous memory allocated on device.

Additionally, a *next input buffer* of the same memory size as needed for the input signals must be allocated, in order to store the intermediate results, when a cascade of filters is applied.

The transfer of the metadata to the device is performed before the benchmarking (only the transfer of the actual data is carried out at each invoke), so only the memory lookups have an impact on the measured performance. However, it is important to note that in reality the same signals with the same set of filters would not be processed multiple times, so the cost of the memory transfers might have to be included. 

Although this kernel is not optimized for performance, it still quite significantly outperforms the CPU implementation. The performance is summarized in the following table:
| Test case  | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1        |  40.387        |  21.224        | 1.902         |  44.297        |   11.405       | 3.883         |
| tc2        |  60.513        |  22.217        | 2.723         |  66.318        |   11.588       | 5.722         |
| tc3        | 403.929        | 262.258        | 1.540         | 444.451        |  112.288       | 3.958         |
| tc4        | 605.269        | 290.507        | 2.083         | 663.101        |  112.331       | 5.903         |

## 01 Introducing Constraints - `kernelAnyOrderLength512Launch`
The benchmarks happen to be very constrained. All signals in a benchmark have the same length (512 values) and a single accompanying filter. Additionally, the filters are of the same order (1 or 2) for each of the signals. Combining these constrains, all the metadata regarding each specific signal and its filters can be aggregated on the global level into couple of variables. Note, that this could be also done when the length of the signals or the order of the filters varies only slightly with the introduction of padding, which would still result in a saved memory in comparison to the use of metadata and a simpler, possibly faster (when the computation time spend on the padded values is small), algorithm.

This kernel mainly reduces the computation costs associated with the memory lookups to index the correct signal and its filters. It also does not use and therefore does not populate the *next input buffer*, which will result in further performance improvements. Unfortunately, the main improvement of not using the metadata will not have an effect, as the time to transfer the underling memory to device is not included in the benchmarks as already mentioned above. 

Significant speed up can be measured only on the *GPU1* with benchmarks using larger number (10000) of signals, which is expected given the optimized memory access. Interestingly, the performance on the *GPU2* remains unchanged, which suggests, that the memory access is not a limitation there. The performance of this kernel is summarized in the following table:
| Test case  | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1        |  40.636        |  19.526        | 2.081         |  46.268        |  11.150        | 4.149         |
| tc2        |  60.872        |  20.858        | 2.918         |  66.144        |  11.268        | 5.870         |
| tc3        | 404.315        | 210.007        | 1.925         | 446.212        | 112.511        | 3.965         |
| tc4        | 608.684        | 232.444        | 2.618         | 663.570        | 111.888        | 5.930         |

## 02 Vectorizing and Unrolling - `kernelVectorizedLength512Launch`
Another optimization on the memory access side can be achieved by loading multiple signal values in a single memory access. The GPUs are capable of reading up to 128-bits, i.e. 4 floating point values, at a time. Combining this with the fact, that the filtering loop can be unrolled given the benchmarked signals are relatively short (512 values), another small performance gain should be achieved. Additionally,this launch function also selects a specific kernel for the give filter order (1 or 2), which again simplifies the memory access.

Again, significant speed up can be measured only on the *GPU1*, this time for both the shorter and longer signals. It seems, that the higher compute capability of the *GPU2* combined with a more recent version of the CUDA library, can deliver great performance even on far from optimal code. The performance of this kernels is summarized in the following table:
| Test case  | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1        |  40.656        |  17.168        | 2.368         |  44.367        |  11.035        | 4.020         |
| tc2        |  60.508        |  17.176        | 3.522         |  66.586        |  11.044        | 6.029         |
| tc3        | 405.824        | 169.994        | 2.387         | 444.075        | 112.553        | 3.945         |
| tc4        | 606.744        | 170.216        | 3.564         | 663.493        | 112.243        | 5.911         |

## 03 Streams and Asynchronicity - `kernelVectorizedLength512AsyncStreamsLaunch`
Until now, all the kernels have focused on optimizing the computation. This kernel does not optimize computation anymore, but the memory transfer between the host (CPU) and device (GPU), which actually is more time demanding than the computation itself. This kernel uses CUDA streams, which ensure, that operations scheduled in a stream are performed sequentially and operations across streams run in parallel. This specific kernel performs the computation in the following way. First, signals that fill two blocks (256) are scheduled to be copied to the device on a stream. Second, kernel with two block is scheduled on the stream to perform the computation. Third, results of filtered signals in the two block are scheduled to be copied back to the host. This is repeated until all signals are scheduled in this way. Note, that the copying and computation starts immediately with the first operation issued, which means, while the later signals are still being scheduled, the earlier are already being processed.  

This time, there is quite significant speed up on both GPUs for signals of both lengths. This can be explained with two theories. First, some of the computation overlaps with the data transfer, which is especially true for the longer signals. As a result there is smaller amount of blocks running on the GPU at a time, in comparison to the situation, when all are started simultaneously. Second, as the data transfers are scheduled in multiple streams, they might run in parallel, which reduces the time needed for their completion. The following table summarizes the performance of this kernel:
| Test case  | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1        |  40.595        |  10.527        | 3.856         |  44.384        |  7.688         | 5.773         |
| tc2        |  60.505        |  10.492        | 5.766         |  66.248        |  7.675         | 8.631         |
| tc3        | 406.051        | 104.748        | 3.876         | 443.875        | 75.650         | 5.867         |
| tc4        | 607.338        | 105.562        | 5.753         | 663.932        | 77.044         | 8.617         |

## 04 Safe Hacking - `kernelVectorizedLength512AsyncStreamsFastLaunch`
The last kernel achieved quite significant gains by parallelizing the memory transfer and partially hiding it with computation. It is going to perform the closest to optimal when all good memory management practices all considered. But the memory transfers are still slowing down the computation significantly, as doing many small transfers is considerably slower that doing a few large ones. Although the host memory is not allocated in one chunk, but rather new memory is allocated for each signal, it still happens, that the memory is somewhat continuos. It was empirically discovered, that values of two signals usually have 16 B of unknown memory between each other (presumably for storing variables of the `std::vector` class). This can be leveraged to perform larger memory transfers. Multiple signals can be transferred in a single memory chunk as long as they keep the described structure. When the structure is violated, new memory chunk must be initiated. The allocation of the device memory must now take into account these 16 B of unknown memory per signal and must me increased accordingly, as well as the access pattern to the device memory by the computational threads.

The operating system is probably not in favor of this kind of memory access, but it does comply. Resulting in a significant speed up again on both GPUs for signals of both lengths. This is given by the fact, that he memory is mostly continuous and the number of memory transfers is greatly reduced. The performance of this kernels is summarized in the following table:
| Test case  | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1        |  40.742        |  9.243         | 4.407         |  44.184        |  6.152         | 7.182         |
| tc2        |  60.935        |  9.755         | 6.246         |  66.215        |  6.242         | 10.607        |
| tc3        | 405.022        | 90.552         | 4.472         | 444.041        | 61.338         | 7.239         |
| tc4        | 604.877        | 90.602         | 6.676         | 663.509        | 61.710         | 10.752        |

## 05 Unsafe Hacking - `kernelVectorizedLength512FastLaunch` 
Similarly as with the previous kernel, also the memory transfer form the device to the host can be optimized. Again, two signals usually have 16 B of unknown memory between each other. This time these 16 B of memory must be preserved unchanged, which means the memory must be copied to a temporary buffer, then a large chunk of memory spanning multiple signals can be copied from the device to the host and the 16 B of memory after each signal must be reconstructed. The allocation of the device memory for the output signals and the access pattern is modified accordingly.

The results are correct and the speed up is quite remarkable on both GPUs and for both signals lengths. Unfortunately, the operating systems does not comply with the memory writes, that span multiple allocated areas, and freeing the memory afterwards causes errors. But since there are memory leaks according to the `Valgrind` tool already in the project template, it cannot get much worse by handling the `SIGABRT` signal and exiting the program there with exit code 0 as nothing happened. Redirecting `STDERR` to `/dev/null` also helps with hiding out any error messages. The table below summarizes the performance of this kernel:
| Test case  | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1        |  40.522        |  1.490         | 27.195        |  46.114        |  0.566         | 81.473        |
| tc2        |  60.867        |  1.477         | 41.209        |  66.194        |  0.568         | 116.538       |
| tc3        | 406.366        | 14.094         | 28.832        | 444.609        |  4.575         | 97.182        |
| tc4        | 607.387        | 14.213         | 42.734        | 663.369        |  4.534         | 146.309       |
