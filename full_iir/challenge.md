# Assignment 2 — Full IIR Optimization Challenge
This document presents the optimization steps taken for a full infinite impulse response (IIR) filter challenge. Each section describes an optimization idea and the corresponding speed up achieved. Apart from the baseline implementation all of the other optimizations are tailored for specific filters and signal lengths. Therefore, they can be use only for specific test cases. Unexpected test cases will default to the baseline implementation if they do not satisfy the conditions of any of the optimizations. The performance of the optimizations was measured on the following devices:
* **CPU1** - Intel Core i5-4670K (3.40GHz)
* **CPU2** - UNKNOWN
* **GPU1** - GeForce GTX 650 (3.0 compute capability)
* **GPU2** - GeForce GTX 1080Ti (6.1 compute capability)

## All Test Cases Baseline - `kernelGenericLaunchWithDevicePointers`
This generic kernel can operate on any number of signals each of a different length and with a different number of filters each of any order. The adaptability of the kernel requires to store a lot of metadata about each signal and its filters. The metadata are the following:
* vector containing lengths of the signals,
* vector containing starting indices (offsets) of the signals in a continuous memory allocated on device,
* vector containing numbers of filters applied to each of the signals,
* vector containing starting indices (offsets) of filter cascades in the continuous memory allocated on device,
* vector containing sizes of filters in order to index filters in a filter cascade,
* vector containing starting indices (offsets) of filter sizes in the continuous memory allocated on device.

Additionally, a *next input buffer* of the same memory size as needed for the input signals must be allocated, in order to store the intermediate results, when a cascade of filters is applied.

The measured times do not include transfer of the signal values to the GPU. Although this kernel is not optimized for performance, it still in the majority of cases outperforms the CPU implementation. The performance is summarized in the following table:
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1         |   40.570       |    5.280       |   7.684       |   44.373       |   0.374        |  118.644      |
| tc2         |   60.736       |    6.972       |   8.711       |   66.430       |   0.464        |  143.168      |
| tc3         |  404.258       |   67.826       |   5.960       |  445.024       |   0.432        | 1030.148      |
| tc4         |  605.263       |   93.366       |   6.483       |  667.139       |   0.507        | 1315.856      |
| tc5         |  662.128       | 2960.960       |   0.224       |  729.353       | 410.716        |    1.776      |
| tc6         | 1029.570       | 3836.400       |   0.268       | 1092.540       | 494.775        |    2.208      |
| tc7         |   78.950       |   11.702       |   6.747       |   86.298       |   0.703        |  122.757      |
| bonus_tc8   |  163.300       |   32.394       |   5.041       |  178.575       |   0.386        |  462.630      |
| bonus_tc9   |  252.956       |   44.741       |   5.654       |  266.219       |   0.472        |  564.023      |
| bonus_tc10  |  326.322       |   62.690       |   5.205       |  356.396       |   1.589        |  224.289      |
| bonus_tc11  |  489.513       |   91.953       |   5.324       |  533.350       |   1.943        |  274.498      |
| bonus_tc12  |  651.934       |  119.078       |   5.475       |  712.649       |   7.360        |   96.827      |
| bonus_tc13  |  978.572       |  153.083       |   6.392       | 1067.190       |   8.682        |  122.920      |

## TC1 and TC3 - `kernelStateSpaceMatrixOrder1ParallelLaunch`
This kernel leverages the limited precision of floating point numbers (7 decimal places) and the properties of the filters in tc1 and tc3, which cause a rapid decrease in significance of the already filtered values on the future filtered values. The values of the filter when converted to the state space representation are zero in terms of floating point precision only after 9 and 12 time steps for the respective test cases.

It means that the state can be computed at any time step of the signal and consequently be used for a state space matrix filtering. Additionally, given the sparse state space matrix, each filtered value can be computed with just 9 and 12 multiply-adds respectively. The size of the state space matrix was chosen to 32 by 32 to ensure that synchronization of the state remains only across a single thread warp. The best performance was measured when using 4 thread warps per signal for tc1 and 2 thread warps per signal for tc3, which means that each thread computes 4 or 5 and 8 or 9 output values respectively. The performance of this kernel is summarized in the following table:
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1         |   42.195       |  0.345         |  122.304      |   44.261       | 0.011          | 4023.727      |
| tc3         |  413.195       |  3.606         |  114.585      |  444.552       | 0.075          | 5927.360      |

## TC2 and TC4 - `kernelStateSpaceMatrixOrder2Launch`
Unfortunately, the filters in tc2 and tc4 do not have the same properties as the filters in tc1 and tc3. Their converted values to state space representation do not approach zero, therefore the optimization described above is not possible. In this case, each of the input signals are assigned with a single thread warp, which must filter the signal in a sequential manner. Nevertheless, filtering using a slightly modified state space matrix is very fast. The modification lies in moving of the first 2 state rows to the bottom of the matrix as well as moving of the first 2 columns as the last 2 columns. This small modification of the state space matrix simplifies its indexing and is used also for other test cases. Summarization of the performance is shown in the following table:
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc2         |   62.825       |  0.568         |  110.607      |   66.179       | 0.015          | 4411.933      |
| tc4         |  607.102       |  5.418         |  112.053      |  663.807       | 0.091          | 7294.582      |

## TC5 - `kernelStateSpaceMatrixOrder1LongSignalLaunch`
This kernel is very similar to the kernel used for tc1 and tc3. Again, the values of the filter when converted to the state space representation are zero in terms of floating point precision only after 9 time steps. The main difference is that each thread warp is loading only a specific part of the signal to shared memory, while in the kernel for tc1 and tc3 the whole thread block cooperates in the population of the shared memory. Therefore, the kernel for tc5 requires only synchronization of threads inside a warp. The performance of this kernel is summarized in the following table:
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc5         |  669.992       |  5.693         |  117.687      |  727.868       | 0.113          | 6441.310      |

## TC6 - `kernelFiniteInfiniteFilterLongSignalLaunch`
The values of the filter when converted to the state space representation are zero in terms of floating point precision after 58 time steps. 58 values is too many for an effective filtering using the state space matrix. Therefore, each thread of this kernel first computes 2 output values using the 58 state space filter values for certain time steps of the signal and then proceeds the filtering sequentially using the original filter values and the computed time steps as the last output and second to last output.

The best performance was measured when each thread filters 8 output values and is captured in the table below:
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc6         | 1017.760       |  3.599         |  282.780      | 1088.720       | 0.120          | 9072.667      |

## TC7 - `kernelDifferentSignalsStateSpaceMatrixLaunch`
The kernel for tc7 uses the full 32 by 32 state space matrix for filtering. The main was optimization was achieved by ordering the signals in a way so that each thread block filters signals of the same length. Additionally, each thread warp filters one signal, which allows for synchronization only across warps when populating the shared memory. The performance of this kernel is summarized in the following table:
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc7         |   79.935       |  1.965         |   40.679      |   85.866       | 0.028          | 3066.643      |

## TC8, TC9, TC10 and TC11 - `kernelFiniteFilterBlockPerSignalLaunch`
This kernel leverages the relatively small length (256 and 512) of the filtered signals and the fact that matrix multiplication is a linear transformation and therefore multiple matrix multiplications can be pre-computed and simplified to a single matrix multiplication. It means that a state space matrix of size signal length by signal length without the state space rows and columns can be created for each of the filters in the cascade. These matrices can be then multiplied to obtain a single matrix later used for the actual filtering. Furthermore, only the row with the larges number of non-zero entries of the matrix must be transferred to the GPU as the values in the rows are repetitive. This pre-computation of the filter values is performed on the CPU and is very resource demanding, but it could be pre-computed once and then stored.

The lengths of the pre-computation filter values are 58, 63, 61 and 315 respectively for the bonus test cases. This kernel is designed such that each thread block filters one signal and each thread filters just a single output value. This allows a rather simple population of the shared memory and very small number of multiply-adds, equaling to the number of filter values, performed by each thread. Different kernel was also tried for the bonus_tc11, where each thread computed just one forth of the filtered value, which were then summed. This kernel decreased the number of multiply-adds per thread by the factor of 4, but also increased the number of needed threads by the same factor, which did not lead to any performance increase. The performance of the kernel mentioned in the section headline is captured in the table below:
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| bonus_tc8   |  163.180       |  2.550         |   63.992      |  178.053       | 0.073          | 2439.082      |
| bonus_tc9   |  248.383       |  2.576         |   96.422      |  265.360       | 0.079          | 3358.987      |
| bonus_tc10  |  328.132       |  2.548         |  128.780      |  355.262       | 0.089          | 3991.708      |
| bonus_tc11  |  505.508       | 10.738         |   47.077      |  531.407       | 0.328          | 1620.143      |

## TC12 and TC13 - `kernelFiniteFilterMoreBlocksPerSignalLaunch`
The kernel for the remaining bonus test cases is very similar to the previous one in how the filter values are pre-computed as well as the filtering itself. The only difference is in that a single signal is filtered by multiple thread blocks, as the number of threads required for the longer (1024 values) signals was not delivering good performance when scheduled in one thread block. This kernel limits the number of threads per a thread block to 256. The fact that each thread filters just a single value remains unchanged. The performance of this kernel is captured in the table below:
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| bonus_tc12  |  679.249       |  2.712         |  250.461      |  709.296       | 0.086          | 8247.628      |
| bonus_tc13  |  994.228       | 13.045         |   76.215      | 1061.310       | 0.391          | 2714.348      |

# Results
For better clarity and comparison of the achieved improvements, the tables with baseline performance and best performance for each test case are shown below again.

## Baseline Performance
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1         |   40.570       |    5.280       |   7.684       |   44.373       |   0.374        |  118.644      |
| tc2         |   60.736       |    6.972       |   8.711       |   66.430       |   0.464        |  143.168      |
| tc3         |  404.258       |   67.826       |   5.960       |  445.024       |   0.432        | 1030.148      |
| tc4         |  605.263       |   93.366       |   6.483       |  667.139       |   0.507        | 1315.856      |
| tc5         |  662.128       | 2960.960       |   0.224       |  729.353       | 410.716        |    1.776      |
| tc6         | 1029.570       | 3836.400       |   0.268       | 1092.540       | 494.775        |    2.208      |
| tc7         |   78.950       |   11.702       |   6.747       |   86.298       |   0.703        |  122.757      |
| bonus_tc8   |  163.300       |   32.394       |   5.041       |  178.575       |   0.386        |  462.630      |
| bonus_tc9   |  252.956       |   44.741       |   5.654       |  266.219       |   0.472        |  564.023      |
| bonus_tc10  |  326.322       |   62.690       |   5.205       |  356.396       |   1.589        |  224.289      |
| bonus_tc11  |  489.513       |   91.953       |   5.324       |  533.350       |   1.943        |  274.498      |
| bonus_tc12  |  651.934       |  119.078       |   5.475       |  712.649       |   7.360        |   96.827      |
| bonus_tc13  |  978.572       |  153.083       |   6.392       | 1067.190       |   8.682        |  122.920      |

## Best Performance
| Test case   | CPU1 time [ms] | GPU1 time [ms] | GPU1 speed up | CPU2 time [ms] | GPU2 time [ms] | GPU2 speed up |
|-------------|----------------|----------------|---------------|----------------|----------------|---------------|
| tc1         |   42.195       |  0.345         |  122.304      |   44.261       | 0.011          | 4023.727      |
| tc2         |   62.825       |  0.568         |  110.607      |   66.179       | 0.015          | 4411.933      |
| tc3         |  413.195       |  3.606         |  114.585      |  444.552       | 0.075          | 5927.360      |
| tc4         |  607.102       |  5.418         |  112.053      |  663.807       | 0.091          | 7294.582      |
| tc5         |  669.992       |  5.693         |  117.687      |  727.868       | 0.113          | 6441.310      |
| tc6         | 1017.760       |  3.599         |  282.780      | 1088.720       | 0.120          | 9072.667      |
| tc7         |   79.935       |  1.965         |   40.679      |   85.866       | 0.028          | 3066.643      |
| bonus_tc8   |  163.180       |  2.550         |   63.992      |  178.053       | 0.073          | 2439.082      |
| bonus_tc9   |  248.383       |  2.576         |   96.422      |  265.360       | 0.079          | 3358.987      |
| bonus_tc10  |  328.132       |  2.548         |  128.780      |  355.262       | 0.089          | 3991.708      |
| bonus_tc11  |  505.508       | 10.738         |   47.077      |  531.407       | 0.328          | 1620.143      |
| bonus_tc12  |  679.249       |  2.712         |  250.461      |  709.296       | 0.086          | 8247.628      |
| bonus_tc13  |  994.228       | 13.045         |   76.215      | 1061.310       | 0.391          | 2714.348      |
