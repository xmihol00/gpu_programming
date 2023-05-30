# Assignment 3 — Fully-Connected Neural Network Inference Optimization Challenge
This document presents the optimization steps taken for a fully-connected neural network (NN) inference optimization challenge. Each section describes an optimization idea and the corresponding speed up achieved. The performance of the optimized algorithms was measured on the following devices:
* **GPU1** - GeForce GTX 650 (3.0 compute capability)
* **GPU2** - GeForce RTX 3080Ti (8.6 compute capability)

## Baseline - `matMulKernel` with `reluKernel`
The baseline implementation computes the outputs of the NN using separate mathematical operations applied one after another. The operations are matrix multiplication and element-wise ReLU activation. This results in to 5 subsequent kernel launches, two for each hidden layer and one for the output layer, which does not have an activation function.

The measured times, applies to all summaries bellow as well, do not include transfer of the signal values to the GPU. The performance is summarized in the following table:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         | 67.776         | 1.0                           | 0.392          |  1.0                          |
| 100x100     |                |                               | 57.948         |  1.0                          |

## Fused Global Memory - `fusedGlobalMemoryKernel`
The fused global memory implementation performs the computation of an output of the NN with a single kernel while storing the intermediate results after each layer in a global memory. Meaning that each thread performs a dot products of to it assign row and column followed by an activation function for each layer of the NN with the exception of the output layer. The output layer is only computed by each 16th thread, as the number of outputs is 16 times smaller than the number of intermediate results.

This implementation should perform better, since there are less memory reads and writes as well as kernel launches, but it does not. The reason is most likely the memory access pattern. Threads in the baseline implementation are launched in a 2D grid for the `matMulKernel` kernel, which results in that threads in each thread block access 16 different columns of the input matrix and 16 different rows of the layer 0 and 1 weight matrices, while the `fusedGlobalMemoryKernel` kernel is launched using 1D grid in a way that threads in each thread block access just 4 columns of the input matrix but all 64 rows of the layer 0 and 1 weight matrices. On a thread warp level the `matMulKernel` kernel accesses 16 columns and 2 rows, while the `fusedGlobalMemoryKernel` kernel accesses 1 column and 32 rows. Although both load the same amount of data, the former shares more data across warps, i.e. all the columns, while the later always differs in the read column or rows or both.

The performance of the fused global memory implementation is summarized by the table below:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         | 108.209        | 0.599                         |  0.501         | 0.782                         |
| 100x100     |                |                               | 79.790         | 0.726                         |

## Fused Shared Memory - `fusedSharedMemKernel`
The fused shared memory implementation stores the intermediate results as well as the weights inside shared memory. Storing the intermediate results reduces the global memory interactions just to a read of the input values and write of the output values while storing the weights inside shared memory also reduces the number of reads by a factor of 4 since one thread block computes 4 columns.

This implementation finally improves upon the baseline implementation and is summarized with the following table:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         |  16.327        | 4.151                         |  0.188         | 2.085                         |
| 100x100     |                |                               | 28.650         | 2.023                         |

## Whole NN In Shared Memory - `wholeNetInSharedMemKernel`
This is the first implementation that stores the whole NN in a shared memory at once and each thread runs for multiple iterations computing output values of multiple columns. Increasing the workload per a thread reduces the overall number of threads needed to be launched, therefore the number of times the weights must be loaded from global memory.

Nevertheless, this implementation shows only a very minor improvement on the GPU2 and quite significantly worse performance on the GPU1. This is most likely caused by a small occupancy of the streaming multiprocessors (SMs) since the weights and the intermediate results take up too much space in the shared memory. The performance is summarized in the following table:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         |  26.635        | 2.544                         |  0.182         | 2.154                         |
| 100x100     |                |                               | 24.304         | 2.384                         |

## NN In Shared Memory And Registers - `netInSharedMemAndRegsKernel`
Saving some of the weights into shared memory, i.e. the first and last layer, and the rest in registers, i.e. the second layer, addresses the issue with occupancy of the SMs from the previous implementation, while also improving the performance of the second layer. This implementation reaches more significant improvements this time on both GPUs as summarized in the table below:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         |  12.546        | 5.402                         |  0.134         | 2.925                         |
| 100x100     |                |                               | 11.959         | 4.846                         |

## Last Layer Co-Operation - `lastLayerCoopKernel`
Until now all of the implementations apart from the baseline one used only a fraction of the threads, i.e. first 4 in each thread warp, to compute the outputs of the last layer. This implementation enables all threads to take part in the computation of the outputs, each thread computing a partial result, i.e. one sixteenth of an output value. The final output is obtained with a tree sum reduction using the `__shfl_xor_sync` instruction. This optimization delivers only a minor improvement summarized with the following table:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         |  12.322        | 5.500                         |  0.128         | 3.063                         |
| 100x100     |                |                               | 11.554         | 5.015                         |

## Whole NN In Registers - `wholeNetInRegsKernel`
Finally, this implementation stores the whole NN only inside registers. It is not as straight forward as it may sound, since to reach a decent occupancy of the SMs, it is not possible to store a whole row of weights per thread for each layer. Each thread only stores half of the weights for layers 0 and 1 and 2 weights for layer 2. Meaning that two threads now must co-operate in computation of the intermediate results of hidden layers and the whole thread warp co-operates in computing a single output value, again by using the tree sum reduction with the `__shfl_xor_sync` instruction. This reduces the number of performed multiply-adds by each thread in an iteration by half, but on the other hand increases the total number of iterations by a factor of 2. It is still a good tradeoff resulting in a quite significant performance gain shown in the table below:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         | 10.819         | 6.264                         | 0.0923         |  4.247                        |
| 100x100     |                |                               | 8.632          |  6.713                        |

## Whole NN In Registers Half Precision - `wholeNetInRegsHalfKernel` (<span style="color:red">max threshold 0.15</span>)
Replacing the single float precisions inputs as well as the weights with half float precision is a quite straight forward modification to the previous implementation resulting in another quite significant improvement in performance. Although, it must be mentioned that the results can differ to the CPU single precision benchmark as much as 0.15 in a scaled absolute difference. Another thing to be mentioned is that the GPU1 does not support half precision arithmetics and the operands must be converted to single precision and back with each arithmetic operation. Nevertheless, both GPUs perform better running this implementation as summarized with the following table:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         | 9.338          | 7.258                         | 0.0595         |  6.588                        |
| 100x100     |                |                               | 6.029          |  9.612                        |

## Reduced Bank Conflicts - `reducedBankConflictsKernel` (<span style="color:red">max threshold 0.15</span>)
The previous two implementations computed outputs of 2 columns of the input matrix per iteration per thread block. This implementation computes 32 columns per iteration per thread block. The previous implementations resulted in a 16-way bank conflict when reading the input values, i.e. each even thread in a thread warp accesses the upper half of the column and each odd thread the lower part. Increasing the number of columns computed at once enables to schedule the threads in a circular manner, where threads 0 and 1 access the first column, thread 2 and 3 access the second column, ..., thread 30 and 31 access the 16th colum in the first iteration, then threads 0 and 1 access the second column, thread 2 and 3 access the third column, ..., thread 30 and 31 access the first colum in the second iteration, which should result in no bank conflict at all. 32 columns are needed to be able to run up to 8 warps, i.e. 256 threads, per a thread block.

I cannot provide the numbers of bank conflicts neither prior nor post this optimization since I do not have access to any profiling tools on neither of the GPUs, but it seems that at least some bank conflicts were indeed eliminated judging by the performance gain achieved on the GPU2 (performance of the GPU1 cannot really be taken seriously due to the single/half precision conversions described above) summarized in the table below:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         | 9.839          | 6.889                         | 0.0488         |   8.033                       |
| 100x100     |                |                               | 4.940          |  11.730                       |

## Coalesced Read Of Weights - `coalescedWeightsReadsKernel` (<span style="color:red">max threshold 0.15</span>)
The weights are stored by default in a row-major format, which is usually the preferred format since weights in the same row can be accessed continuously. But this is not the case on a GPU, for which it is faster to transpose the weight matrix on the CPU and load the weights into the registers with a stride equal to a multiply of 32, i.e. 64 in this case. In this implementation thread 0 starts by accessing the 0th index, thread 1 accesses the 1st index, thread 2 the 2nd index, ..., in the first iteration. Then they move to the 64th, 65th, 66th, ..., indices respectively in the second iteration, which results in a coalesced global memory access pattern further improving the performance summarized with the following table:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         | 7.187          | 9.430                         | 0.038          |  10.315                       |
| 100x100     |                |                               | 4.637          |  12.496                       |

## Hiding The Loading Of New Inputs - `inputsLoadHidingKernel` (<span style="color:red">max threshold 0.15</span>)
The idea of this implementation is to compute the outputs of the last layer with half of the threads/thread warps in a thread block and devote the other half to load the inputs for the next iteration. This optimization is only applicable for the 100x100 test case. The 5x5 test case with only 12 800 columns using the current setting of 32 inferred columns per layer per thread block with a launch of 400 thread blocks is processed using the described inference loop just once ($400 * 32 = 12 800$). 

Nevertheless, quite surprisingly, the implementation performs better for the 5x5 test case and worse for the 100x100 test case on the GPU2. The results can be explained by that more workload per thread and less synchronization caused by the tree sum reduction improves the performance for the former test case and that the HW is better at hiding the loads of the new inputs across more thread blocks than a dedicated implementation for the latter test case. The following table summarizes the performance of this implementation:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         | 7.034          | 9.635                         | 0.0354         |  11.073                       |
| 100x100     |                |                               | 4.914          |  11.792                       |

## Improving Occupancy Of SMs - `betterOccupancyKernel` (<span style="color:red">max threshold 0.15</span>)
All the implementations using registers to store the weights of the network used at least 70 registers so far (36 for the first layer, 32 for the second layer and 2 or more for the output layer). According to this https://xmartlabs.github.io/cuda-calculator/ CUDA occupancy calculator the occupancy of each SM for the GPU2 is 0.5 for 256 threads launched in a thread block with a use of up to 80 registers (the occupancy for the GPU1 according to the calculator is 0, which seems a bit odd). The number of registers is the limiting factor, therefore this implementation focuses on reducing it.

Each thread now stores 18, 16 and 16 weights of the first, second and output layers respectively, which is in all cases a one fourth of a row. In total each thread uses at least 50 registers, but definitely no more than 64, which is now the threshold to achieve 0.667 occupancy of the SMs on GPU2. The trade of is that now 4 threads must co-operate in computing a single output value of each layer, which reduces the number of outputs per one inference loop and consequently does not bring a performance increase on the GPU2. On the other hand, there is a quite significant performance increase on the GPU1, where the increased occupancy, now after the decrease in the amount of used registers and shared memory, is according to the calculator 0.625, which probably allows to hide the global memory transfers better as the older HW might be slower in that regard. The performance of this implementation is again available in the table below:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         | 6.374          | 10.633                        | 0.0471         |  8,323                        |
| 100x100     |                |                               | 5.226          |  11.088                       |

## Best Implementation For GPU2 - `best5x5Kernel` and `best100x100Kernel` (<span style="color:red">max threshold 0.15</span>)
This implementation is mostly based on the `coalescedWeightsReadsKernel` implementation with some modifications for each test case learned by the implementations, which followed after it. 

As mentioned above, the 5x5 test case requires only one inference loop per thread block, which allows to reduce the number of needed registers for the weights to 36, i.e. half of the number of weight in a row of the first layer, by reusing the registers for the weights of the second and output layers. Meaning 32 intermediate outputs is computed and stored into shared memory for the first layer, the weights of the second layer are loaded into the registers, the intermediate outputs of the second layer are computed etc. This enables higher occupancy of the SMs while keeping the same number of outputs per an inference loop as well as the synchronization between threads.

The final optimization for the 100x100 test case just slightly modifies the `coalescedWeightsReadsKernel` implementation. The modification lies in the number of registers allocated for the output layer per thread. The best number turned out to be 8, 4 times more than the previous implementation, leading at least to 76 registers allocated per thread. This allows to compute the output of the last layer using less iterations and synchronization between the co-operating threads. The occupancy is probably still preserved at 0.5 given the better performance.

The best so far measured performance across multiple separate launches on both the GPU2 is summarized with the following table:
| Test case   | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|
| 5x5         | 0.0343         |  11.428                       |
| 100x100     | 3.932          |  14.738                       |

# Summary
As can be seen above, using an algorithm specifically designed to store the given NN efficiently brings a great performance increase. Another performance gain can be obtained by a use of half precision floating point numbers with a trade-off in accuracy in comparison to the benchmark, although during inference on never before seen data the single precision implementation could be more inaccurate than the half precision as simpler models should generalize better. Additionally, significant performance gain can be achieved by having a clear idea of how the HW works and further optimizing for it, i.e. performing coalesced memory access, reducing bank conflicts and increasing SM occupancy.

It would be interesting to look at the profiling data and see if there are some other possible ares of improvement or what is slowing the computation down and how it could be addressed.
