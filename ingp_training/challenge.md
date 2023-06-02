# Assignment 4 — Simplified Instant Neural Graphics Primitives Pipeline
This document presents the optimization steps taken for a simplified instant neural graphics primitives pipeline optimization challenge. Each section describes an optimization idea and the corresponding speed up achieved. The performance of the optimized algorithms was measured on the following devices:
* **GPU1** - GeForce GTX 650 (3.0 compute capability)
* **GPU2** - GeForce RTX 3080Ti (8.6 compute capability)

## Baseline - `rayGenerationKernel`, `sampleGenerationKernel`, `frequencyEncodingKernel`, `positionEncoderKernel`, `networkKernel` and `accumulationKernel`
The baseline implementation runs the pipeline using multiple kernels each implementing a specific step of it. The intermediate results computed by each kernel are stored into a global memory. Meaning that all the kernels must be launched in multiple successive iterations, in order to compute larger images, as the intermediate results require far more memory than the final results. Furthermore, the `rayGenerationKernel` and `accumulationKernel` kernels are launched ony with the same number of threads as there are pixels in an image, while the other kernels are launched with a number of thread blocks equaling the number of the pixels. Each thread block containing 512 threads, as 512 samples are generated for each ray.

The measured times, which applies to all summaries bellow as well, include a transfer of the camera settings to the GPU, but do not include transfers of the embeddings, offsets and AABBs. The performance of the baseline implementation is summarized in the following table:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         |    38.227      | 1.0                           |                |                               |
| 10x10       |   139.779      | 1.0                           |     3.526      |  1.0                          |
| 100x100     | 13719.8        | 1.0                           |                |                               |
| 800x800     |                |                               | 30427.3        |  1.0                          |

## All In One - `allInOneKernel`
The all in one implementation merges all the kernels from the baseline implementation into a single kernel. In this case the ray generation as well as the accumulation of the final results is executed by the same number of threads as in the other steps, i.e. number of pixels in an image multiplied by 512. The generation is computed by all threads, basically each thread in a kernel computes exactly the same intermediate results, while the accumulation is always computed only by the first thread of a thread block. Again, the intermediate results are stored into a global memory, but the compute iterations are moved to the kernel itself, so there is only a single kernel launch.

The all in one implementation delivers respectable speedups on both of the GPUs as summarized by the table below:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         |   21.354       | 1.790                         |                |                               |
| 10x10       |   70.429       | 1.985                         |    2.515       |  1.402                        |
| 100x100     | 6108.17        | 2.246                         |                |                               |
| 800x800     |                |                               | 7623.42        |  3.991                        |

## Frequency Encoding Co-operation - `frequencyEncodingCoopKernel`
Since the frequency encoding of the samples is same for all sample within one thread block, it is reasonable to distribute the computation of each encoded value to a single thread. Meaning that now each of the first 39 threads in a thread block computes a single encoded value. Additionally, a whole thread block can then co-operate and compute biases for the input layer of the network. The biases are just pre-computed values resulting from a matrix (the corresponding weights of the input layer) by vector (the frequency encodings) multiplication. This implementation, and the following implementations until specified otherwise, also moves from 512 threads per thread block to 256 threads per thread block. Meaning that each thread now performs 2 iterations to compute a single output.

This optimization leads to a quite significant performance increase on the GPU2, while on the GPU1 it has an opposite effect and the performance is worsen. An explaining reason might be a suboptimal implementation of this concept, but which can still perform better on the newer HW. The performance is summarized in the following table:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         |   25.845       | 1.479                         |                |                               |
| 10x10       |   96.042       | 1.455                         |     0.525      |  6.716                        |
| 100x100     | 9497.34        | 1.445                         |                |                               |
| 800x800     |                |                               |  1955.8        | 15.557                        |

## Half Precision - `halfPrecisionKernel`
The next optimization step, as oppose to position encoding co-operation, is the switch to half precision floating point numbers for the network evaluation as well as the encoding of the inputs. It must be taken into account that the older GPU1 does not support native arithmetics in half precision and that it must be simulated by converting from half precision to single precision and back with each instruction. 

Nevertheless, it is only the GPU1 where a performance improvement was measured as summarized in the table below:
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| 5x5         |   25.078       | 1.524                         |                |                               |
| 10x10       |   93.366       | 1.497                         |     0.655      |  5.383                        |
| 100x100     | 9258.41        | 1.482                         |                |                               |
| 800x800     |                |                               | 2209.69        | 13.770                        |

## Position Encoding Co-operation - `positionEncodingCoopKernel`
Finally, this implementation focuses on optimizing the computation of the position encodings. Each thread computes just 4 of the encoded values, resulting in 32 encoded samples per iteration, which is the exact number of samples the network needs to perform the best. Nevertheless, this implementation is only more of a prove of a concept, because the evaluation is still run in subsequent iterations for all the 512 samples.

Still, this optimization lead to a very significant performance increase on the GPU1 and less significant improvement on the GPU2. Both improvements are likely caused by the reduction in needed shared memory resulting in better occupancy of the streaming multiprocessors. The gained improvements are shown in the below:
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
| 5x5         |    2.991       | 12.781                        |                |                               |
| 10x10       |   32.472       |  4.305                        |    0.511       |  6.900                        |
| 100x100     | 2940.96        |  4.665                        |                |                               |
| 800x800     |                |                               | 1951.42        | 15.592                        |

## Early Stopping - `earlyStoppingKernel`
The obvious step after the previous optimization is to add early stopping to the pipeline. This optimization ensures, that only as many batches of 32 samples as necessary are evaluated. Meaning that when the alpha value gets close to zero, the rest of the samples is not computed anymore. Additionally, this implementation also introduces more co-operation between the threads at several places of the pipeline, e.g the ray generation, computation of the near and far plane or the accumulation of the final results.

This optimization should lead mainly to improvements on the larger images, as there the thread blocks do more iterations, but as can be seen in the table below, there is also a reasonable improvement for the smaller images.
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
| 5x5         |    2.581       | 14.811                        |                |                               |
| 10x10       |   29.960       |  4.666                        |    0.413       |  8.517                        |
| 100x100     | 2792.15        |  4.914                        |                |                               |
| 800x800     |                |                               | 1649.91        | 18.442                        |

## Less Threads - `lessThreadsKernel`
This implementation builds up on the idea of early stopping. Where the previous implementation used 256 threads per thread block and computed batches of 32 samples, this implementation uses only 128 threads per block and computes batches of 16 samples. This more granular approach can potentially stop the pipeline sooner, as the test of the alpha value happens more often.

As always, the performance of this optimization is summarized by the following table:
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
| 5x5         |    2.284       | 16.737                        |                |                               |
| 10x10       |   26.181       |  5.339                        |    0.451       |  7.818                        |
| 100x100     | 2474.31        |  5.545                        |                |                               |
| 800x800     |                |                               | 1212.66        | 25.091                        |

## Pixel Pool - `pixelPoolKernel`
The pixel pool implementation introduces a pixel counter in global memory. This counter is used to assign pixels to be processed by each thread block. The idea behind it is that there can be thread blocks processing pixels, which are finished in just a few of the pipeline iterations, while, on the other hand, there can be thread blocks processing pixels, which require all of the iterations of the pipeline. This optimization allows to schedule the work across the thread blocks more evenly.

This optimization has no effect on the smaller images, where only a single pixel is processed by a thread block. The performance increase on the larger images is not negligible as shown in the table below:
|-------------|----------------|-------------------------------|----------------|-------------------------------|
| Test case   | GPU1 time [ms] | speed up relative to baseline | GPU2 time [ms] | speed up relative to baseline |
| 5x5         |    2.245       | 17.028                        |                |                               |
| 10x10       |   26.682       |  5.239                        |    0.435       |  8.106                        |
| 100x100     | 2405.87        |  5.703                        |                |                               |
| 800x800     |                |                               | 1020.51        | 29.816                        |

# Summary
As with the previous assignments, adjusting an algorithm to a specific problem can bring great performance benefits. In this case, the largest performance increase on the GPU1 was measured on the smallest image of just slightly above 17 times the baseline performance. On the other hand, the algorithm on the GPU2 performs better in relation to the baseline implementation for the largest image with almost 30 fold performance increase.
