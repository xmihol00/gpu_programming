# GPU programming
Optimizing code to run as fast as possible on a GPU. All credits goes to the **GPU programming** group at TU Graz, for providing the framework and all necessary files for compilation and testing. Code written by me is in the following files:
```
-- simple_iir/
    |
    |-- include/
    |    |
    |    |-- IirFilterEngine.cuh
    |    |
    |    |-- IirFilterEngine.h
    |
    |-- src/
    |    |
    |    |-- IirFilterEngine.cpp
    |    |
    |    |-- IirFilterEngine.cu
    |
    |-- challenge.md

-- full_iir/
    |
    |-- include/
    |    |
    |    |-- IirFilterEngine.cuh
    |    |
    |    |-- IirFilterEngine.h
    |
    |-- src/
    |    |
    |    |-- IirFilterEngine.cpp
    |    |
    |    |-- IirFilterEngine.cu
    |
    |-- challenge.md

-- ingp_forward/
    |
    |-- include/
    |    |
    |    |-- GpuMlpEngine.h
    |
    |-- src/
    |    |
    |    |-- GpuMlpEngine.cpp
    |    |
    |    |-- GpuMlpEngine.cu
    |
    |-- challenge.md
    |
    |-- csv_diff.py
    |
    |-- print_matrices.py

-- ingp_forward/
    |
    |-- include/
    |    |
    |    |-- GpuNGPEngine.h
    |
    |-- src/
    |    |
    |    |-- GpuNGPEngine.cpp
    |    |
    |    |-- GpuNGPEngine.cu
    |
    |-- challenge.md
    |
    |-- csv_diff.py
```

## Performance Challenge
The **GPU programming** group at TU Graz organizes a performance challenge, where students compete with their solutions. The performance tables are available below, my student ID is 12211951. See the `challenge.md` files in the respective directories for descriptions of the implemented optimizations. 

# Simple Infinite Impulse Response
Tied 1st place:
|tc |1st     |2nd     |3rd     |
|---|--------|--------|--------|
|tc1|12211951|11712885|11771801|
|tc2|12211951|11712885|11771801|
|tc3|11712885|12211951|11771801|
|tc4|11712885|12211951|11771801|

# Full Infinite Impulse Response
Winner of all test cases:
|tc |1st     |2nd     |3rd     |
|---|--------|--------|--------|
|tc1|12211951|11814329|11826384|
|tc2|12211951|11908621|advanced|
|tc3|12211951|advanced|11814329|
|tc4|12211951|advanced|11804173|
|tc5|12211951|11826380|11814993|
|tc6|12211951|11826380|11814993|
|tc7|12211951|11908621|11814329|

# INGP Network Forward Pass
Overall second, but first with an implementation without tensor cores (not available at my machine to write and debug the code):
|tc     |1st     |2nd     |3rd     |
|-------|--------|--------|--------|
|100x100|11771801|12211951|11826380|
|5x5    |11771801|12211951|11826380|

# INGP Whole Pipeline
Currently winning all test cases, but challenge is still ongoing:
|tc                          |1st     |2nd      |3rd     |
|----------------------------|--------|---------|--------|
|10x10_200_perspectives_100  |12211951|reference|11908621|
|10x10_200_perspectives_101  |12211951|reference|11908621|
|10x10_200_perspectives_102  |12211951|reference|11908621|
|10x10_200_perspectives_103  |12211951|reference|11908621|
|10x10_200_perspectives_104  |12211951|reference|11908621|
|800x800_200_perspectives_100|12211951|51800273 |11908621|
|800x800_200_perspectives_101|12211951|51800273 |11908621|
|800x800_200_perspectives_102|12211951|51800273 |11908621|
|800x800_200_perspectives_103|12211951|51800273 |11908621|
|800x800_200_perspectives_104|12211951|51800273 |11908621|
