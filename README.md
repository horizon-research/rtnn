# RTNN: Accelerating Neighbor Search Using Hardware Ray Tracing

## Build Instructions

### Requirements

* Nvidia CUDA Toolkit. Tested on NVCC V11.4.48 and V11.3.109.
* CMake 3.0 minimum (http://www.cmake.org/cmake/resources/software.html). Tested on 3.20.5.
* g++. Tested on 7.5.0.
* A RTX-capable GPU (Turing architecture and later) from Nvidia. Tested on RTX 2080 and RTX 2800 Ti.

### Code structure

### Build

```
cd src
mkdir build
cd build
cmake -DKNN=5 ..
make
```
Executable should be found in the `bin` directory.
