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

## Publication

The project contains the artifact of the PPoPP 2022 paper [RTNN: Accelerating Neighbor Search Using Hardware Ray Tracing](https://www.cs.rochester.edu/horizon/pubs/ppopp22.pdf).

```
@inproceedings{zhu2022rtnn,
    title={RTNN: Accelerating Neighbor Search Using Hardware Ray Tracing},
    author={Zhu, Yuhao},
    booktitle={Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming},
    year={2022},
    organization={ACM}
}
```
