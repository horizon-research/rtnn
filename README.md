# RTNN: Accelerating Neighbor Search Using Hardware Ray Tracing

This repository contains the code that uses the hardware ray tracing capability provided by Nvidia's RT cores to accelerate neighbor search in low-dimensional space (lower than 3D), which is prevalent in engineering and science fields (e.g., computational fluid dynamics, graphics, vision), because they deal with physical data such as particles and surface samples that inherently reside in the 2D/3D space.

While RT cores are designed (and optimized) for ray tracing, we show how to map neighbor search to the ray tracing hardware to achieve significant speedups (over an order of magnitude) to traditional GPU (CUDA) implementations of neighbor search. The code is primarily developed using the OptiX programming interface (for ray tracing) and also uses CUDA to parallelize many non-ray tracing helper functions. The technical aspects of the code are discussed in this [PPoPP 2022 paper](https://www.cs.rochester.edu/horizon/pubs/ppopp22.pdf).

### What forms of neighbor search are supported?

Two types of neighbor search exist: fixed-radius search (a.k.a., range search) and K nearest neighbor search (KNN). RTNN optimizes for both types. For both types of search we assume a search interface that provides a search radius `r` and a maximum neighbor count `K`, consistent with the interface of many existing neighbor search libraries. We could emulate an unbounded KNN search by providing a very large `r` and emulate an unbounded range search by providing a very large `K`.

Why do we need a search radius (even in KNN searches)? In practical applications the returned neighbors are usually bounded by a search radius, beyond which the neighbors are discarded. This is because the significance of a neighbor (e.g., the force that a particle exerts on another) is minimal and of little interest when it is too far away.

Why do we need to bound `K` (even in range searches)? In practical applications the maximum amount of returned neighbors is bounded in order to bound the memory consumption and to interface with downstream tasks, which usually expect a fixed amount of neighbors.

## Build Instructions

### Requirements

* NVCC. Tested on 11.3.109 and 11.4.48.
* CMake. Tested on 3.20.5.
* g++. Tested on 7.5.0.
* [Thrust](https://github.com/NVIDIA/thrust). Tested on v10.11.00.
* A RTX-capable GPU (Turing architecture and later) from Nvidia. Tested on RTX 2080 and RTX 2800 Ti.

You do not have to install the OptiX SDK yourself. The code is developed using the SDK as a template and includes all the necessary headers. The particular OptiX SDK used is 7.1.0.

### Code structure

`include`: headers needed for OptiX. Copied from the OptiX SDK 7.1 without modifications.

`src`:
- `optixNSearch/`: the main source code.
- `sutil/`: the utility library from the OptiX SDK. We keep only those that are actually used in this project.
- `CMakeLists.txt`: the usual cmake file.
- `CMake/`: contains a bunch of `.cmake` files that are used by `CMakeLists.txt to find libraries, etc. This is also copied from the OptiX SDK without any change.
- `samplepc.txt`: a sample point cloud file illustrating the input file format.

### Build

```
cd src
mkdir build
cd build
cmake -DKNN=5 ..
make
```
The executable is `bin/optixNSearch`.

`-DKNN=5` specifies that the maximum number of returned neighbors is 5 by passing a preprocessor macro through cmake. See `optixNSearch/CMakeLists.txt`. Set it to a number that fits your application.

## Run

## FAQ

#### What do I do when I get an "out of memory" error?

Like many other GPU-based neighbor search libraries, we need to build a bunch of data structures to enable fast search. Unlike other libraries that statically allocate the memory for those data structures, we do so based on the total GPU memory capacity queried dynamically. So you should see this error much less often than in others. If you do see this, it's most likely because there are other jobs running on the GPU eating part of the GPU memory. If you can't kill those jobs, you could pass the occupied GPU memory in MB through the `-gmu` switch.

#### It seems like the first time I launch the code it takes a long time to bootstrap. Why?

Your OptiX device code is compiled after the program is launch and cached. Subsequent launches would be faster if the cache hasn't been flushed. See the discussion [here](https://forums.developer.nvidia.com/t/why-does-the-first-launch-of-optix-take-so-long/70895).

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
