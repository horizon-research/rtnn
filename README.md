# RTNN: Accelerating Neighbor Search Using Hardware Ray Tracing

This repository contains the code that uses the hardware ray tracing capability provided by Nvidia's RT cores to accelerate neighbor search in low-dimensional space (lower than 3D), which is prevalent in engineering and science fields (e.g., computational fluid dynamics, graphics, vision), because they deal with physical data such as particles and surface samples that inherently reside in the 2D/3D space.

While RT cores are designed (and optimized) for ray tracing, we show how to map neighbor search to the ray tracing hardware to achieve significant speedups (an order of magnitude) over traditional GPU (CUDA) implementations. The code is primarily developed using the OptiX programming interface (for ray tracing) and also uses CUDA to parallelize many non-ray tracing helper functions. The technical aspects of the code are discussed in this [PPoPP 2022 paper](https://www.cs.rochester.edu/horizon/pubs/ppopp22.pdf).

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

You do not have to install the OptiX SDK yourself. The code is developed using the SDK as a template and includes all the necessary headers. For reference, the particular OptiX SDK used is 7.1.0.

### Code structure

`include`: headers needed for OptiX. Copied from the OptiX SDK 7.1 without modifications.

`src`:
- `optixNSearch/`: the main source code.
- `sutil/`: the utility library from the OptiX SDK. We keep only those that are actually used in this project.
- `CMakeLists.txt`: the usual cmake file.
- `CMake/`: contains a bunch of `.cmake` files that are used by `CMakeLists.txt to find libraries, etc. This is also copied from the OptiX SDK without any change.
- `samplepc.txt`: a sample point cloud file illustrating the input file format.
- `memstatlib/`: this builds a dynamic library that tracks memory allocation in CUDA. It's not needed for the main RTNN code. See the readme file there for more information.

### Build

```
cd src
mkdir build
cd build
cmake -DKNN=5 ..
make
```
The executable is `bin/optixNSearch`.

`-DKNN=5` specifies that the maximum number of returned neighbors is 5 by passing a preprocessor macro through cmake. See `optixNSearch/CMakeLists.txt`. This `K` number is used only in the KNN search and will be overwritten by a run-time commandline flag for range search (see the description [here](#specify-maximum-returned-neighbors)), but you have to give a number here nevertheless.

## Run

### Input format

See `samplepc.txt` for an example. Each point takes a line. Each line has three coordinates separated by commas.

### Simple run

To get started, in the `build` directory run: `bin/optixNSearch -f ../samplepc.txt`. `-f` specifies the input file. This runs a range search using a radius of 2; points in `samplepc.txt` are used as both queries and search points. See the information printed in the terminal for the exact run configuration.

### Common configurations

Below are commands to some common search configurations that you might find handy.

#### KNN search on GPU 0 with a custom range of 10

`bin/optixNSearch -f ../samplepc.txt -sm knn -d 0 -r 10`

`-sm` specifies the search mode, which could either be `radius` for range search (default) or `knn` for KNN search. `-d` specifies the device/GPU ID, and `-r` specifies the range.

#### Specify maximum returned neighbors

For range search, the deafult `K` is 50. You can change it by using the `-k` switch. For instance, to return 100 neighbors run: `bin/optixNSearch -f ../samplepc.txt -k 100`.

For KNN search, the way to change `K` is to use the `-DKNN` switch during cmake and recompile. Whatever is passed in through `-k` is ignored. We need a compile-time `K` such that the priority queue used in the KNN search has a known size, which hopefully will allow the compiler to place the priority queue in registers rather than the global memory.

#### Use file f1.txt for search points and file f2.txt for queries

`bin/optixNSearch -f f1.txt -q f2.txt -p 0`

`-f` specifies the file for search points, and `-q` specifies the file for queries. If only `-f` is given, search points are used as queries.

`-p 0` disables query partitioning, a performance enhancing technique, which is currently not supported when queries and search points are different. This is by no means a fundamental limitation; the code just needs to be streamlined to support it.

### Advanced configurations

Use the `-h` switch to dump all the configuration options and their default values, which should be self-explanatory. We briefly explain some of the key options below. Needless to say, refer to the code when in doubt!

#### Query partitioning and batching

Query partition is a technique RTNN uses to improve performance. This idea is to partition queries and build a specialized BVH tree for each partition with the smallest possible AABB size that minimizes the amount of search work for that partition. By default query partitioning is enabled. Disabled it by passing `-p 0`.

Query partitioning introduces overhead that might offset the gains. Having more partitions requires building more BVHs but reduces search time, so there exists a sweet spot as to how many partitions to have. RTNN uses an analytical performance model to batch partitions to maximize the performance gain. By default this automatic batching is enabled. You can turn it off by passing `-ab 0`. You could also manually set the number of batches by using the `-nb` switch. Both switches are ignored when query partitioning is disabled.

The analytical model is constructed empirically based on measurements on RTX 2080 assuming there are no other concurrent jobs on the GPU. The model is empirical; no OptiX performance models exist. We welcome contributions to build a more accurate one.

#### Approximate search

Many applications that use neighbor search do not require exact searches, which we can leverage to improve performance. Approximation is particularly useful for KNN search, which tends to be very slow (certainly much slower than range search).

In RTNN, range search is always exact (it's fast enough anyways). KNN search is also exact when query partitioning is diabled. With query partitioning enabled, RTNN allows you to control whether/how much to approximate KNN search through the `-a` switch --- with 3 approximation levels: `0` for exact search and `2` for the most aggressive approximation. Default is `2`.

Even with the default approximate search, incorrect results *rarely* occur. In our benchmarking of datasets from different application domains (N-body simulation, graphics, autonomous machines), incorrect searches make up than 0.001% of the queries in the default setting. This is exactly our goal: noticeable performance gains (~50%) with virtually no accuracy loss.

The exact approximation mechanism we rely on is to relax the search radius of each partition to be smaller than what's strictly necessary for correctness. The default aproximation setting (`-a 2`) falls back to an exact search if the point distribution is uniform.


## FAQ

#### What do I do when I get an "out of memory" error?

Like many other GPU-based neighbor search libraries, we need to build a bunch of data structures, mainly a grid, to enable fast search. Unlike other libraries that statically allocate the memory for those data structures, we do so based on the total GPU memory capacity queried dynamically. So you should see this error much less often than in others. If you do see this, it's most likely because there are other jobs running on the GPU eating part of the GPU memory. If you can't kill those jobs, you could pass the occupied GPU memory in MB through the `-gmu` switch.

It's also possible to get an OOM error when you use different files for queries and search points. The way RTNN is currently implemented we allocate memory for the grid based on search points and use the same grid granularity for queries, which could be too fine-grained. It's a known issue. See the comment [here](https://github.com/horizon-research/rtnn/blob/main/src/optixNSearch/util.cpp#L387).

#### It seems like the first time I launch the code it takes a long time to bootstrap. Why?

Your OptiX device code is compiled after the program is launch and cached. Subsequent launches would be faster if the cache hasn't been flushed. See the discussion [here](https://forums.developer.nvidia.com/t/why-does-the-first-launch-of-optix-take-so-long/70895).

#### Is neighbor search in RTNN approximate or exact?

Range search is always exact (it's fast enough). Whether/how to approximate KNN search is controlled by the `-a` switch. See the description [here](#approximate-search).

## Acknowledgement

- The grid-based sorting is partially adapted from [cuNSearch](https://github.com/InteractiveComputerGraphics/cuNSearch).
- The prioritity queue implementation in OptiX is adaptad from the [CUDA implementation](https://github.com/facebookresearch/pytorch3d/blob/f593bfd3c258b0ff2b7bdbabfb06ab5210b43a52/pytorch3d/csrc/utils/mink.cuh) in PyTorch3D.

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
