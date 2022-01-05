# RTNN: Accelerating Neighbor Search Using Hardware Ray Tracing

This repository contains the code that uses the hardware ray tracing capability provided by Nvidia's RT cores to accelerate neighbor search in low-dimensional space (lower than 3D), which is prevalent in engineering and science fields (e.g., computational fluid dynamics, graphics, vision), because they deal with physical data such as particles and surface samples that inherently reside in the 2D/3D space.

While RT cores are designed (and optimized) for ray tracing, we show how to map neighbor search to the ray tracing hardware to achieve significant speedups (over an order of magnitude) to traditional GPU (CUDA) implementations of neighbor search. The code is primarily developed using the OptiX programming interface (for ray tracing) and also uses CUDA to parallelize many non-ray tracing helper functions. The technical aspects of the code are discussed in this [PPoPP 2021 paper](https://www.cs.rochester.edu/horizon/pubs/ppopp22.pdf).

## Build Instructions

### Requirements

* Nvidia CUDA Toolkit. Tested on 11.3.109 and 11.4.48. We use [Thrust](https://github.com/NVIDIA/thrust) that is installed along with the Toolkit.
* CMake. Tested on 3.20.5.
* g++. Tested on 7.5.0.
* A RTX-capable GPU (Turing architecture and later) from Nvidia. Tested on RTX 2080 and RTX 2800 Ti.

You do not have to install the Optix SDK yourself. The code is developed using the SDK as a template and includes all the necessary headers. The particular Optix SDK used is 7.1.0.

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
