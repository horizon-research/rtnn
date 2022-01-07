This will build a dynamic library that intercepts CUDA calls to dump memory allocation statistics, which is useful to track OOM issues during development. Reference: https://stackoverflow.com/questions/63924563/intercepting-cuda-calls.

1. Build the memstat library simply by running `make`. This is going to generate `libmemstatlib.so`. Note that you have to point `CUDA_HOME` in the `Makefile` to where CUDA is installed.

2. Go back to the build directory and build the main RTNN code with the `USE_SHARED_CUDA_LIBS` switch on:

```
cd ..
cmake -DUSE_SHARED_CUDA_LIBS=ON -DKNN=5 ..
LIBRARY_PATH=~/rtnn/src/memstablib make
```

Make sure `LIBRARY_PATH` point to where the library is built.

3. Run the binary as usual. Make sure `libmemstatlib.so` is at a place that can be found at run-time, or set `LD_LIBRARY_PATH`. Note that `LIBRARY_PATH` above and `LD_LIBRARY_PATH` have different functions. The former is used when linking, and the latter is used at run time. They can be pointing to the same directory though.

