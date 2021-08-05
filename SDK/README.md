if we want to compile the code while dynamically linked to shared cuda library, do:
`cmake -DUSE_SHARED_CUDA_LIBS=ON ..`
follow by:
`LIBRARY_PATH=~/optixSDK/SDK/mylib/ make`
Run code by:
`LD_PRELOAD=~/optixSDK/SDK/mylib/libmylib.so bin/optixRangeSearch -f file`
