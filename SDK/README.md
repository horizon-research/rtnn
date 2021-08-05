if we want to compile the code while dynamically linked to shared cuda library, do:
`cmake -DUSE_SHARED_CUDA_LIBS=ON ..`
follow by:
`LIBRARY_PATH=$HOME/.local/lib make`
Make sure the overloadedd cuda library is at that place and can be found at run-time (or set LD_LIBRARY_PATH).
This way the code will be linked to the overloaded cuda library. We don't have to specify LD_PRELOAD.
