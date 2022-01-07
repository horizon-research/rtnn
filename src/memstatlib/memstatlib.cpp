#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <map>

std::map<void*, double> memmap;
double tot_alloc_size = 0;

cudaError_t cudaMalloc ( void** dst, size_t count )
{
    cudaError_t (*lcudaMalloc) ( void**, size_t ) = (cudaError_t (*) ( void**, size_t ))dlsym(RTLD_NEXT, "cudaMalloc");
    cudaError_t msg = lcudaMalloc( dst, count );
    double size = (double)count/1024/1024;
    printf("[MEM_STATS] cudaMalloc (%p): %lf MB\n", *dst, size);
    memmap[*dst] = size;
    tot_alloc_size += size;
    printf("[MEM_STATS] %lf MB\n", tot_alloc_size);
    return msg;
}

cudaError_t cudaFree ( void* dst )
{
    cudaError_t (*lcudaFree) ( void* ) = (cudaError_t (*) ( void* ))dlsym(RTLD_NEXT, "cudaFree");
    cudaError_t msg = lcudaFree( dst );
    printf("[MEM_STATS] cudaFree (%p): %lf MB\n", dst, memmap[dst]);
    tot_alloc_size -= memmap[dst];
    printf("[MEM_STATS] %lf MB\n", tot_alloc_size);
    return msg;
}

//cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
//{
//    cudaError_t (*lcudaMemcpy) ( void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind  ))dlsym(RTLD_NEXT, "cudaMemcpy");
//    printf("cudaMemcpy: %lf MB\n", (double)count/1024/1024);
//    return lcudaMemcpy( dst, src, count, kind );
//}
//
//cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str )
//{
//    cudaError_t (*lcudaMemcpyAsync) ( void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind, cudaStream_t   ))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
//    printf("cudaMemcpyAsync: %lf MB\n", (double)count/1024/1024);
//    return lcudaMemcpyAsync( dst, src, count, kind, str );
//}
