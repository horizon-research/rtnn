#pragma once

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include <vector_types.h>
#include <optix_types.h>
#include <sutil/vec_math.h>

#include "state.h"
#include "grid.h"

void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<unsigned int>, unsigned int, cudaStream_t );
void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<unsigned int>, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int>, unsigned int, cudaStream_t );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int>, unsigned int );
void sortByKey( thrust::device_vector<float>*, thrust::device_ptr<float3> );
void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<float3>, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<int>, unsigned int );
void gatherByKey ( thrust::device_vector<unsigned int>*, thrust::device_ptr<float3>, thrust::device_ptr<float3> );
void gatherByKey ( thrust::device_vector<unsigned int>*, thrust::device_ptr<float3>, thrust::device_ptr<float3>, cudaStream_t );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, thrust::device_ptr<float3>, unsigned int, cudaStream_t );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, thrust::device_ptr<float3>, unsigned int );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_vector<float>*, thrust::device_ptr<float>, unsigned int, cudaStream_t );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_vector<float>*, thrust::device_ptr<float>, unsigned int );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float>, thrust::device_ptr<float>, unsigned int );
void genSeqDevice(thrust::device_ptr<unsigned int>, unsigned int);
void genSeqDevice(thrust::device_ptr<unsigned int>, unsigned int, cudaStream_t);
void exclusiveScan(thrust::device_ptr<unsigned int>, unsigned int, thrust::device_ptr<unsigned int>, cudaStream_t);
void exclusiveScan(thrust::device_ptr<unsigned int>, unsigned int, thrust::device_ptr<unsigned int>);
void fillByValue(thrust::device_ptr<unsigned int>, unsigned int, int, cudaStream_t);
void fillByValue(thrust::device_ptr<unsigned int>, unsigned int, int);
void copyIfIdMatch(float3*, unsigned int, thrust::device_ptr<int>, thrust::device_ptr<float3>, int);
void copyIfIdInRange(float3*, unsigned int, thrust::device_ptr<int>, thrust::device_ptr<float3>, int, int);
void copyIfNonZero(float3*, unsigned int, thrust::device_ptr<bool>, thrust::device_ptr<float3>);
unsigned int countById(thrust::device_ptr<int>, unsigned int, int);
unsigned int uniqueByKey(thrust::device_ptr<unsigned int>, unsigned int N, thrust::device_ptr<unsigned int> dest);
unsigned int countUniq(thrust::device_ptr<unsigned int>, unsigned int);
void thrustCopyD2D(thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int>, unsigned int N);
unsigned int thrustGenHist(const thrust::device_ptr<int>, thrust::device_vector<unsigned int>&, unsigned int);

// take an unallocated thrust device pointer, allocate device memory and set the thrust pointer and return the raw pointer.
// https://stackoverflow.com/questions/353180/how-do-i-find-the-name-of-the-calling-function/378165
// https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
template <typename T> T* allocThrustDevicePtr(thrust::device_ptr<T>* d_memory, unsigned int N, const char* str = __builtin_FUNCTION()) {
  T* d_memory_raw;
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_memory_raw),
             N * sizeof(T) ) );
  *d_memory = thrust::device_pointer_cast(d_memory_raw);

  return d_memory_raw;
}

void kComputeMinMax (unsigned int, unsigned int, float3*, unsigned int, int3*, int3*);
void kInsertParticles(unsigned int, unsigned int, GridInfo, float3*, unsigned int*, unsigned int*, unsigned int*, bool);
void kCountingSortIndices(unsigned int, unsigned int, GridInfo, unsigned int*, unsigned int*, unsigned int*, unsigned int*);
void kCountingSortIndices_setRayMask(unsigned int, unsigned int, GridInfo, unsigned int*, unsigned int*, unsigned int*, unsigned int*, int*, int*);
void kCalcSearchSize(unsigned int,
                     unsigned int,
                     GridInfo,
                     bool, 
                     unsigned int*,
                     unsigned int*,
                     float3*,
                     float,
                     float,
                     unsigned int,
                     int*
                    );
void calcSearchSize(int3,
                    GridInfo,
                    bool, 
                    unsigned int*,
                    float,
                    float,
                    unsigned int,
                    int*
                   );
float kGetWidthFromIter(int, float);

void sanityCheck(WhittedState&);

void computeMinMax(unsigned, float3*, float3&, float3&);
void gridSort(WhittedState&, unsigned int, float3*, float3*, bool);
void sortParticles(WhittedState&, ParticleType, int);
thrust::device_ptr<unsigned int> sortQueriesByFHCoord(WhittedState&, thrust::device_ptr<unsigned int>, int);
thrust::device_ptr<unsigned int> sortQueriesByFHIdx(WhittedState&, thrust::device_ptr<unsigned int>, int);
void gatherQueries(WhittedState&, thrust::device_ptr<unsigned int>, int);

void kGenAABB(float3*, float, unsigned int, OptixAabb*, cudaStream_t);
void uploadData(WhittedState&);
void createGeometry(WhittedState&, int, float);
void launchSubframe(unsigned int*, WhittedState&, int);
void initLaunchParams(WhittedState&);
void setupOptiX(WhittedState&);
void cleanupState(WhittedState&);
float maxInscribedWidth(float, int);

int tokenize(std::string, std::string, float3**, unsigned int);
void parseArgs(WhittedState&, int, char**);
void readData(WhittedState&);
void initBatches(WhittedState&);
bool isClose(float3, float3);

void search(WhittedState&, int);
void gasSortSearch(WhittedState&, int);
thrust::device_ptr<unsigned int> initialTraversal(WhittedState&);
