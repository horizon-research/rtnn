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

// the SDK cmake defines NDEBUG in the Release build, but we still want to use assert
// TODO: fix it in cmake files?
#undef NDEBUG
#include <assert.h>

//void sortByKey( thrust::device_vector<float>*, thrust::device_vector<unsigned int>* );
//void sortByKey( thrust::device_vector<float>*, thrust::device_ptr<unsigned int>);
void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<unsigned int>, unsigned int, cudaStream_t );
void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<unsigned int>, unsigned int );
//void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_vector<unsigned int>*, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int>, unsigned int, cudaStream_t );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int>, unsigned int );
void sortByKey( thrust::device_vector<float>*, thrust::device_ptr<float3> );
void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<float3>, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<bool>, unsigned int );
void gatherByKey ( thrust::device_vector<unsigned int>*, thrust::device_ptr<float3>, thrust::device_ptr<float3> );
void gatherByKey ( thrust::device_vector<unsigned int>*, thrust::device_ptr<float3>, thrust::device_ptr<float3>, cudaStream_t );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, thrust::device_ptr<float3>, unsigned int, cudaStream_t );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, thrust::device_ptr<float3>, unsigned int );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_vector<float>*, thrust::device_ptr<float>, unsigned int, cudaStream_t );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_vector<float>*, thrust::device_ptr<float>, unsigned int );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float>, thrust::device_ptr<float>, unsigned int );
thrust::device_ptr<unsigned int> getThrustDevicePtr(unsigned int);
thrust::device_ptr<float3> getThrustDeviceF3Ptr(unsigned int);
thrust::device_ptr<float> getThrustDeviceF1Ptr(unsigned int);
thrust::device_ptr<bool> getThrustDeviceBoolPtr(unsigned int);
thrust::device_ptr<unsigned int> genSeqDevice(unsigned int);
thrust::device_ptr<unsigned int> genSeqDevice(unsigned int, cudaStream_t);
void exclusiveScan(thrust::device_ptr<unsigned int>, unsigned int, thrust::device_ptr<unsigned int>, cudaStream_t);
void exclusiveScan(thrust::device_ptr<unsigned int>, unsigned int, thrust::device_ptr<unsigned int>);
void fillByValue(thrust::device_ptr<unsigned int>, unsigned int, int, cudaStream_t);
void fillByValue(thrust::device_ptr<unsigned int>, unsigned int, int);
void copyIfStencil(float3*, unsigned int, thrust::device_ptr<bool>, thrust::device_ptr<float3>, bool);
unsigned int countByPred(thrust::device_ptr<bool>, unsigned int, bool);

void kComputeMinMax (unsigned int, unsigned int, float3*, unsigned int, int3*, int3*);
void kInsertParticles(unsigned int, unsigned int, GridInfo, float3*, unsigned int*, unsigned int*, unsigned int*, bool);
void kCountingSortIndices(unsigned int, unsigned int, GridInfo, unsigned int*, unsigned int*, unsigned int*, unsigned int*);
void kCountingSortIndices_genMask(unsigned int, unsigned int, GridInfo, unsigned int*, unsigned int*, unsigned int*, unsigned int*, bool*, bool*);

void sanityCheck(WhittedState&);

void computeMinMax(WhittedState&, ParticleType);
void gridSort(WhittedState&, ParticleType, bool);
void sortParticles(WhittedState&, ParticleType, int);
thrust::device_ptr<unsigned int> sortQueriesByFHCoord(WhittedState&, thrust::device_ptr<unsigned int>, int);
thrust::device_ptr<unsigned int> sortQueriesByFHIdx(WhittedState&, thrust::device_ptr<unsigned int>, int);
void gatherQueries(WhittedState&, thrust::device_ptr<unsigned int>, int);

void kGenAABB(float3*, float, unsigned int, CUdeviceptr, cudaStream_t);
void setupCUDA(WhittedState&);
void uploadData(WhittedState&);
void createGeometry(WhittedState&, int);
void launchSubframe(unsigned int*, WhittedState&, int);
void initLaunchParams(WhittedState&);
void setupOptiX(WhittedState&);
void cleanupState(WhittedState&);

int tokenize(std::string, std::string, float3**, unsigned int);
void parseArgs(WhittedState&, int, char**);
void readData(WhittedState&);

void search(WhittedState&, int);
void gasSortSearch(WhittedState&, int);
thrust::device_ptr<unsigned int> initialTraversal(WhittedState&);
