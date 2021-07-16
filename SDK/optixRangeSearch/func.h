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
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<char>, unsigned int );
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
thrust::device_ptr<char> getThrustDeviceCharPtr(unsigned int);
void genSeqDevice(thrust::device_ptr<unsigned int>, unsigned int);
void genSeqDevice(thrust::device_ptr<unsigned int>, unsigned int, cudaStream_t);
void exclusiveScan(thrust::device_ptr<unsigned int>, unsigned int, thrust::device_ptr<unsigned int>, cudaStream_t);
void exclusiveScan(thrust::device_ptr<unsigned int>, unsigned int, thrust::device_ptr<unsigned int>);
void fillByValue(thrust::device_ptr<unsigned int>, unsigned int, int, cudaStream_t);
void fillByValue(thrust::device_ptr<unsigned int>, unsigned int, int);
void copyIfIdMatch(float3*, unsigned int, thrust::device_ptr<char>, thrust::device_ptr<float3>, char);
void copyIfIdInRange(float3*, unsigned int, thrust::device_ptr<char>, thrust::device_ptr<float3>, char, char);
void copyIfNonZero(float3*, unsigned int, thrust::device_ptr<bool>, thrust::device_ptr<float3>);
unsigned int countById(thrust::device_ptr<char>, unsigned int, char);
unsigned int uniqueByKey(thrust::device_ptr<unsigned int>, unsigned int N, thrust::device_ptr<unsigned int> dest);
void thrustCopyD2D(thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int>, unsigned int N);
unsigned int thrustGenHist(const thrust::device_ptr<char>, thrust::device_vector<unsigned int>&, unsigned int);

void kComputeMinMax (unsigned int, unsigned int, float3*, unsigned int, int3*, int3*);
void kInsertParticles(unsigned int, unsigned int, GridInfo, float3*, unsigned int*, unsigned int*, unsigned int*, bool);
void kCountingSortIndices(unsigned int, unsigned int, GridInfo, unsigned int*, unsigned int*, unsigned int*, unsigned int*);
void kCountingSortIndices_setRayMask(unsigned int, unsigned int, GridInfo, unsigned int*, unsigned int*, unsigned int*, unsigned int*, char*, char*);
uint kToCellIndex_MortonMetaGrid(const GridInfo&, int3);
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
                     char*
                    );
float kGetWidthFromIter(int, float);

void sanityCheck(WhittedState&);

void computeMinMax(WhittedState&, ParticleType);
void gridSort(WhittedState&, ParticleType, bool);
void sortParticles(WhittedState&, ParticleType, int);
thrust::device_ptr<unsigned int> sortQueriesByFHCoord(WhittedState&, thrust::device_ptr<unsigned int>, int);
thrust::device_ptr<unsigned int> sortQueriesByFHIdx(WhittedState&, thrust::device_ptr<unsigned int>, int);
void gatherQueries(WhittedState&, thrust::device_ptr<unsigned int>, int);

void kGenAABB(float3*, float, unsigned int, CUdeviceptr, cudaStream_t);
void uploadData(WhittedState&);
void createGeometry(WhittedState&, int);
void launchSubframe(unsigned int*, WhittedState&, int);
void initLaunchParams(WhittedState&);
void setupOptiX(WhittedState&);
void cleanupState(WhittedState&);

int tokenize(std::string, std::string, float3**, unsigned int);
void parseArgs(WhittedState&, int, char**);
void readData(WhittedState&);
void initBatches(WhittedState&);
bool isClose(float3, float3);

void search(WhittedState&, int);
void gasSortSearch(WhittedState&, int);
thrust::device_ptr<unsigned int> initialTraversal(WhittedState&);
