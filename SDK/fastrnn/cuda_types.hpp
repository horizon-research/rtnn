// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis
#pragma once

struct HitGroupData { int dummy; };
struct RayGenData { int dummy; };
struct MissData { int dummy; };

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct statistics_t
{
    unsigned int totalGather;
    unsigned int maxGather;
    unsigned int minGather;

    float avgGather;
};

struct query_t
{
    float3 position;
    float radius;
    unsigned int count;
};

struct Params
{
    OptixTraversableHandle gasHandle;

    query_t* queries;
    float3* samplePos;

    unsigned int numSamples;
    unsigned int knn;

    unsigned int* totalCount;
    unsigned int* minCount;
    unsigned int* maxCount;

    int* optixIndices;
    float* optixDists;
};
