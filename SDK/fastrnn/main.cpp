// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>

#include <optix.h>
#include <optix_types.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "radius_search.hpp"
#include "optix_types.h"
#include "Exception_modified.h"
#include "cuda_buffer.hpp"
#include "vec_math.hpp"
#include "utils.hpp"

#include <curand.h>

using namespace bvh_radSearch;

//void test_uniform_synthetic(const OptixDeviceContext & pCntx) noexcept
//{
//    auto index = bvh_radSearch::bvh_index(pCntx);
//    index.init();
//
//    const uint32_t numSamples = 500000;
//    const uint32_t numQueries = 500000;
//
//    const float_t scaling = 10.f;
//    const float_t radius = 0.1;
//
//    curandGenerator_t rng;
//    cudaBuffer<float3> samples{};
//    cudaBuffer<float3> queries{};
//
//    samples.alloc(numSamples);
//    queries.alloc(numQueries);
//
//    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
//    curandSetPseudoRandomGeneratorSeed(rng, 91);
//
//    curandGenerateUniform(rng, &samples.getPtr()->x, samples.mArraySize * 3);
//    curandGenerateUniform(rng, &queries.getPtr()->x, queries.mArraySize * 3);
//
//    std::vector<float3> hostSamples = samples.getData();
//    std::vector<float3> hostPoints = queries.getData();
//    std::vector<query_t> hostQueries{};
//    std::vector<OptixAabb> aabbs{};
//
//    hostQueries.reserve(queries.mArraySize);
//
//    statistics_t stats{};
//
//    std::transform(hostSamples.begin(), hostSamples.end(),
//        hostSamples.begin(),
//        [&](float3& pVal) { return pVal * scaling; });
//
//    std::for_each(hostPoints.begin(), hostPoints.end(),
//        [&](float3& pVal) {
//            query_t q{};
//            q.position = pVal * scaling;
//            q.radius = radius;
//            q.count = 0;
//            hostQueries.push_back(q); });
//
//    buildAabbs(hostSamples, radius, aabbs);
//
//    hostSamples.clear();
//    hostPoints.clear();
//    queries.destroy();
//    samples.destroy();
//
//    std::vector<int32_t> indices;
//    std::vector<float_t> dists;
//
//    float_t gas_size = 0.f;
//    float_t ms_build_timer = index.build(aabbs, &gas_size);
//
//#ifdef _DEBUG
//    statistics_t dstats{};
//    std::vector<query_t> dhostQueries = hostQueries;
//    index.radius_search_count_brute_force(dhostQueries, dstats);
//    float_t ms_search_timer = index.radius_search_count(hostQueries, stats);
//
//    assert_rad_search(dhostQueries, hostQueries);
//#else
//    float_t ms_search_timer = index.radius_search_count(hostQueries, stats);
//#endif
//
//    index.destroy();
//    log_timer(gas_size, ms_build_timer, ms_search_timer, stats);
//}

void test_point_cloud(const OptixDeviceContext& pCntx,
    const std::string & pFileName,
    float_t pRadii, float_t pSplitFrc, int32_t knn = -1)
{
    std::vector<float3> points{};
    std::vector<float3> queryPoints{};
    std::vector<query_t> queries{};
    std::vector<OptixAabb> aabbs{};
    std::vector<int32_t> indices{};
    std::vector<float_t> dists{};

    loadFile(pFileName, points);
    splitPointCloud(points, queryPoints, pSplitFrc);
    buildAabbs(points, pRadii, aabbs);

    std::for_each(queryPoints.begin(), queryPoints.end(),
        [&](float3& pVal) {
            query_t q{};
            q.position = pVal;
            q.radius = pRadii;
            q.count = 0;
            queries.push_back(q); });

    auto index = bvh_radSearch::bvh_index(pCntx);
    index.init();

    float_t gas_size = 0.f;
    float_t ms_build_timer = index.build(aabbs, &gas_size);
    float_t ms_search_timer = 0.f;

    uint32_t maxCapacity = 0;

#ifdef _DEBUG

    std::vector<query_t> dqueries = queries;
    std::vector<int32_t> dindices{};
    std::vector<float_t> ddists{};
    uint32_t dmaxCapacity = 0;

    if (knn == -1)
    {
        index.radius_search_brute_force(dqueries, dindices, ddists, &dmaxCapacity);
        ms_search_timer = index.radius_search(queries, indices, dists, &maxCapacity);
    }
    else
    {
        index.truncated_knn_brute_force(dqueries, knn, dindices, ddists);
        ms_search_timer = index.truncated_knn(queries, knn, indices, dists);
    }

    assert(dmaxCapacity == maxCapacity);
    assert_rad_search(dqueries, dindices, queries, indices, maxCapacity);
#else

    if (knn == -1)
    {
        ms_search_timer = index.radius_search(queries, indices, dists, &maxCapacity);
    }
    else
    {
        ms_search_timer = index.truncated_knn(queries, knn, indices, dists);
    }
#endif

    index.destroy();
    std::cout << "Log: " << pFileName << std::endl;
    log_timer(gas_size, ms_build_timer, ms_search_timer);
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    OptixDeviceContext context = nullptr;

    try
    {
        std::string outfile;
        outfile = argv[1];
        float radius = std::stof(argv[2]);
        int knn = atoi(argv[3]);
        int32_t device_id = atoi(argv[4]);

        int32_t device_count = 0;
        CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
        std::cout << "Total GPUs visible: " << device_count << std::endl;

        cudaDeviceProp prop;
        CUDA_CHECK( cudaGetDeviceProperties ( &prop, device_id ) );
        CUDA_CHECK( cudaSetDevice( device_id ) );
        std::cout << "\t[" << device_id << "]: " << prop.name << std::endl;

        CUDA_CHECK(cudaFree(0));

        CUcontext cuCtx = 0;
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        //options.logCallbackFunction = &context_log_cb;
        options.logCallbackFunction = nullptr;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

        //test_uniform_synthetic(context);

        //**Note:** Uncompress the data files in assets/data/*
        //          before executing the following examples

        test_point_cloud(context, outfile.c_str(), radius, 1.f, knn);
        /*
        //params: optixContext, objFile, radius, fraction of samples as queries, knn

        test_point_cloud(context, "Metope.obj", .8f, .5f, 1);
        test_point_cloud(context, "Metope.obj", .9f, 1.f, 4);
        test_point_cloud(context, "Metope.obj", 1.2f, 1.f, 8);

        test_point_cloud(context, "Lionhead.obj", 1.5f, .5f, 1);
        test_point_cloud(context, "Lionhead.obj", 1.0f, 1.f, 4);
        test_point_cloud(context, "Lionhead.obj", 1.4f, 1.f, 8);

        test_point_cloud(context, "DoraBlock.obj", .4f, .5f, 1);
        test_point_cloud(context, "DoraBlock.obj", .5f, 1.f, 4);
        test_point_cloud(context, "DoraBlock.obj", .7f, 1.f, 8);

        test_point_cloud(context, "Hermes.obj", .6f, .5f, 1);
        test_point_cloud(context, "Hermes.obj", .65f, 1.f, 4);
        test_point_cloud(context, "Hermes.obj", .9f, 1.f, 8);
        */

        OPTIX_CHECK(optixDeviceContextDestroy(context));
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
    }

    return 0;
}
