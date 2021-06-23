// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis
#pragma once

#include <curand.h>

#include <iostream>
//#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <set>

#include "cuda_buffer.hpp"

using namespace bvh_radSearch;

static void context_log_cb(
    unsigned int level, const char* tag,
    const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

static void log_timer(float_t pIndex_size, float_t pBuild, float_t pSearch)
{
    std::cout << "ADS (" << pIndex_size << "MB" << ")" << " build time: " << pBuild << "ms"<< std::endl;
    std::cout << "Search time: " << pSearch << "ms" << std::endl;
}

static void log_timer(float_t pIndex_size, float_t pBuild, float_t pSearch, statistics_t pStats)
{
    std::cout << "ADS (" << pIndex_size << "MB" << ")" << " build time: " << pBuild << "ms" << std::endl;
    std::cout << "Search time: " << pSearch << "ms" << std::endl;

    std::cout << "Total gather: " << pStats.totalGather << std::endl;
    std::cout << "Max gather: " << pStats.maxGather << std::endl;
    std::cout << "Min gather: " << pStats.minGather << std::endl;
    std::cout << "Avg gather: " << pStats.avgGather << std::endl;
}

//float3* read_pc_data(const char* data_file, unsigned int* N) {
static void loadFile(const std::string& data_file, std::vector<float3>& points) {
  std::ifstream file;

  file.open(data_file.c_str());
  if( !file.good() ) {
    std::cerr << "Could not read the frame data...\n";
    //assert(0);
  }

  char line[1024];
  unsigned int lines = 0;

  while (file.getline(line, 1024)) {
    lines++;
  }
  file.clear();
  file.seekg(0, std::ios::beg);
  //float3* points = new float3[lines];
  //*N = lines;

  lines = 0;
  while (file.getline(line, 1024)) {
    double x, y, z;

    sscanf(line, "%lf,%lf,%lf\n", &x, &y, &z);
    float3 t;
    t.x = x;
    t.y = y;
    t.z = z;
    points.push_back(t);
    //std::cerr << points[lines].x << "," << points[lines].y << "," << points[lines].z << std::endl;
    lines++;
  }

  file.close();

  //return points;
}

//static void loadFile(const std::string& pFileName, std::vector<float3>& pPoints)
//{
//    std::wstring wpath = std::filesystem::current_path().native();
//    std::string path = std::string(wpath.begin(), wpath.end()) + "\\assets\\data\\";
//    tinyobj::attrib_t attrib;
//    std::vector<tinyobj::shape_t> shapes;
//    std::vector<tinyobj::material_t> materials;
//
//    std::string err;
//    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
//        (path + pFileName).c_str(), path.c_str());
//
//    if (!ret)
//    {
//        std::cout << err << std::endl;
//        return;
//    }
//
//    using h = std::hash<float_t>;
//    auto hash = [](const float3& v) {
//        return ((17 * 31 + h()(v.x)) * 31 + h()(v.y)) * 31 + h()(v.z); };
//
//    auto equal = [](const float3& l, const float3& r) { return l.x == r.x && l.y == r.y && l.z == r.z; };
//    std::unordered_map<float3, float3, decltype(hash), decltype(equal)> unqV(8, hash, equal);
//
//    for (size_t i = 0; i < attrib.vertices.size(); i += 3)
//    {
//        float3 v = make_float3(
//            attrib.vertices[i],
//            attrib.vertices[i + 1],
//            attrib.vertices[i + 2]);
//
//        unqV[v] = v;
//    }
//
//    pPoints.reserve(unqV.size());
//    for (auto& v : unqV) { pPoints.push_back(v.second); }
//}

static void splitPointCloud(
    std::vector<float3>& pSamples,
    std::vector<float3>& pQueries,
    float_t pFraction) noexcept
{
    pFraction = 1.f;
    if (pFraction == 1.f)
    {
        pSamples = pSamples;
        pQueries = pSamples;
        return;
    }

    //uint32_t numQueries = pFraction > 1.f ?
    //    uint32_t(pFraction) : pSamples.size() * pFraction;

    //cudaBuffer<float_t> randIndices;
    //randIndices.alloc(numQueries);

    //curandGenerator_t rng;
    //curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    //curandSetPseudoRandomGeneratorSeed(rng, 91);

    //curandGenerateUniform(rng, randIndices.getPtr(), randIndices.mArraySize);

    //std::vector<float3> queries;
    //std::vector<float3> newSamples;
    //std::vector<float_t> hostIndices = randIndices.getData();
    //std::vector<bool> mask(pSamples.size(), false);

    //std::set<uint32_t> cache;

    //for (uint32_t i = 0; i < numQueries; ++i)
    //{
    //    const size_t index = size_t(hostIndices[i] * pSamples.size());

    //    if (cache.find(index) == cache.cend())
    //    {
    //        queries.push_back(pSamples[index]);
    //        mask[index] = true;
    //        cache.insert(index);
    //    }
    //}

    //for (uint32_t i = 0; i < mask.size(); ++i)
    //{
    //    if (!mask[i]) newSamples.push_back(pSamples[i]);
    //}

    //pSamples = std::move(newSamples);
    //pQueries = std::move(queries);
}

static void buildAabbs(
    const std::vector<float3> & pSamples,
    float_t pRadii,
    std::vector<OptixAabb>& pAabbs)
{
    pAabbs.resize(pSamples.size());

    for (size_t i = 0; i < pSamples.size(); ++i)
    {
        float3 tmp = pSamples[i];

        OptixAabb& aabb = pAabbs[i];
        aabb.minX = tmp.x - pRadii;
        aabb.minY = tmp.y - pRadii;
        aabb.minZ = tmp.z - pRadii;

        aabb.maxX = tmp.x + pRadii;
        aabb.maxY = tmp.y + pRadii;
        aabb.maxZ = tmp.z + pRadii;
    }
}

static void assert_rad_search(
    const std::vector<query_t> & pRefQueries,
    const std::vector<query_t>& pQueries)
{
    for (size_t i = 0; i < pRefQueries.size(); ++i)
    {
        const query_t& refq = pRefQueries[i];
        const query_t& q = pQueries[i];
        assert(refq.count == q.count);
    }
}

static void assert_rad_search(
    const std::vector<query_t>& pRefQueries,
    const std::vector<int32_t>& pRefIndices,
    const std::vector<query_t>& pQueries,
    const std::vector<int32_t>& pIndices,
    const uint32_t pMaxCapacity)
{
    for (size_t i = 0; i < pRefQueries.size(); ++i)
    {
        const query_t& refq = pRefQueries[i];
        const query_t& q = pQueries[i];
        assert(refq.count == q.count);

        for (uint32_t k = 0; k < pMaxCapacity; ++k)
        {
            assert(
                pRefIndices[(pMaxCapacity * i) + k] ==
                pIndices[(pMaxCapacity * i) + k]);
        }
    }
}
