// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis
#pragma once

#include <vector>

#include "cuda_buffer.hpp"
#include "cuda_types.hpp"

namespace bvh_radSearch
{
    class bvh_index final
    {
        public:

            bvh_index(const OptixDeviceContext &);
            ~bvh_index(void);

            bvh_index(const bvh_index &) = default;
            bvh_index(bvh_index &&) = default;

            bvh_index& operator=(const bvh_index &) = default;
            bvh_index& operator=(bvh_index&&) = default;

            void destroy(void);
            void init(void);

            float build(std::vector<OptixAabb> & pSamples, float * pGaS_size = nullptr);

            float radius_search(std::vector<query_t>& pQueries, std::vector<int32_t> & pIndices, std::vector<float>& pDists, unsigned int * pMaxCapacity);
            float truncated_knn(std::vector<query_t>& pQueries, unsigned int knn, std::vector<int32_t>& pIndices, std::vector<float>& pDists);
            float radius_search_count(std::vector<query_t> & pQueries, statistics_t & pStats);

            float truncated_knn_brute_force(std::vector<query_t>& pQueries, unsigned int knn, std::vector<int32_t>& pIndices, std::vector<float>& pDists);
            float radius_search_brute_force(std::vector<query_t>& pQueries, std::vector<int32_t>& pIndices, std::vector<float>& pDists, unsigned int* pMaxCapacity);
            float radius_search_count_brute_force(std::vector<query_t>& pQueries, statistics_t& pStats);

        protected:

            // Empty

        private:

            enum MODULE_E : unsigned int
            {
                COUNT_BRUTE_FORCE = 0,
                COUNT,
                RAD_SEARCH_BRUTE_FORCE,
                RAD_SEARCH,
                SIZE
            };

            struct shader_t
            {
                OptixProgramGroup raygen;
                OptixProgramGroup hit;
                OptixProgramGroup miss;

                cudaBuffer<RayGenSbtRecord> gen_rec;
                cudaBuffer<HitGroupSbtRecord> hit_rec;
                cudaBuffer<MissSbtRecord> miss_rec;

                shader_t(void) noexcept :
                    raygen      (nullptr),
                    hit         (nullptr),
                    miss        (nullptr)
                { /* Empty */ }

                ~shader_t(void)
                {
                    OPTIX_CHECK(optixProgramGroupDestroy(raygen));
                    OPTIX_CHECK(optixProgramGroupDestroy(hit));
                    OPTIX_CHECK(optixProgramGroupDestroy(miss));

                    gen_rec.destroy();
                    hit_rec.destroy();
                    miss_rec.destroy();
                }
            };

            const OptixDeviceContext & mOptixContext;
            CUdeviceptr mGasBuffer;

            cudaBuffer<Params> mDeviceParams;
            cudaBuffer<unsigned int> mTotalCount;
            cudaBuffer<unsigned int> mMinCount;
            cudaBuffer<unsigned int> mMaxCount;
            cudaBuffer<float3> mSamples;

            Params mParams;

            CUstream mCudaStream;

            OptixModule mModule;

            std::vector<shader_t> mShaders;

            OptixPipeline mPipeline;
            OptixShaderBindingTable mSBT;

            float _count(std::vector<query_t>& pQueries, statistics_t& pStats);
            float _truncated_knn(std::vector<query_t>& pQueries, unsigned int knn, std::vector<int32_t>& pIndices, std::vector<float>& pDists);
    };
}
