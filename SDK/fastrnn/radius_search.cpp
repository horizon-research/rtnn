// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix.h>
#include <optix_types.h>
#include <cuda_runtime.h>

//#include <filesystem>
#include <sutil/sutil.h>

#include "radius_search.hpp"
#include "Exception_modified.h"
#include "cuda_timer.hpp"
#include "sutil_modified.h"

#include "vec_math.hpp"

namespace bvh_radSearch
{
    bvh_index::bvh_index(const OptixDeviceContext & pOptixContext) :
        mOptixContext           (pOptixContext),
        mGasBuffer              (0),
        mDeviceParams           (),
        mTotalCount             (),
        mMinCount               (),
        mMaxCount               (),
        mParams                 (),
        mModule                 (nullptr),
        mPipeline               (nullptr),
        mCudaStream             (nullptr),
        mSBT                    ({})
    {
        this->mShaders.resize(MODULE_E::SIZE);
    }

    bvh_index::~bvh_index(void) { /* Empty */ }

    void bvh_index::destroy(void)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->mGasBuffer)));

        CUDA_CHECK(cudaStreamDestroy(this->mCudaStream));

        OPTIX_CHECK(optixModuleDestroy(this->mModule));

        this->mShaders.clear();

        OPTIX_CHECK(optixPipelineDestroy(this->mPipeline));

        this->mDeviceParams.destroy();
        this->mTotalCount.destroy();
        this->mMinCount.destroy();
        this->mMaxCount.destroy();
        this->mSamples.destroy();
    }

    void bvh_index::init(void)
    {
        this->mShaders.resize(MODULE_E::SIZE);

        char log[2048]{};

        OptixPipelineCompileOptions pipeline_compile_options = {};
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 2;
        pipeline_compile_options.numAttributeValues = 2;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

        //std::wstring wpath = std::filesystem::current_path().native();
        //std::string path = std::string(wpath.begin(), wpath.end()) + "\\assets\\ptx\\";
        //std::string ptx{};
        //getPtxString(ptx, path.c_str(), "radius_search.ptx");
        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "radius_search.cu" );
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            this->mOptixContext,
            &module_compile_options,
            &pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &this->mModule
        ));

        std::vector<std::string> module_names({
            "radSearch_count_bruteforce",
            "radSearch_count",
            "radSearch_bruteforce",
            "radSearch" });

        std::vector<OptixProgramGroup> programs;

        for (size_t i = 0; i < MODULE_E::SIZE; ++i)
        {
            char log[2048];
            size_t sizeof_log = sizeof(log);

            OptixProgramGroupOptions program_group_options = { 0 };
            OptixProgramGroupDesc raygen_prog_group_desc = {};

            std::string rg_func = "__raygen__" + module_names[i];
            std::string is_func = "__intersection__" + module_names[i];

            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = this->mModule;
            raygen_prog_group_desc.raygen.entryFunctionName = rg_func.c_str();

            OPTIX_CHECK_LOG(optixProgramGroupCreate(this->mOptixContext,
                &raygen_prog_group_desc, 1, &program_group_options,
                log, &sizeof_log, &this->mShaders[i].raygen));

            RayGenSbtRecord rg_sbt;
            rg_sbt.data = {};
            this->mShaders[i].gen_rec.alloc(1);
            OPTIX_CHECK(optixSbtRecordPackHeader(this->mShaders[i].raygen, &rg_sbt));
            this->mShaders[i].gen_rec.copyHostToDevice(&rg_sbt);

            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = nullptr;
            miss_prog_group_desc.miss.entryFunctionName = nullptr;

            OPTIX_CHECK_LOG(optixProgramGroupCreate(this->mOptixContext,
                &miss_prog_group_desc, 1, &program_group_options,
                log, &sizeof_log, &this->mShaders[i].miss));

            MissSbtRecord miss_sbt;
            miss_sbt.data = {};
            this->mShaders[i].miss_rec.alloc(1);
            OPTIX_CHECK(optixSbtRecordPackHeader(this->mShaders[i].miss, &miss_sbt));
            this->mShaders[i].miss_rec.copyHostToDevice(&miss_sbt);

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
            hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.moduleIS = this->mModule;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = is_func.c_str();

            OPTIX_CHECK_LOG(optixProgramGroupCreate(this->mOptixContext,
                &hitgroup_prog_group_desc, 1, &program_group_options,
                log, &sizeof_log, &this->mShaders[i].hit));

            HitGroupSbtRecord hg_sbt;
            hg_sbt.data = {};
            this->mShaders[i].hit_rec.alloc(1);
            OPTIX_CHECK(optixSbtRecordPackHeader(this->mShaders[i].hit, &hg_sbt));
            this->mShaders[i].hit_rec.copyHostToDevice(&hg_sbt);

            programs.push_back(this->mShaders[i].raygen);
            programs.push_back(this->mShaders[i].miss);
            programs.push_back(this->mShaders[i].hit);
        }

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = 1;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        OPTIX_CHECK_LOG(optixPipelineCreate(
            this->mOptixContext,
            &pipeline_compile_options,
            &pipeline_link_options,
            programs.data(),
            programs.size(),
            log,
            &sizeof_log,
            &this->mPipeline));

        OptixStackSizes stack_sizes = {};
        for (auto& prog_group : programs)
        {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1,
            0,  // maxCCDepth
            0,  // maxDCDEpth
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(this->mPipeline, direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state, continuation_stack_size,
            1  // maxTraversableDepth
        ));

        this->mSBT.missRecordStrideInBytes = sizeof(MissSbtRecord);
        this->mSBT.missRecordCount = this->mShaders.size();
        this->mSBT.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        this->mSBT.hitgroupRecordCount = this->mShaders.size();

        CUDA_CHECK(cudaStreamCreate(&this->mCudaStream));

        this->mDeviceParams.alloc();
        this->mMaxCount.alloc();
        this->mMinCount.alloc();
        this->mTotalCount.alloc();
    }

    float_t bvh_index::build(std::vector<OptixAabb>& pSamples,
        float_t* pGaS_size)
    {
        OptixAccelBuildOptions accel_options = {};

        accel_options.buildFlags =
            OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
            OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        cudaBuffer<OptixAabb> aabbs;
        aabbs.alloc(pSamples.size());
        aabbs.copyHostToDevice(pSamples.data());

        OptixBuildInput aabb_input = {};

        uint32_t aabb_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
        aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        aabb_input.customPrimitiveArray.aabbBuffers = &aabbs.getRawPtr();
        aabb_input.customPrimitiveArray.numPrimitives = pSamples.size();
        aabb_input.customPrimitiveArray.flags = aabb_input_flags;
        aabb_input.customPrimitiveArray.numSbtRecords = 1;
        aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(this->mOptixContext, &accel_options, &aabb_input, 1, &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
            compactedSizeOffset + 8));

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

        this->mSamples.destroy();
        std::vector<float3> hostSamples(pSamples.size());

        for (size_t i = 0; i < pSamples.size(); ++i)
        {
            hostSamples[i] = 0.5 * make_float3(
                pSamples[i].maxX + pSamples[i].minX,
                pSamples[i].maxY + pSamples[i].minY,
                pSamples[i].maxZ + pSamples[i].minZ);
        }

        this->mSamples.alloc(pSamples.size());
        this->mSamples.copyHostToDevice(hostSamples.data());
        this->mParams.samplePos = this->mSamples.getPtr();
        this->mParams.numSamples = pSamples.size();
        hostSamples.clear();

        cudaTimer timer;
        timer.start();

        OPTIX_CHECK(optixAccelBuild(this->mOptixContext,
            0,
            &accel_options,
            &aabb_input,
            1,
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size,
            gas_buffer_sizes.outputSizeInBytes,
            &this->mParams.gasHandle,
            &emitProperty,
            1));

        CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));

        size_t compacted_gas_size;

        CUDA_CHECK(cudaMemcpy(&compacted_gas_size,
            (void*)emitProperty.result,
            sizeof(size_t), cudaMemcpyDeviceToHost));

        if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
        {
            if(pGaS_size) *pGaS_size = compacted_gas_size / (1024.f * 1024.f);

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mGasBuffer), compacted_gas_size));

            OPTIX_CHECK(optixAccelCompact(this->mOptixContext, 0,
                this->mParams.gasHandle, this->mGasBuffer,
                compacted_gas_size,
                &this->mParams.gasHandle));

            CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
        }
        else
        {
            if (pGaS_size) *pGaS_size = gas_buffer_sizes.outputSizeInBytes / (1024.f * 1024.f);
            this->mGasBuffer = d_buffer_temp_output_gas_and_compacted_size;
        }

        return timer.stop();
    }

    float_t bvh_index::radius_search_count(
        std::vector<query_t>& pQueries,
        statistics_t& pStats)
    {
        this->mSBT.raygenRecord = this->mShaders[MODULE_E::COUNT].gen_rec.getRawPtr();
        this->mSBT.missRecordBase = this->mShaders[MODULE_E::COUNT].miss_rec.getRawPtr();
        this->mSBT.hitgroupRecordBase = this->mShaders[MODULE_E::COUNT].hit_rec.getRawPtr();

        return this->_count(pQueries, pStats);
    }

    float_t bvh_index::radius_search_count_brute_force(
        std::vector<query_t>& pQueries,
        statistics_t& pStats)
    {
        this->mSBT.raygenRecord = this->mShaders[MODULE_E::COUNT_BRUTE_FORCE].gen_rec.getRawPtr();
        this->mSBT.missRecordBase = this->mShaders[MODULE_E::COUNT_BRUTE_FORCE].miss_rec.getRawPtr();
        this->mSBT.hitgroupRecordBase = this->mShaders[MODULE_E::COUNT_BRUTE_FORCE].hit_rec.getRawPtr();

        return this->_count(pQueries, pStats);
    }

    float_t bvh_index::_count(std::vector<query_t>& pQueries, statistics_t& pStats)
    {
        cudaBuffer<query_t> queries;
        queries.alloc(pQueries.size());
        queries.copyHostToDevice(pQueries.data());

        this->mParams.queries = queries.getPtr();
        this->mMinCount.memset(0x00000064U);
        this->mMaxCount.reset();
        this->mTotalCount.reset();
        this->mParams.totalCount = this->mTotalCount.getPtr();
        this->mParams.maxCount = this->mMaxCount.getPtr();
        this->mParams.minCount = this->mMinCount.getPtr();
        this->mDeviceParams.copyHostToDevice(&this->mParams, 1);

        cudaTimer timer;

        timer.start();
        OPTIX_CHECK(optixLaunch(
            this->mPipeline,
            this->mCudaStream,
            this->mDeviceParams.getRawPtr(),
            sizeof(Params),
            &this->mSBT,
            pQueries.size(), 1, 1));
        CUDA_SYNC_CHECK();
        float_t elapsed_ms = timer.stop();

        pStats.maxGather = this->mMaxCount.getData()[0];
        pStats.minGather = this->mMinCount.getData()[0];
        pStats.totalGather = this->mTotalCount.getData()[0];
        pStats.avgGather = pStats.totalGather / float_t(pQueries.size());
        pQueries = queries.getData();

        return elapsed_ms;
    }

    float_t bvh_index::radius_search(
        std::vector<query_t>& pQueries,
        std::vector<int32_t>& pIndices,
        std::vector<float_t>& pDists,
        uint32_t* pMaxCapacity)
    {
        statistics_t stats{};

        this->radius_search_count(pQueries, stats);
        // so we first do a radius search and just count how many neighbors
        // each query has; we then allocate GPU memory that way and do the
        // search again, this time with the actual neighbor indices stored and
        // transferred back.
        *pMaxCapacity = stats.maxGather;
        std::cout << stats.maxGather << std::endl;
        return this->truncated_knn(pQueries, stats.maxGather, pIndices, pDists);
    }

    float_t bvh_index::truncated_knn(
        std::vector<query_t>& pQueries,
        uint32_t knn,
        std::vector<int32_t>& pIndices,
        std::vector<float_t>& pDists)
    {
        this->mSBT.raygenRecord = this->mShaders[MODULE_E::RAD_SEARCH].gen_rec.getRawPtr();
        this->mSBT.missRecordBase = this->mShaders[MODULE_E::RAD_SEARCH].miss_rec.getRawPtr();
        this->mSBT.hitgroupRecordBase = this->mShaders[MODULE_E::RAD_SEARCH].hit_rec.getRawPtr();

        return this->_truncated_knn(pQueries, knn, pIndices, pDists);
    }

    float_t bvh_index::radius_search_brute_force(
        std::vector<query_t>& pQueries,
        std::vector<int32_t>& pIndices,
        std::vector<float_t>& pDists,
        uint32_t* pMaxCapacity)
    {
        statistics_t stats{};

        this->radius_search_count(pQueries, stats);
        *pMaxCapacity = stats.maxGather;
        return this->truncated_knn_brute_force(pQueries, stats.maxGather, pIndices, pDists);
    }

    float_t bvh_index::truncated_knn_brute_force(
        std::vector<query_t>& pQueries,
        uint32_t knn,
        std::vector<int32_t>& pIndices,
        std::vector<float_t>& pDists)
    {
        this->mSBT.raygenRecord = this->mShaders[MODULE_E::RAD_SEARCH_BRUTE_FORCE].gen_rec.getRawPtr();
        this->mSBT.missRecordBase = this->mShaders[MODULE_E::RAD_SEARCH_BRUTE_FORCE].miss_rec.getRawPtr();
        this->mSBT.hitgroupRecordBase = this->mShaders[MODULE_E::RAD_SEARCH_BRUTE_FORCE].hit_rec.getRawPtr();

        return this->_truncated_knn(pQueries, knn, pIndices, pDists);;
    }

    float_t bvh_index::_truncated_knn(
        std::vector<query_t>& pQueries,
        uint32_t knn,
        std::vector<int32_t>& pIndices,
        std::vector<float_t>& pDists)
    {
        cudaBuffer<query_t> queries;
        cudaBuffer<int32_t> indices;
        cudaBuffer<float_t> dists;

        queries.alloc(pQueries.size());
        queries.copyHostToDevice(pQueries.data());

        float_t maxf = std::numeric_limits<float_t>::max();

        indices.alloc(pQueries.size() * knn, 0xFFFFFFFF);

        dists.alloc(pQueries.size() * knn,
            reinterpret_cast<int32_t&>(const_cast<float_t&>(maxf)));

        this->mParams.knn = knn;
        this->mParams.optixIndices = indices.getPtr();
        this->mParams.optixDists = dists.getPtr();
        this->mParams.queries = queries.getPtr();
        this->mDeviceParams.copyHostToDevice(&this->mParams, 1);

        cudaTimer timer;

        timer.start();
        OPTIX_CHECK(optixLaunch(
            this->mPipeline,
            this->mCudaStream,
            this->mDeviceParams.getRawPtr(),
            sizeof(Params),
            &this->mSBT,
            pQueries.size(), 1, 1));
        CUDA_SYNC_CHECK();
        float_t elapsed_ms = timer.stop();

        // the neighbor indices
        pIndices = indices.getData();
        // the neighbor distances
        pDists = dists.getData();
        // the queries themselves
        pQueries = queries.getData();

        return elapsed_ms;
    }
}
