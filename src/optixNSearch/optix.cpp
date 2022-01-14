//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Timing.h>

#include <iomanip>
#include <cstring>
#include <fstream>
#include <string>
#include <random>
#include <cstdlib>
#include <queue>
#include <unordered_set>

#include "optixNSearch.h"
#include "state.h"
#include "func.h"
#include "grid.h"

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT )

    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<GeomData>        RayGenRecord;
typedef Record<MissData>        MissRecord;
typedef Record<HitGroupData>    HitGroupRecord;

void filterRemoteQueries ( RTNNState& state ) {
  // TODO: another idea is that the grid doesn't HAVE to be union. it just
  // needs to include the query scene, and we can collapse all out-of-boundary
  // points to the edge cells. this would allow us to potentially have
  // finer-grained cells, but edge cells can have lots of points that generate
  // some overly dense partitions.

  if ((state.qMin >= state.pMin) && (state.qMax <= state.pMax)) return;

  float3 tMin = {state.pMin.x - state.radius, state.pMin.y - state.radius, state.pMin.z - state.radius};
  float3 tMax = {state.pMax.x + state.radius, state.pMax.y + state.radius, state.pMax.z + state.radius};

  unsigned int count = countIfInRange(thrust::device_pointer_cast(state.params.queries), state.numQueries, tMin, tMax);
  thrust::device_ptr<float3> tQueries;
  allocThrustDevicePtr(&tQueries, count, &state.d_pointers);
  copyIfInRange(state.params.queries, state.numQueries, thrust::device_pointer_cast(state.params.queries), tQueries, tMin, tMax);
  fprintf(stdout, "Filter queries: %u (%.3f)\n", state.numQueries - count, (1 - (float)count/state.numQueries)*100);

  if (count == 0) {
    fprintf(stdout, "no queries left after filtering\n");
    exit(0);
  }

  // set up filtered queries on the host for sanity check
  if (state.sanCheck) {
    state.numFltQs = state.numQueries - count;
    state.h_fltQs = new float3[state.numFltQs];
    // quite heavy
    copyIfNotInRange(state.h_queries, state.numQueries, state.h_queries, state.h_fltQs, tMin, tMax);
  }

  assert(state.params.points != state.params.queries); // otherwise it's samepq, which wouldn't pass the test earlier
  state.d_pointers.erase(state.d_pointers.find(state.params.queries));
  CUDA_CHECK( cudaFree( state.params.queries ) );

  state.params.queries = thrust::raw_pointer_cast(tQueries);
  state.numQueries = count;
  // Just for sanity check purpose. TODO: Really only needed when there's no
  // partition and no query sorting, both of which will update state.h_queries
  // using the queries from the device.
  if (state.sanCheck) {
    thrust::copy(thrust::device_pointer_cast(state.params.queries),
        thrust::device_pointer_cast(state.params.queries) + state.numQueries, state.h_queries);
  }
  computeMinMax(state.numQueries, state.params.queries, state.qMin, state.qMax);

  state.Min = fminf(state.qMin, state.pMin);
  state.Max = fmaxf(state.qMax, state.pMax);
}

void uploadData ( RTNNState& state ) {
  Timing::startTiming("upload points and/or queries");
    // Allocate device memory for points/queries
    thrust::device_ptr<float3> d_points_ptr;
    state.params.points = allocThrustDevicePtr(&d_points_ptr, state.numPoints, &state.d_pointers);

    thrust::copy(state.h_points, state.h_points + state.numPoints, d_points_ptr);
    computeMinMax(state.numPoints, state.params.points, state.pMin, state.pMax);

    if (state.samepq) {
      // by default, params.queries and params.points point to the same device
      // memory. later if we decide to reorder the queries, we will allocate new
      // space in device memory and point params.queries to that space. this is
      // lazy query allocation.
      state.params.queries = state.params.points;
      state.qMin = state.pMin;
      state.qMax = state.pMax;
    } else {
      thrust::device_ptr<float3> d_queries_ptr;
      state.params.queries = allocThrustDevicePtr(&d_queries_ptr, state.numQueries, &state.d_pointers);
      
      thrust::copy(state.h_queries, state.h_queries + state.numQueries, d_queries_ptr);
      computeMinMax(state.numQueries, state.params.queries, state.qMin, state.qMax);
    }

    Timing::startTiming("filter queries");
      // filter out queries that are theorerically impossible to reach any search
      // points given the search radius, then create a unified grid. why? query
      // partition in theory will need a unified grid anyways, and if without
      // filtering the grid could be overly large such that the cells are big. the
      // unified grid after filtering will have desirable size for partitioning and
      // point sorting. the only concern is for query partitioning, where is query
      // density is much smaller the unified grid could be too coarse-grained.
      // could generate a dedicated query grid for query sorting.

      state.Min = fminf(state.qMin, state.pMin);
      state.Max = fmaxf(state.qMax, state.pMax);

      if (state.filterQueries) filterRemoteQueries(state);
    Timing::stopTiming(true);

  Timing::stopTiming(true);
}

static void buildGas(
    RTNNState &state,
    const OptixAccelBuildOptions &accel_options,
    const OptixBuildInput &build_input,
    OptixTraversableHandle &gas_handle,
    CUdeviceptr &d_gas_output_buffer,
    int batch_id
    )
{
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr d_temp_buffer_gas;

    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes));

    //fprintf(stdout, "\tTemp storage for initial building: %f MB\n", (float)gas_buffer_sizes.tempSizeInBytes/1024/1024);
    // temporary storage for building the initial, non-compacted tree
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS.
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    //fprintf(stdout, "\tNon-compacted GAS size: %f MB\n", (float)(compactedSizeOffset + 8)/1024/1024);
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        state.stream[batch_id],
        &accel_options,
        &build_input,
        1,
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        &emitProperty,
        1) );

    // once the initial tree is built, the temporary storage used for building the tree could be freed
    state.d_temp_buffer_gas[batch_id] = reinterpret_cast<void*>(d_temp_buffer_gas);
    CUDA_CHECK( cudaFree( state.d_temp_buffer_gas[batch_id] ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpyAsync( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost, state.stream[batch_id] ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        // compacted size is smaller, so store the compacted GAS in new device memory and free the original GAS memory/
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, state.stream[batch_id], gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        //state.d_buffer_temp_output_gas_and_compacted_size[batch_id] = (void*)d_buffer_temp_output_gas_and_compacted_size;
        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        // original size is smaller, so point d_gas_output_buffer directly to the original device GAS memory.
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
    fprintf(stdout, "\tFinal GAS size: %f MB\n", (float)compacted_gas_size/(1024 * 1024));
}

CUdeviceptr createAABB( RTNNState& state, int batch_id, float radius )
{
  // Load AABB into device memory
  unsigned int numPrims = state.numPoints;

  //float radius = state.launchRadius[batch_id] / state.gsrRatio;
  //std::cout << "\tAABB radius: " << radius << std::endl;
  //std::cout << "\tnum of points in GAS: " << numPrims << std::endl;

  OptixAabb* d_aabb;
  if (state.d_aabb[batch_id] == nullptr) {
    thrust::device_ptr<OptixAabb> d_aabb_ptr;
    // do not track it so that we can free it early
    d_aabb = allocThrustDevicePtr(&d_aabb_ptr, numPrims);
  } else {
    // if pointers are not null, simply reuse the previously allocated device
    // memory to save memory consumption and/or free overhead. this can happen
    // when |gsrRatio| isn't 1.
    d_aabb = reinterpret_cast<OptixAabb*>(state.d_aabb[batch_id]);
  }

  kGenAABB(state.params.points,
           radius,
           numPrims,
           d_aabb,
           state.stream[batch_id]
          );

  return reinterpret_cast<CUdeviceptr>(d_aabb);
}

void createGeometry( RTNNState& state, int batch_id, float radius )
{
  Timing::startTiming("create and upload geometry");
    CUdeviceptr d_aabb = createAABB(state, batch_id, radius);

    unsigned int numPrims = state.numPoints;

    // Setup AABB build input. Don't disable AH.
    uint32_t aabb_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;
    aabb_input.customPrimitiveArray.numPrimitives = numPrims;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer         = 0;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes    = sizeof( uint32_t );
    aabb_input.customPrimitiveArray.primitiveIndexOffset         = 0;

    OptixAccelBuildOptions accel_options = {
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags
        OPTIX_BUILD_OPERATION_BUILD         // operation
    };

    buildGas(
        state,
        accel_options,
        aabb_input,
        state.gas_handle[batch_id],
        state.d_gas_output_buffer[batch_id],
        batch_id);

    state.d_aabb[batch_id] = reinterpret_cast<void*>(d_aabb);
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>(d_aabb) ) );
    OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) );
  Timing::stopTiming(true);
}

void createModules( RTNNState &state )
{
    OptixModuleCompileOptions module_compile_options = {
        100,                                    // maxRegisterCount
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,     // optLevel
        OPTIX_COMPILE_DEBUG_LEVEL_NONE          // debugLevel
    };
    char log[2048];
    size_t sizeof_log = sizeof(log);

    {
        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "geometry.cu" );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &state.geometry_module ) );
    }

    {
        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "camera.cu" );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &state.camera_module ) );
    }
}

static void createCameraProgram( RTNNState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           cam_prog_group;
    OptixProgramGroupOptions    cam_prog_group_options = {};
    OptixProgramGroupDesc       cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.camera_module;
    if (state.searchMode == "knn")
      cam_prog_group_desc.raygen.entryFunctionName = "__raygen__knn";
    else
      cam_prog_group_desc.raygen.entryFunctionName = "__raygen__radius";

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &cam_prog_group_desc,
        1,
        &cam_prog_group_options,
        log,
        &sizeof_log,
        &cam_prog_group ) );

    program_groups.push_back(cam_prog_group);
    state.raygen_prog_group = cam_prog_group;
}

static void createMetalSphereProgram( RTNNState &state, std::vector<OptixProgramGroup> &program_groups )
{
    char    log[2048];
    size_t  sizeof_log = sizeof( log );

    OptixProgramGroup           radiance_sphere_prog_group;
    OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
    radiance_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    radiance_sphere_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    if (state.searchMode == "knn")
      radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere_knn";
    else
      radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere_radius";
    radiance_sphere_prog_group_desc.hitgroup.moduleCH               = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.moduleAH               = state.geometry_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = "__anyhit__terminateRay";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_sphere_prog_group_desc,
        1,
        &radiance_sphere_prog_group_options,
        log,
        &sizeof_log,
        &radiance_sphere_prog_group ) );

    program_groups.push_back(radiance_sphere_prog_group);
    state.radiance_metal_sphere_prog_group = radiance_sphere_prog_group;
}

static void createMissProgram( RTNNState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroupOptions    miss_prog_group_options = {};
    OptixProgramGroupDesc       miss_prog_group_desc = {};
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module             = nullptr;
    miss_prog_group_desc.miss.entryFunctionName  = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        log,
        &sizeof_log,
        &state.radiance_miss_prog_group ) );

    program_groups.push_back(state.radiance_miss_prog_group);
}

void createPipeline( RTNNState &state )
{
    const int max_trace = 2;

    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
        8,                                                      // numPayloadValues; need 8 for 7nn search
        0,                                                      // numAttributeValues
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createModules( state );
    createCameraProgram( state, program_groups );
    createMetalSphereProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
        max_trace,                          // maxTraceDepth
        OPTIX_COMPILE_DEBUG_LEVEL_FULL      // debugLevel
    };
    char    log[2048];
    size_t  sizeof_log = sizeof(log);

    for (int i = 0; i < state.maxBatchCount; i++) {
      OPTIX_CHECK_LOG( optixPipelineCreate(
          state.context,
          &state.pipeline_compile_options,
          &pipeline_link_options,
          program_groups.data(),
          static_cast<unsigned int>( program_groups.size() ),
          log,
          &sizeof_log,
          &state.pipeline[i] ) );
    }

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace,
                                             0,  // maxCCDepth
                                             0,  // maxDCDepth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    for (int i = 0; i < state.maxBatchCount; i++) {
      OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline[i], direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size,
                                              1  // maxTraversableDepth
                                              ) );
    }
}

void createSBT( RTNNState &state )
{
    // Raygen program record
    {
        CUdeviceptr d_raygen_record;
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_raygen_record ),
            sizeof( RayGenRecord ) ) );

        RayGenRecord rg_sbt;
        optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_raygen_record ),
            &rg_sbt,
            sizeof(rg_sbt),
            cudaMemcpyHostToDevice
        ) );

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof( MissRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_miss_record ),
            sizeof_miss_record*RAY_TYPE_COUNT ) );

        MissRecord ms_sbt;
        optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_miss_record ),
            &ms_sbt,
            sizeof_miss_record,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.missRecordBase          = d_miss_record;
        state.sbt.missRecordCount         = 1;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record
    {
        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_hitgroup_records ),
            sizeof_hitgroup_record
        ) );

        HitGroupRecord hit_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.radiance_metal_sphere_prog_group,
            &hit_sbt ) );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_hitgroup_records ),
            &hit_sbt,
            sizeof_hitgroup_record,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.hitgroupRecordBase            = d_hitgroup_records;
        state.sbt.hitgroupRecordCount           = 1;
        state.sbt.hitgroupRecordStrideInBytes   = static_cast<uint32_t>( sizeof_hitgroup_record );
    }
}

//static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
//{
//    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
//              << message << "\n";
//}

void createContext( RTNNState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    //options.logCallbackFunction       = &context_log_cb;
    options.logCallbackFunction       = nullptr;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}

void launchSubframe( unsigned int* output_buffer, RTNNState& state, int batch_id )
{
    unsigned int numQueries = state.numActQueries[batch_id];
    state.params.handle = state.gas_handle[batch_id];
    state.params.queries = state.d_actQs[batch_id];
    state.params.frame_buffer = output_buffer;

    fprintf(stdout, "\tLaunch %u (%.4f%%) queries\n", numQueries, (float)numQueries/(float)state.numQueries*100.0);
    fprintf(stdout, "\tSearch radius: %f\n", state.params.radius);
    fprintf(stdout, "\tSearch K: %u\n", state.params.limit);
    fprintf(stdout, "\tSearch mode: %d\n", state.params.mode);

    thrust::device_ptr<Params> d_params_ptr;
    state.d_params = allocThrustDevicePtr(&d_params_ptr, 1, &state.d_pointers);
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ),
                                 &state.params,
                                 sizeof( Params ),
                                 cudaMemcpyHostToDevice,
                                 state.stream[batch_id]
    ) );

    OPTIX_CHECK( optixLaunch(
        state.pipeline[batch_id],
        state.stream[batch_id],
        reinterpret_cast<CUdeviceptr>( state.d_params ),
        sizeof( Params ),
        &state.sbt,
        numQueries, // launch width
        1,          // launch height
        1           // launch depth
    ) );
}

void cleanupState( RTNNState& state )
{
    for (int i = 0; i < state.maxBatchCount; i++) {
      OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline[i]           ) );
    }
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.camera_module           ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );

    for (int i = 0; i < state.numOfBatches; i++) {
      if (state.numActQueries[i] == 0) continue;

      CUDA_CHECK( cudaStreamDestroy(state.stream[i]) );

      CUDA_CHECK( cudaFreeHost(state.h_res[i] ) );
      delete state.h_actQs[i];

      //CUDA_CHECK( cudaFree( state.d_temp_buffer_gas[i] ) );
      // if compaction isn't successful, d_gas and d_buffer_temp point will point to the same device memory.
      //if (reinterpret_cast<void*>(state.d_gas_output_buffer[i]) != state.d_buffer_temp_output_gas_and_compacted_size[i] )
      //  CUDA_CHECK( cudaFree( state.d_buffer_temp_output_gas_and_compacted_size[i] ) );
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer[i] ) ) );
    }

    delete state.gas_handle;
    delete state.d_gas_output_buffer;
    delete state.stream;
    delete state.numActQueries;
    delete state.launchRadius;
    delete state.h_res;
    delete state.d_actQs;
    delete state.h_actQs;
    delete state.d_aabb;
    delete state.d_temp_buffer_gas;
    delete state.d_buffer_temp_output_gas_and_compacted_size;
    delete state.d_r2q_map;
    //delete state.h_points;

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );

    for (auto it = state.d_pointers.begin(); it != state.d_pointers.end(); it++) {
      CUDA_CHECK( cudaFree( *it ) );
    }
    if (state.deferFree) freeGridPointers(state);
}

void setupOptiX( RTNNState& state ) {
  Timing::startTiming("create context");
    createContext  ( state );
  Timing::stopTiming(true);
 
  Timing::startTiming("create pipeline");
    createPipeline ( state );
  Timing::stopTiming(true);

  Timing::startTiming("create SBT");
    createSBT      ( state );
  Timing::stopTiming(true);
}

