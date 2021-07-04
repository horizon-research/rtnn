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

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <sutil/Timing.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include <GLFW/glfw3.h>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <string>
#include <random>
#include <cstdlib>

// the SDK cmake defines NDEBUG in the Release build, but we still want to use assert
#undef NDEBUG
#include <assert.h>

#include "optixRangeSearch.h"


//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

const int         max_trace = 12;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

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

const uint32_t OBJ_COUNT = 4;

struct WhittedState
{
    OptixDeviceContext          context                   = 0;
    OptixTraversableHandle      gas_handle                = {};
    CUdeviceptr                 d_gas_output_buffer       = {};

    OptixModule                 geometry_module           = 0;
    OptixModule                 camera_module             = 0;

    OptixProgramGroup           raygen_prog_group         = 0;
    OptixProgramGroup           radiance_miss_prog_group  = 0;
    OptixProgramGroup           radiance_metal_sphere_prog_group  = 0;

    OptixPipeline               pipeline                  = 0;
    OptixPipelineCompileOptions pipeline_compile_options  = {};

    CUstream                    stream                    = 0;
    Params                      params;
    Params*                     d_params                  = nullptr;

    float*                      d_key                     = nullptr;
    float3*                     h_points                  = nullptr;
    float3*                     h_queries                 = nullptr;
    unsigned int*               d_r2q_map                 = nullptr;

    std::string                 outfile;
    unsigned int                sortMode                  = 2;
    int                         preSortMode               = 1; // morton order
    float                       crRatio                   = 8; // celSize = radius / crRatio
    float                       sortingGAS                = 1;
    bool                        toGather                  = false;
    bool                        reorderPoints             = false;
    bool                        isShuffle                 = false;

    float3                      Min;
    float3                      Max;

    OptixShaderBindingTable     sbt                       = {};
};

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void sortByKey( thrust::device_vector<float>*, thrust::device_vector<unsigned int>* );
void sortByKey( thrust::device_vector<float>*, thrust::device_ptr<unsigned int>);
void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<unsigned int>, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_vector<unsigned int>*, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int>, unsigned int );
void sortByKey( thrust::device_vector<float>*, thrust::device_ptr<float3> );
void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<float3>, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, unsigned int );
void gatherByKey ( thrust::device_vector<unsigned int>*, thrust::device_ptr<float3>, thrust::device_ptr<float3> );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, thrust::device_ptr<float3>, unsigned int );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_vector<float>*, thrust::device_ptr<float>, unsigned int );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float>, thrust::device_ptr<float>, unsigned int );
thrust::device_ptr<unsigned int> getThrustDevicePtr(unsigned int);
thrust::device_ptr<unsigned int> genSeqDevice(unsigned int);
void exclusiveScan(thrust::device_ptr<unsigned int>, unsigned int, thrust::device_ptr<unsigned int>);
void fillByValue(thrust::device_ptr<unsigned int>, unsigned int, int);

void kComputeMinMax (unsigned int, unsigned int, float3*, unsigned int, float, int3*, int3*);
void kInsertParticles_Morton(unsigned int, unsigned int, GridInfo, float3*, unsigned int*, unsigned int*, unsigned int*, bool);
void kCountingSortIndices(unsigned int, unsigned int, GridInfo, unsigned int*, unsigned int*, unsigned int*, unsigned int*);
void computeMinMax(WhittedState&);
void gridSort(WhittedState&, bool);

float3* read_pc_data(const char* data_file, unsigned int* N, bool isShuffle ) {
  std::ifstream file;

  file.open(data_file);
  if( !file.good() ) {
    std::cerr << "Could not read the frame data...\n";
    assert(0);
  }

  char line[1024];
  unsigned int lines = 0;

  while (file.getline(line, 1024)) {
    lines++;
  }
  file.clear();
  file.seekg(0, std::ios::beg);
  *N = lines;

  float3* points = new float3[lines];

  if (isShuffle) {
    std::vector<float3> vpoints;

    while (file.getline(line, 1024)) {
      double x, y, z;

      sscanf(line, "%lf,%lf,%lf\n", &x, &y, &z);
      vpoints.push_back(make_float3(x, y, z));
    }
    unsigned seed = std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count();
    std::shuffle(std::begin(vpoints), std::end(vpoints), std::default_random_engine(seed));

    unsigned int i = 0;
    for (std::vector<float3>::iterator it = vpoints.begin(); it != vpoints.end(); it++) {
      points[i++] = *it;
    }
  } else {
    std::vector<float3> vpoints;
    lines = 0;
    while (file.getline(line, 1024)) {
      double x, y, z;

      sscanf(line, "%lf,%lf,%lf\n", &x, &y, &z);
      points[lines] = make_float3(x, y, z);
      //std::cerr << points[lines].x << ", " << points[lines].y << ", " << points[lines].z << std::endl;
      lines++;
    }
  }

  file.close();

  return points;
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for point cloud input\n";
    std::cerr << "         --radius | -r               Search radius\n";
    std::cerr << "         --knn | -k                  Max K returned\n";
    std::cerr << "         --sort | -s                 Sort mode\n";
    std::cerr << "         --presort | -ps             preSort mode\n";
    std::cerr << "         --crratio| -cr              cell/radius ratio\n";
    std::cerr << "         --sortingGAS | -sg          Param for SortingGAS\n";
    std::cerr << "         --gather | -g               Whether to gather queries after sort \n";
    std::cerr << "         --reorderpoints | -rp       Whether to reorder points after query sort \n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

void oneDSort ( WhittedState& state ) {
  // pre sort queries based on point coordinates (x/y/z)
  unsigned int N = state.params.numPrims;

  // TODO: do this whole thing on GPU.
  // create 1d points as the sorting key and upload it to device memory
  thrust::host_vector<float> h_key(N);
  for(unsigned int i = 0; i < N; i++) {
    h_key[i] = state.h_queries[i].x;
  }

  float* d_key = nullptr;
  CUDA_CHECK( cudaMalloc(
      reinterpret_cast<void**>(&d_key),
      N * sizeof(float) ) );
  thrust::device_ptr<float> d_key_ptr = thrust::device_pointer_cast(d_key);
  thrust::copy(h_key.begin(), h_key.end(), d_key_ptr);

  // actual sort
  thrust::device_ptr<float3> d_queries_ptr = thrust::device_pointer_cast(state.params.queries);
  sortByKey( d_key_ptr, d_queries_ptr, N );
  CUDA_CHECK( cudaFree( (void*)d_key ) );

  // TODO: lift it outside of this function and combine with other sorts?
  // copy the sorted queries to host so that we build the GAS in the same order
  // note that the h_queries at this point still point to what h_points points to
  thrust::copy(d_queries_ptr, d_queries_ptr + N, state.h_queries);
}

void uploadPoints ( WhittedState& state ) {
  // Allocate device memory for points/queries
  // Optionally sort the points/queries

  CUDA_CHECK( cudaMalloc(
      reinterpret_cast<void**>( &state.params.points ),
      state.params.numPrims * sizeof(float3) ) );
  
  CUDA_CHECK( cudaMemcpyAsync(
      reinterpret_cast<void*>( state.params.points ),
      state.h_points,
      state.params.numPrims * sizeof(float3),
      cudaMemcpyHostToDevice,
      state.stream
  ) );

  // by default, params.queries and params.points point to the same device
  // memory. later if we decide to reorder the queries, we will allocate new
  // space in device memory and point params.queries to that space. this is
  // lazy query allocation.
  state.params.queries = state.params.points;

  // below is what we need to do if queries and points are separate, or if we
  // want to presort points not the queries.
  //state.h_queries = (float3*)malloc(state.params.numPrims * sizeof(float3));
  //std::copy(state.h_points, state.h_points + state.params.numPrims, state.h_queries);
  //CUDA_CHECK( cudaMalloc(
  //    reinterpret_cast<void**>( &state.params.queries),
  //    state.params.numPrims * sizeof(float3) ) );
  //
  //CUDA_CHECK( cudaMemcpyAsync(
  //    reinterpret_cast<void*>( state.params.queries ),
  //    state.h_queries,
  //    state.params.numPrims * sizeof(float3),
  //    cudaMemcpyHostToDevice,
  //    state.stream
  //) );
}

void initLaunchParams( WhittedState& state )
{
    state.params.frame_buffer = nullptr; // the result buffer
    state.params.d_r2q_map = nullptr; // contains the index to reorder rays

    state.params.max_depth = max_trace;
    state.params.scene_epsilon = 1.e-4f;
}

static void sphere_bound(float3 center, float radius, float result[6])
{
    OptixAabb *aabb = reinterpret_cast<OptixAabb*>(result);

    float3 m_min = center - radius;
    float3 m_max = center + radius;

    *aabb = {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

static void buildGas(
    const WhittedState &state,
    const OptixAccelBuildOptions &accel_options,
    const OptixBuildInput &build_input,
    OptixTraversableHandle &gas_handle,
    CUdeviceptr &d_gas_output_buffer
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

    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        state.stream,
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

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        //OPTIX_CHECK( optixAccelCompact( state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );
        OPTIX_CHECK( optixAccelCompact( state.context, state.stream, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createSortingGeometry( WhittedState &state, float sortingGAS )
{
    // TODO: we might want to create a simple GAS for sorting queries, but need
    // to weight the trade-off between the overhead of creating the GAS and the
    // time saved from traversing a simpler GAS. Right now seems like creating
    // geometry is quite heavy, whereas the inital traversal (for sorting) is
    // quite lightweight.

    // Build Custom Primitives

    unsigned int numPrims = state.params.numPrims;
    // Load AABB into device memory
    OptixAabb* aabb = (OptixAabb*)malloc(numPrims * sizeof(OptixAabb));
    CUdeviceptr d_aabb;

    float newRadius = state.params.radius/sortingGAS;
    for(unsigned int i = 0; i < numPrims; i++) {
      sphere_bound(
          state.h_points[i], newRadius,
          reinterpret_cast<float*>(&aabb[i]));
    }

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
        ), numPrims* sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( d_aabb ),
                aabb,
                numPrims * sizeof( OptixAabb ),
                cudaMemcpyHostToDevice,
                state.stream
    ) );

    // Setup AABB build input
    uint32_t* aabb_input_flags = (uint32_t*)malloc(numPrims * sizeof(uint32_t));

    for (unsigned int i = 0; i < numPrims; i++) {
      //aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
      aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
    }

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
        state.gas_handle,
        state.d_gas_output_buffer);

    CUDA_CHECK( cudaFree( (void*)d_aabb) );
}

void createGeometry( WhittedState &state )
{
    //
    // Build Custom Primitives
    //

    // Load AABB into device memory
    OptixAabb* aabb = (OptixAabb*)malloc(state.params.numPrims * sizeof(OptixAabb));
    CUdeviceptr d_aabb;

    for(unsigned int i = 0; i < state.params.numPrims; i++) {
      sphere_bound(
          state.h_points[i], state.params.radius,
          reinterpret_cast<float*>(&aabb[i]));
    }

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
        ), state.params.numPrims* sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( d_aabb ),
                aabb,
                state.params.numPrims * sizeof( OptixAabb ),
                cudaMemcpyHostToDevice,
                state.stream
    ) );

    // Setup AABB build input
    uint32_t* aabb_input_flags = (uint32_t*)malloc(state.params.numPrims * sizeof(uint32_t));

    for (unsigned int i = 0; i < state.params.numPrims; i++) {
      //aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
      aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
    }

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;
    aabb_input.customPrimitiveArray.numPrimitives = state.params.numPrims;
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
        state.gas_handle,
        state.d_gas_output_buffer);

    CUDA_CHECK( cudaFree( (void*)d_aabb) );
}

void createModules( WhittedState &state )
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

static void createCameraProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           cam_prog_group;
    OptixProgramGroupOptions    cam_prog_group_options = {};
    OptixProgramGroupDesc       cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole_camera";

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

static void createMetalSphereProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    char    log[2048];
    size_t  sizeof_log = sizeof( log );

    OptixProgramGroup           radiance_sphere_prog_group;
    OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
    radiance_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    radiance_sphere_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
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

static void createMissProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
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

void createPipeline( WhittedState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
        2,                                                      // numPayloadValues
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
    OPTIX_CHECK_LOG( optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        static_cast<unsigned int>( program_groups.size() ),
        log,
        &sizeof_log,
        &state.pipeline ) );

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
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}

void createSBT( WhittedState &state )
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

void createContext( WhittedState& state )
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

void launchSubframe( sutil::CUDAOutputBuffer<unsigned int>& output_buffer, WhittedState& state )
{
    unsigned int* result_buffer_data = output_buffer.getDevicePointer();
    state.params.frame_buffer = result_buffer_data;

    // need to manually set the cuda-malloced device memory. note the semantics
    // of cudamemset: it sets #count number of BYTES to value; literally think
    // about what each byte has to be.
    CUDA_CHECK( cudaMemsetAsync ( result_buffer_data, 0xFF,
                                  state.params.numPrims * state.params.limit * sizeof(unsigned int),
                                  state.stream ) );

    state.params.handle = state.gas_handle;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ),
                                 &state.params,
                                 sizeof( Params ),
                                 cudaMemcpyHostToDevice,
                                 state.stream
    ) );

    OPTIX_CHECK( optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>( state.d_params ),
        sizeof( Params ),
        &state.sbt,
        state.params.numPrims, // launch width
        1,                     // launch height
        1                      // launch depth
    ) );
    //output_buffer.unmap();
    // TODO: quick hack; if the first traversal, will sync stream before sort
    //if (state.params.limit != 1) CUDA_CHECK( cudaStreamSynchronize( state.stream ) );
    //CUDA_SYNC_CHECK();
}


void cleanupState( WhittedState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.camera_module           ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer    ) ) );
    if (state.d_r2q_map)
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_r2q_map     ) ) );
    if (state.params.queries != state.params.points)
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.queries       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.points          ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
    if (state.d_key)
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_key                ) ) );

    if (state.h_queries != state.h_points) delete state.h_queries;
    delete state.h_points;
}

void sanityCheck( WhittedState& state, void* data ) {
  // this is stateful in that it relies on state.params.limit

  unsigned int totalNeighbors = 0;
  unsigned int totalWrongNeighbors = 0;
  double totalWrongDist = 0;
  for (unsigned int q = 0; q < state.params.numPrims; q++) {
    for (unsigned int n = 0; n < state.params.limit; n++) {
      unsigned int p = reinterpret_cast<unsigned int*>( data )[ q * state.params.limit + n ];
      //std::cout << p << std::endl; break;
      if (p == UINT_MAX) break;
      else {
        totalNeighbors++;
        float3 diff = state.h_points[p] - state.h_queries[q];
        float dists = dot(diff, diff);
        if (dists > state.params.radius*state.params.radius) {
          //fprintf(stdout, "Point %u [%f, %f, %f] is not a neighbor of query %u [%f, %f, %f]. Dist is %lf.\n", p, state.h_points[p].x, state.h_points[p].y, state.h_points[p].z, q, state.h_points[q].x, state.h_points[q].y, state.h_points[q].z, sqrt(dists));
          totalWrongNeighbors++;
          totalWrongDist += sqrt(dists);
          //exit(1);
        }
      }
      //std::cout << p << " ";
    }
    //std::cout << "\n";
  }
  std::cerr << "Sanity check done." << std::endl;
  std::cerr << "Avg neighbor/query: " << (float)totalNeighbors/state.params.numPrims << std::endl;
  std::cerr << "Total wrong neighbors: " << totalWrongNeighbors << std::endl;
  if (totalWrongNeighbors != 0) std::cerr << "Avg wrong dist: " << totalWrongDist / totalWrongNeighbors << std::endl;
}

thrust::device_ptr<unsigned int> sortQueriesByFHCoord( WhittedState& state, thrust::device_ptr<unsigned int> d_firsthit_idx_ptr ) {
  // this is sorting queries by the x/y/z coordinate of the first hit primitives.

  Timing::startTiming("sort queries init");
    // allocate device memory for storing the keys, which will be generated by a gather and used in sort_by_keys
    float* d_key;
    cudaMalloc(reinterpret_cast<void**>(&d_key),
               state.params.numPrims * sizeof(float) );
    thrust::device_ptr<float> d_key_ptr = thrust::device_pointer_cast(d_key);
    state.d_key = d_key; // just so we have a handle to free it later
  
    // create indices for gather and upload to device
    thrust::host_vector<float> h_orig_points_1d(state.params.numPrims);
    // TODO: need to optimize this...
    for (unsigned int i = 0; i < state.params.numPrims; i++) {
      h_orig_points_1d[i] = state.h_points[i].z; // could be other dimensions
    }
    thrust::device_vector<float> d_orig_points_1d = h_orig_points_1d;

    // initialize a sequence to be sorted, which will become the r2q map
    thrust::device_ptr<unsigned int> d_r2q_map_ptr = genSeqDevice(state.params.numPrims);
    // TODO: need to free this
  Timing::stopTiming(true);
  
  Timing::startTiming("sort queries");
    //CUDA_CHECK( cudaStreamSynchronize( state.stream ) );
    // TODO: do thrust work in a stream: https://forums.developer.nvidia.com/t/thrust-and-streams/53199

    // first use a gather to generate the keys, then sort by keys
    gatherByKey(d_firsthit_idx_ptr, &d_orig_points_1d, d_key_ptr, state.params.numPrims);
    sortByKey( d_key_ptr, d_r2q_map_ptr, state.params.numPrims );
    state.d_r2q_map = thrust::raw_pointer_cast(d_r2q_map_ptr);
  Timing::stopTiming(true);
  
  // if debug, copy the sorted keys and values back to host
  bool debug = false;
  if (debug) {
    thrust::host_vector<unsigned int> h_vec_val(state.params.numPrims);
    thrust::copy(d_r2q_map_ptr, d_r2q_map_ptr+state.params.numPrims, h_vec_val.begin());

    thrust::host_vector<float> h_vec_key(state.params.numPrims);
    thrust::copy(d_key_ptr, d_key_ptr+state.params.numPrims, h_vec_key.begin());
    
    for (unsigned int i = 0; i < h_vec_val.size(); i++) {
      std::cout << h_vec_key[i] << "\t" 
                << h_vec_val[i] << "\t" 
                << state.h_points[h_vec_val[i]].x << "\t"
                << state.h_points[h_vec_val[i]].y << "\t"
                << state.h_points[h_vec_val[i]].z
                << std::endl;
    }
  }

  return d_r2q_map_ptr;
}

thrust::device_ptr<unsigned int> sortQueriesByFHIdx( WhittedState& state, thrust::device_ptr<unsigned int> d_firsthit_idx_ptr ) {
  // this is sorting queries just by the first hit primitive IDs

  // initialize a sequence to be sorted, which will become the r2q map
  Timing::startTiming("sort queries init");
    thrust::device_ptr<unsigned int> d_r2q_map_ptr = genSeqDevice(state.params.numPrims);
  Timing::stopTiming(true);

  Timing::startTiming("sort queries");
    sortByKey( d_firsthit_idx_ptr, d_r2q_map_ptr, state.params.numPrims );
    // thrust can't be used in kernel code since NVRTC supports only a
    // limited subset of C++, so we would have to explicitly cast a
    // thrust device vector to its raw pointer. See the problem discussed
    // here: https://github.com/cupy/cupy/issues/3728 and
    // https://github.com/cupy/cupy/issues/3408. See how cuNSearch does it:
    // https://github.com/InteractiveComputerGraphics/cuNSearch/blob/master/src/cuNSearchDeviceData.cu#L152
    state.d_r2q_map = thrust::raw_pointer_cast(d_r2q_map_ptr);
  Timing::stopTiming(true);

  bool debug = false;
  if (debug) {
    thrust::host_vector<unsigned int> h_vec_val(state.params.numPrims);
    thrust::copy(d_r2q_map_ptr, d_r2q_map_ptr+state.params.numPrims, h_vec_val.begin());

    thrust::host_vector<unsigned int> h_vec_key(state.params.numPrims);
    thrust::copy(d_firsthit_idx_ptr, d_firsthit_idx_ptr+state.params.numPrims, h_vec_key.begin());

    for (unsigned int i = 0; i < h_vec_val.size(); i++) {
      std::cout << h_vec_key[i] << "\t"
                << h_vec_val[i] << "\t"
                << state.h_points[h_vec_val[i]].x << "\t"
                << state.h_points[h_vec_val[i]].y << "\t"
                << state.h_points[h_vec_val[i]].z
                << std::endl;
    }
  }

  return d_r2q_map_ptr;
}

void gatherQueries( WhittedState& state, thrust::device_ptr<unsigned int> d_indices_ptr ) {
  // Perform a device gather before launching the actual search.

  Timing::startTiming("gather queries");
    // allocate device memory for reordered/gathered queries
    float3* d_reordered_queries;
    cudaMalloc(reinterpret_cast<void**>(&d_reordered_queries),
               state.params.numPrims * sizeof(float3) );
    thrust::device_ptr<float3> d_reord_queries_ptr = thrust::device_pointer_cast(d_reordered_queries);

    // get pointer to original queries in device memory
    thrust::device_ptr<float3> d_orig_queries_ptr = thrust::device_pointer_cast(state.params.queries);

    // gather by key, which is generated by the previous sort
    gatherByKey(d_indices_ptr, d_orig_queries_ptr, d_reord_queries_ptr, state.params.numPrims);

    //state.params.queries = thrust::raw_pointer_cast(&d_reord_queries_ptr[0]);
    state.params.queries = thrust::raw_pointer_cast(d_reord_queries_ptr);
    assert(state.params.points != state.params.queries);
  Timing::stopTiming(true);

  // Copy reordered queries to host for sanity check
  thrust::host_vector<float3> host_reord_queries(state.params.numPrims);
  state.h_queries = new float3[state.params.numPrims]; // don't overwrite h_points
  thrust::copy(d_reord_queries_ptr, d_reord_queries_ptr+state.params.numPrims, state.h_queries);
  assert (state.h_points != state.h_queries);

  if (state.reorderPoints) {
    // will rebuild the GAS later
    state.h_points = state.h_queries;
    CUDA_CHECK( cudaFree( (void*)state.params.points ) );
    state.params.points = state.params.queries;
  }

  bool debug = false;
  if (debug) {
    for (unsigned int i = 0; i < state.params.numPrims; i++) {
      fprintf(stdout, "orig: %f, %f, %f\n", state.h_points[i].x, state.h_points[i].y, state.h_points[i].z);
      fprintf(stdout, "reor: %f, %f, %f\n\n", state.h_queries[i].x, state.h_queries[i].y, state.h_queries[i].z);
    }
  }
}

void parseArgs( WhittedState& state,  int argc, char* argv[] ) {
  for( int i = 1; i < argc; ++i )
  {
      const std::string arg = argv[i];
      if( arg == "--help" || arg == "-h" )
      {
          printUsageAndExit( argv[0] );
      }
      else if( arg == "--file" || arg == "-f" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.outfile = argv[++i];
      }
      else if( arg == "--knn" || arg == "-k" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.params.knn = atoi(argv[++i]);
      }
      else if( arg == "--radius" || arg == "-r" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.params.radius = std::stof(argv[++i]);
      }
      else if( arg == "--sort" || arg == "-s" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.sortMode = atoi(argv[++i]);
      }
      else if( arg == "--presort" || arg == "-ps" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.preSortMode = atoi(argv[++i]);
      }
      else if( arg == "--crratio" || arg == "-cr" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.crRatio = std::stof(argv[++i]);
      }
      else if( arg == "--gather" || arg == "-g" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.toGather = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--reorderpoints" || arg == "-rp" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.reorderPoints = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--sortingGAS" || arg == "-sg" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.sortingGAS = std::stof(argv[++i]);
          if (state.sortingGAS <= 0)
              printUsageAndExit( argv[0] );
      }
      else if( arg == "--shuffle" || arg == "-sf" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.isShuffle = (bool)(atoi(argv[++i]));
      }
      else
      {
          std::cerr << "Unknown option '" << argv[i] << "'\n";
          printUsageAndExit( argv[0] );
      }
  }
}

void setupCUDA( WhittedState& state, int32_t device_id ) {
  int32_t device_count = 0;
  CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
  std::cerr << "Total GPUs visible: " << device_count << std::endl;
  
  cudaDeviceProp prop;
  CUDA_CHECK( cudaGetDeviceProperties ( &prop, device_id ) );
  CUDA_CHECK( cudaSetDevice( device_id ) );
  std::cerr << "\t[" << device_id << "]: " << prop.name << std::endl;

  CUDA_CHECK( cudaStreamCreate( &state.stream ) );
}

void computeMinMax(WhittedState& state)
{
  // TODO: change these to unit3 since gridInfo.GridDimension is unit3
  thrust::host_vector<int3> h_MinMax(2);
  h_MinMax[0] = make_int3(std::numeric_limits<int>().max(), std::numeric_limits<int>().max(), std::numeric_limits<int>().max());
  h_MinMax[1] = make_int3(std::numeric_limits<int>().min(), std::numeric_limits<int>().min(), std::numeric_limits<int>().min());

  thrust::device_vector<int3> d_MinMax = h_MinMax;

  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = state.params.numPrims / threadsPerBlock + 1;
  // compare only the ints since atomicAdd has only int version
  kComputeMinMax(numOfBlocks,
                 threadsPerBlock,
                 state.params.queries,
                 state.params.numPrims,
                 state.params.radius,
                 thrust::raw_pointer_cast(&d_MinMax[0]),
                 thrust::raw_pointer_cast(&d_MinMax[1])
                 );

  h_MinMax = d_MinMax;

  // minCell encloses the scene but maxCell doesn't (floor and int in the kernel) so increment by 1 to enclose the scene.
  int3 minCell = h_MinMax[0];
  int3 maxCell = h_MinMax[1] + make_int3(1, 1, 1);
 
  state.Min.x = minCell.x * state.params.radius;
  state.Min.y = minCell.y * state.params.radius;
  state.Min.z = minCell.z * state.params.radius;
 
  state.Max.x = maxCell.x * state.params.radius;
  state.Max.y = maxCell.y * state.params.radius;
  state.Max.z = maxCell.z * state.params.radius;

  fprintf(stdout, "scene boundary: (%f, %f, %f), (%f, %f, %f)\n", state.Min.x, state.Min.y, state.Min.z, state.Max.x, state.Max.y, state.Max.z);
}

void gridSort(WhittedState& state, bool morton) {
  float3 sceneMin = state.Min;
  float3 sceneMax = state.Max;

  GridInfo gridInfo;
  gridInfo.ParticleCount = state.params.numPrims;
  gridInfo.GridMin = sceneMin;

  float cellSize = state.params.radius/state.crRatio;
  float3 gridSize = sceneMax - sceneMin;
  gridInfo.GridDimension.x = static_cast<unsigned int>(ceilf(gridSize.x / cellSize));
  gridInfo.GridDimension.y = static_cast<unsigned int>(ceilf(gridSize.y / cellSize));
  gridInfo.GridDimension.z = static_cast<unsigned int>(ceilf(gridSize.z / cellSize));

  // Adjust grid size to multiple of cell size
  gridSize.x = gridInfo.GridDimension.x * cellSize;
  gridSize.y = gridInfo.GridDimension.y * cellSize;
  gridSize.z = gridInfo.GridDimension.z * cellSize;

  gridInfo.GridDelta.x = gridInfo.GridDimension.x / gridSize.x;
  gridInfo.GridDelta.y = gridInfo.GridDimension.y / gridSize.y;
  gridInfo.GridDelta.z = gridInfo.GridDimension.z / gridSize.z;

  // TODO: revisit this later. morton code can only be correctly calcuated for a cubic, where each dimension is of the same size;
  gridInfo.meta_grid_dim = std::min({gridInfo.GridDimension.x, gridInfo.GridDimension.y, gridInfo.GridDimension.z});
  gridInfo.meta_grid_size = gridInfo.meta_grid_dim * gridInfo.meta_grid_dim * gridInfo.meta_grid_dim;

  // One meta grid cell contains meta_grid_dim^3 cells. The morton curve is
  // calculated for each metagrid, and the order of metagrid is raster order.
  // So if meta_grid_dim is 1, this is basically the same as raster order
  // across all cells. If meta_grid_dim is the same as GridDimension, this
  // calculates one single morton curve for the entire grid.
  gridInfo.MetaGridDimension.x = static_cast<unsigned int>(ceilf(gridInfo.GridDimension.x / (float)gridInfo.meta_grid_dim));
  gridInfo.MetaGridDimension.y = static_cast<unsigned int>(ceilf(gridInfo.GridDimension.y / (float)gridInfo.meta_grid_dim));
  gridInfo.MetaGridDimension.z = static_cast<unsigned int>(ceilf(gridInfo.GridDimension.z / (float)gridInfo.meta_grid_dim));

  // metagrids will slightly increase the total cells
  unsigned int numberOfCells = (gridInfo.MetaGridDimension.x * gridInfo.MetaGridDimension.y * gridInfo.MetaGridDimension.z) * gridInfo.meta_grid_size;
  fprintf(stdout, "Grid dimension (without meta grids): %u, %u, %u\n", gridInfo.GridDimension.x, gridInfo.GridDimension.y, gridInfo.GridDimension.z);
  fprintf(stdout, "Grid dimension (with meta grids): %u, %u, %u\n", gridInfo.MetaGridDimension.x * gridInfo.meta_grid_dim, gridInfo.MetaGridDimension.y * gridInfo.meta_grid_dim, gridInfo.MetaGridDimension.z * gridInfo.meta_grid_dim);
  fprintf(stdout, "Meta Grid dimension: %u, %u, %u\n", gridInfo.MetaGridDimension.x, gridInfo.MetaGridDimension.y, gridInfo.MetaGridDimension.z);
  fprintf(stdout, "Length of a meta grid: %u\n", gridInfo.meta_grid_dim);
  fprintf(stdout, "Number of cells: %u\n", numberOfCells);
  fprintf(stdout, "Cell size: %f\n", cellSize);
 
  thrust::device_ptr<unsigned int> d_ParticleCellIndices_ptr = getThrustDevicePtr(state.params.numPrims);
  thrust::device_ptr<unsigned int> d_CellParticleCounts_ptr = getThrustDevicePtr(numberOfCells); // this takes a lot of memory
  fillByValue(d_CellParticleCounts_ptr, numberOfCells, 0);
  thrust::device_ptr<unsigned int> d_LocalSortedIndices_ptr = getThrustDevicePtr(state.params.numPrims);

  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = state.params.numPrims / threadsPerBlock + 1;
  kInsertParticles_Morton(numOfBlocks,
                          threadsPerBlock,
                          gridInfo,
                          state.params.queries,
                          thrust::raw_pointer_cast(d_ParticleCellIndices_ptr),
                          thrust::raw_pointer_cast(d_CellParticleCounts_ptr),
                          thrust::raw_pointer_cast(d_LocalSortedIndices_ptr),
                          morton
                          );

  bool debug = false;
  if (debug) {
    thrust::host_vector<unsigned int> temp(numberOfCells);
    thrust::copy(d_CellParticleCounts_ptr, d_CellParticleCounts_ptr + numberOfCells, temp.begin());
    for (unsigned int i = 0; i < numberOfCells; i++) {
      fprintf(stdout, "%u\n", temp[i]);
    }
  }

  thrust::device_ptr<unsigned int> d_CellOffsets_ptr = getThrustDevicePtr(numberOfCells);
  fillByValue(d_CellOffsets_ptr, numberOfCells, 0); // need to initialize it even for exclusive scan
  exclusiveScan(d_CellParticleCounts_ptr, numberOfCells, d_CellOffsets_ptr);

  thrust::device_ptr<unsigned int> d_posInSortedPoints_ptr = getThrustDevicePtr(state.params.numPrims);
  kCountingSortIndices(numOfBlocks,
                       threadsPerBlock,
                       gridInfo,
                       thrust::raw_pointer_cast(d_ParticleCellIndices_ptr),
                       thrust::raw_pointer_cast(d_CellOffsets_ptr),
                       thrust::raw_pointer_cast(d_LocalSortedIndices_ptr),
                       thrust::raw_pointer_cast(d_posInSortedPoints_ptr)
                       );

  // in-place sort; no new device memory is allocated
  sortByKey(d_posInSortedPoints_ptr, thrust::device_pointer_cast(state.params.queries), state.params.numPrims);

  // TODO: do this in a stream
  thrust::device_ptr<float3> d_queries_ptr = thrust::device_pointer_cast(state.params.queries);
  thrust::copy(d_queries_ptr, d_queries_ptr + state.params.numPrims, state.h_queries);
  assert(state.h_points == state.h_queries);

  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_ParticleCellIndices_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_posInSortedPoints_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_CellOffsets_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_LocalSortedIndices_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_CellParticleCounts_ptr) ) );

  debug = false;
  if (debug) {
    thrust::host_vector<uint> temp(state.params.numPrims);
    thrust::copy(d_posInSortedPoints_ptr, d_posInSortedPoints_ptr + state.params.numPrims, temp.begin());
    for (unsigned int i = 0; i < state.params.numPrims; i++) {
      fprintf(stdout, "%u (%f, %f, %f)\n", temp[i], state.h_queries[i].x, state.h_queries[i].y, state.h_queries[i].z);
    }
  }
}

void preSortGPU(WhittedState& state) {
  int preSortMode = state.preSortMode;
  if (preSortMode == 3) {
    oneDSort(state);
  } else {
    computeMinMax(state);

    bool morton; // false for raster order
    if (preSortMode == 1) morton = true;
    else {
      assert(preSortMode == 2);
      morton = false;
    }
    gridSort(state, morton);
  }
}

void preSortCPU(WhittedState& state) {
  // This is only raster sort.
  float3 cpuMin, cpuMax;
  cpuMin = make_float3(std::numeric_limits<float>().max());
  cpuMax = make_float3(std::numeric_limits<float>().min());
  
  float3 *points = state.h_points;
  for (size_t i = 0; i < state.params.numPrims; i++)
  {
    cpuMin.x = std::min(cpuMin.x, points[i].x);
    cpuMin.y = std::min(cpuMin.y, points[i].y);
    cpuMin.z = std::min(cpuMin.z, points[i].z);
    
    cpuMax.x = std::max(cpuMax.x, points[i].x);
    cpuMax.y = std::max(cpuMax.y, points[i].y);
    cpuMax.z = std::max(cpuMax.z, points[i].z);
  }

  //fprintf(stdout, "(%f, %f, %f), (%f, %f, %f)\n", cpuMin.x, cpuMin.y, cpuMin.z, cpuMax.x, cpuMax.y, cpuMax.z);

  float cellSize = state.params.radius/state.crRatio;
  uint3 gridDim = make_uint3((cpuMax.x-cpuMin.x)/cellSize + 1, (cpuMax.y-cpuMin.y)/cellSize + 1, (cpuMax.z-cpuMin.z)/cellSize + 1);
  unsigned int numOfCells = gridDim.x * gridDim.y * gridDim.z;
  fprintf(stdout, "Grid dimension: %u, %u, %u\n", gridDim.x, gridDim.y, gridDim.z);
  fprintf(stdout, "Number of cells: %u\n", numOfCells);
  fprintf(stdout, "Cell size: %f\n", cellSize);

  thrust::host_vector<unsigned int> numOfPointsInEachCell(numOfCells);
  thrust::fill(numOfPointsInEachCell.begin(), numOfPointsInEachCell.end(), 0);

  for (size_t i = 0; i < state.params.numPrims; i++) {
    int cell_x_idx = (points[i].x - cpuMin.x) / cellSize;
    int cell_y_idx = (points[i].y - cpuMin.y) / cellSize;
    int cell_z_idx = (points[i].z - cpuMin.z) / cellSize;

    int cell_idx = (cell_x_idx * gridDim.y + cell_y_idx) * gridDim.z + cell_z_idx;
    numOfPointsInEachCell[cell_idx]++;
  }

  thrust::host_vector<unsigned int> startIdxOfEachCell(numOfCells);
  thrust::fill(startIdxOfEachCell.begin(), startIdxOfEachCell.end(), 0);
  thrust::exclusive_scan(numOfPointsInEachCell.begin(), numOfPointsInEachCell.end(), startIdxOfEachCell.begin());

  thrust::host_vector<unsigned int> curIdxInEachCell(numOfCells);
  thrust::fill(curIdxInEachCell.begin(), curIdxInEachCell.end(), 0);

  float3* h_reord_points = (float3*)malloc(state.params.numPrims* sizeof(float3));

  for (size_t i = 0; i < state.params.numPrims; i++) {
    int cell_x_idx = (points[i].x - cpuMin.x) / cellSize;
    int cell_y_idx = (points[i].y - cpuMin.y) / cellSize;
    int cell_z_idx = (points[i].z - cpuMin.z) / cellSize;

    int cell_idx = (cell_x_idx * gridDim.y + cell_y_idx) * gridDim.z + cell_z_idx;

    int writeIdx = startIdxOfEachCell[cell_idx] + curIdxInEachCell[cell_idx];
    curIdxInEachCell[cell_idx]++;
    h_reord_points[writeIdx] = state.h_points[i];
  }

  delete state.h_points;
  state.h_points = h_reord_points;
  state.h_queries = state.h_points;

  CUDA_CHECK( cudaMemcpyAsync(
      reinterpret_cast<void*>( state.params.points ),
      state.h_points,
      state.params.numPrims * sizeof(float3),
      cudaMemcpyHostToDevice,
      state.stream
  ) );

  bool debug = false;
  if (debug) {
    for (size_t i = 0; i < state.params.numPrims; i++) {
      fprintf(stdout, "%f, %f, %f\n", state.h_points[i].x, state.h_points[i].y, state.h_points[i].z);
    }
  }
}

void uploadPreProcPoints( WhittedState& state ) {
  Timing::startTiming("upload Points");
    uploadPoints ( state );
  Timing::stopTiming(true);

  if (state.preSortMode) {
    Timing::startTiming("presort Points");
      //preSortCPU(state);
      preSortGPU(state);
    Timing::stopTiming(true);
  }
}

void setupOptiX( WhittedState& state ) {
  Timing::startTiming("create Context");
    createContext  ( state );
  Timing::stopTiming(true);

  // creating GAS can be done async with the rest two.
  Timing::startTiming("create and upload Geometry");
    if (!state.sortMode)
      createGeometry ( state );
    else
      createSortingGeometry ( state, state.sortingGAS );
  Timing::stopTiming(true);
 
  Timing::startTiming("create Pipeline");
    createPipeline ( state );
  Timing::stopTiming(true);

  Timing::startTiming("create SBT");
    createSBT      ( state );
  Timing::stopTiming(true);
}

void nonsortedSearch( WhittedState& state, int32_t device_id ) {
  Timing::startTiming("total unsorted time");
    Timing::startTiming("unsorted compute");
      state.params.limit = state.params.knn;
      sutil::CUDAOutputBuffer<unsigned int> output_buffer(
              sutil::CUDAOutputBufferType::CUDA_DEVICE,
              state.params.numPrims * state.params.limit,
              1,
              device_id
              );

      assert(state.h_queries == state.h_points);
      assert(state.params.points == state.params.queries);
      assert(state.params.d_r2q_map == nullptr);
      launchSubframe( output_buffer, state );
      CUDA_CHECK( cudaStreamSynchronize( state.stream ) ); // TODO: just so we can measure time
    Timing::stopTiming(true);

    // cudaMallocHost is time consuming; must be hidden behind async launch
    Timing::startTiming("result copy D2H");
      void* data;
      cudaMallocHost(reinterpret_cast<void**>(&data), state.params.numPrims * state.params.limit * sizeof(unsigned int));

      CUDA_CHECK( cudaMemcpyAsync(
                      static_cast<void*>( data ),
                      output_buffer.getDevicePointer(),
                      state.params.numPrims * state.params.limit * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost,
                      state.stream
                      ) );
      CUDA_CHECK( cudaStreamSynchronize( state.stream ) ); // TODO: just so we can measure time
    Timing::stopTiming(true);
  Timing::stopTiming(true);

  sanityCheck( state, data );
  CUDA_CHECK( cudaFreeHost(data) ); // TODO: just so we can measure time
}

void searchTraversal(WhittedState& state, int32_t device_id) {
  Timing::startTiming("total sorted");
    if ( (state.sortingGAS != 1) || (state.reorderPoints) ) {
      Timing::startTiming("create search GAS");
        createGeometry ( state );
        CUDA_CHECK( cudaStreamSynchronize( state.stream ) );
      Timing::stopTiming(true);
    }

    Timing::startTiming("sorted compute");
      state.params.limit = state.params.knn;
      sutil::CUDAOutputBuffer<unsigned int>* output_buffer = 
          new sutil::CUDAOutputBuffer<unsigned int>(
              sutil::CUDAOutputBufferType::CUDA_DEVICE,
              state.params.numPrims * state.params.limit,
              1,
              device_id
              );

      assert(state.params.d_r2q_map == nullptr);
      // TODO: not sure why, but directly assigning state.params.d_r2q_map in sort routines has a huge perf hit.
      if (!state.toGather) state.params.d_r2q_map = state.d_r2q_map;

      launchSubframe( *output_buffer, state );
      CUDA_CHECK( cudaStreamSynchronize( state.stream ) ); // comment this out for e2e measurement.
    Timing::stopTiming(true);

    Timing::startTiming("result copy D2H");
      void* data;
      cudaMallocHost(reinterpret_cast<void**>(&data), state.params.numPrims * state.params.limit * sizeof(unsigned int));

      CUDA_CHECK( cudaMemcpyAsync(
                      static_cast<void*>( data ),
                      output_buffer->getDevicePointer(),
                      state.params.numPrims * state.params.limit * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost,
                      state.stream
                      ) );
      CUDA_CHECK( cudaStreamSynchronize( state.stream ) );
    Timing::stopTiming(true);
  Timing::stopTiming(true);

  sanityCheck( state, data );
  CUDA_CHECK( cudaFreeHost(data) ); // TODO: just so we can measure time

  delete output_buffer; // calls the CUDAOutputBuffer destructor.
}

sutil::CUDAOutputBuffer<unsigned int>* initialTraversal(WhittedState& state, int32_t device_id) {

  Timing::startTiming("initial traversal");
    state.params.limit = 1;
    sutil::CUDAOutputBuffer<unsigned int>* output_buffer = 
        new sutil::CUDAOutputBuffer<unsigned int>(
            sutil::CUDAOutputBufferType::CUDA_DEVICE,
            state.params.numPrims * state.params.limit,
            1,
            device_id
            );

    assert(state.h_points == state.h_queries);
    assert(state.params.points == state.params.queries);
    assert(state.params.d_r2q_map == nullptr);

    launchSubframe( *output_buffer, state );
    // TODO: could delay this until sort, but initial traversal is lightweight anyways
    CUDA_CHECK( cudaStreamSynchronize( state.stream ) );
  Timing::stopTiming(true);

  return output_buffer;
}

int main( int argc, char* argv[] )
{
    WhittedState state;
    state.params.radius = 2;
    state.params.knn = 50;

    parseArgs( state, argc, argv );

    // read points
    state.h_points = read_pc_data(state.outfile.c_str(), &state.params.numPrims, state.isShuffle);

    // will be updated if queries are later sorted. this is primarily used for sanity check
    state.h_queries = state.h_points;

    std::cerr << "========================================" << std::endl;
    std::cerr << "numPrims: " << state.params.numPrims << std::endl;
    std::cerr << "radius: " << state.params.radius << std::endl;
    std::cerr << "K: " << state.params.knn << std::endl;
    std::cerr << "sortMode: " << state.sortMode << std::endl;
    std::cerr << "preSortMode: " << std::boolalpha << state.preSortMode << std::endl;
    std::cerr << "cellRadiusRatio: " << std::boolalpha << state.crRatio << std::endl; // only useful when preSort == 1/2
    std::cerr << "sortingGAS: " << state.sortingGAS << std::endl; // only useful when sortMode != 0
    std::cerr << "Gather? " << std::boolalpha << state.toGather << std::endl;
    std::cerr << "reorderPoints? " << std::boolalpha << state.reorderPoints << std::endl;
    //std::cerr << "Shuffle? " << std::boolalpha << state.isShuffle << std::endl;
    std::cerr << "========================================" << std::endl << std::endl;

    try
    {
        // Set up CUDA device and stream
        int32_t device_id = 1;
        setupCUDA(state, device_id);

        Timing::reset();

        uploadPreProcPoints(state);

        // Set up OptiX state
        setupOptiX(state);

        initLaunchParams( state );

        if (!state.sortMode) {
          nonsortedSearch(state, device_id);
        } else {
          // Initial traversal (to sort the queries)
          sutil::CUDAOutputBuffer<unsigned int>* init_res_buffer = initialTraversal(state, device_id);
          thrust::device_ptr<unsigned int> d_firsthit_idx_ptr = thrust::device_pointer_cast(init_res_buffer->getDevicePointer());
          assert(init_res_buffer != nullptr);

          // Sort the queries
          thrust::device_ptr<unsigned int> d_indices_ptr;
          if (state.sortMode == 1)
            d_indices_ptr = sortQueriesByFHCoord(state, d_firsthit_idx_ptr);
          else if (state.sortMode == 2)
            d_indices_ptr = sortQueriesByFHIdx(state, d_firsthit_idx_ptr);
          else {
            std::cerr << "Wrong sortMode" << std::endl;
            printUsageAndExit( argv[0] );
          }
          delete init_res_buffer; // calls the CUDAOutputBuffer destructor.

	  // Gather the queries according to the query order, which by itself
	  // is not useful, since we access each query only once (in the RG
	  // program) anyways. in reality we see little gain by gathering
	  // queries. but if queries and points point to the same device
	  // memory, gathering queries effectively reordering the points too.
	  // we access points in the IS program (get query origin using the hit
	  // primIdx), and so it would be nice to coalesce memory by reordering
	  // points. but note two things. First, we access only one point and
	  // only once in each IS program and the bulk of memory access is to
	  // the BVH which is out of our control, so better memory coalescing
	  // has less effect that in traditional grid-search. Second, if the
	  // points are presorted using a decent ordering (raster scan or
	  // z-order), this reordering has almost zero effect. empirically, we
	  // get 10% search time reduction for large point clouds. but this
	  // comes at a chilling overhead that we need to rebuild the GAS (to
	  // make sure the ID of a box in GAS is the ID of the sphere in device
	  // memory; otherwise IS program is in correct), which is on the
	  // critical path and whose overhead can't be hidden. so almost always
	  // this optimization leads to performance degradation, both toGather
	  // and reorderPoints are disabled by default.
          if (state.toGather) {
            gatherQueries( state, d_indices_ptr );
          }

          // Actual traversal with sorted queries
          searchTraversal(state, device_id);
        }

        cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
