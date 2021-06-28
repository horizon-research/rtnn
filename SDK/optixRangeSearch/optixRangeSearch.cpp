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

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <GLFW/glfw3.h>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <string>
#include <random>
#include <cstdlib>

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

    float3*                     d_spheres                 = nullptr;
    float3*                     points = nullptr;

    OptixShaderBindingTable     sbt                       = {};
};

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void sortQueries( thrust::host_vector<unsigned int>*, thrust::host_vector<unsigned int>*, thrust::device_vector<unsigned int>*, thrust::device_vector<unsigned int>* );

float3* read_pc_data(const char* data_file, unsigned int* N) {
  std::ifstream file;

  file.open(data_file);
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
  *N = lines;

  float3* points = new float3[lines];

  bool isShuffle = true;

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
    std::cerr << "Options: --file | -f <filename>      File for point cloud input\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --radius | -r               Search radius\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

void initLaunchParams( WhittedState& state )
{
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped
    state.params.d_vec_val = nullptr;
    state.params.d_vec_key = nullptr;

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
        //0,
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
        OPTIX_CHECK( optixAccelCompact( state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createGeometry( WhittedState &state )
{
    //
    // Allocate device memory for the spheres (points/queries)
    //

    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>(&state.d_spheres),
        state.params.numPrims * sizeof(float3) ) );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( state.d_spheres),
        state.points,
        state.params.numPrims * sizeof(float3),
        cudaMemcpyHostToDevice
    ) );
    state.params.points = state.d_spheres;

    //
    // Build Custom Primitives
    //

    // Load AABB into device memory
    OptixAabb* aabb = (OptixAabb*)malloc(state.params.numPrims * sizeof(OptixAabb));
    CUdeviceptr d_aabb;

    for(unsigned int i = 0; i < state.params.numPrims; i++) {
      sphere_bound(
          state.points[i], state.params.radius,
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
        //rg_sbt.data.spheres = state.d_spheres;

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

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

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

//
//
//

void launchSubframe( sutil::CUDAOutputBuffer<unsigned int>& output_buffer, WhittedState& state )
{
    // this map() thing basically returns the cudaMalloc-ed device pointer.
    unsigned int* result_buffer_data = output_buffer.map();

    // need to manually set the cuda-malloced device memory. note the semantics
    // of cudamemset: it sets #count number of BYTES to value; literally think
    // about what each byte has to be.
    CUDA_CHECK( cudaMemset ( result_buffer_data, 0xFF,
                             state.params.numPrims * state.params.limit * sizeof(unsigned int) ) );
    state.params.frame_buffer = result_buffer_data;

    state.params.queries = state.params.points;
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
        //state.params.numPrims/2, // launch width
        1,                     // launch height
        1                      // launch depth
    ) );
    //output_buffer.unmap();
    //CUDA_SYNC_CHECK();

    //state.params.frame_buffer += state.params.numPrims * state.params.knn / 2;
    //state.params.queries += state.params.numPrims / 2;
    //CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ),
    //                             &state.params,
    //                             sizeof( Params ),
    //                             cudaMemcpyHostToDevice,
    //                             state.stream
    //) );

    //OPTIX_CHECK( optixLaunch(
    //    state.pipeline,
    //    state.stream,
    //    reinterpret_cast<CUdeviceptr>( state.d_params ),
    //    sizeof( Params ),
    //    &state.sbt,
    //    state.params.numPrims -  state.params.numPrims/2, // launch width
    //    1,                     // launch height
    //    1                      // launch depth
    //) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
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
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_spheres              ) ) );
}

void sanityCheck( WhittedState& state, void* data ) {
  // this is stateful in that it relies on state.params.limit

  unsigned int totalNeighbors = 0;
  unsigned int totalWrongNeighbors = 0;
  double totalWrongDist = 0;
  for (unsigned int i = 0; i < state.params.numPrims; i++) {
    for (unsigned int j = 0; j < state.params.limit; j++) {
      unsigned int p = reinterpret_cast<unsigned int*>( data )[ i * state.params.limit + j ];
      //std::cout << p << std::endl; break;
      if (p == UINT_MAX) break;
      else {
        totalNeighbors++;
        float3 diff = state.points[p] - state.points[i];
        float dists = dot(diff, diff);
        if (dists > state.params.radius*state.params.radius) {
          //fprintf(stdout, "Point %u [%f, %f, %f] is not a neighbor of query %u [%f, %f, %f]. Dist is %lf.\n", p, state.points[p].x, state.points[p].y, state.points[p].z, i, state.points[i].x, state.points[i].y, state.points[i].z, sqrt(dists));
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
  std::cerr << "Avg neighbor/query: " << totalNeighbors/state.params.numPrims << std::endl;
  std::cerr << "Avg wrong neighbor/query: " << totalWrongNeighbors/state.params.numPrims << std::endl;
  if (totalWrongNeighbors != 0) std::cerr << "Avg wrong dist: " << totalWrongDist / totalWrongNeighbors << std::endl;
}

int main( int argc, char* argv[] )
{
    WhittedState state;
    // will be overwritten if set explicitly
    state.params.radius = 2;
    state.params.knn = 50;
    std::string outfile;

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
            outfile = argv[++i];
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
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    // read points
    state.points = read_pc_data(outfile.c_str(), &state.params.numPrims);
    std::cerr << "numPrims: " << state.params.numPrims << std::endl;
    std::cerr << "radius: " << state.params.radius << std::endl;
    std::cerr << "K: " << state.params.knn << std::endl;

    Timing::reset();

    try
    {
        //
        // Set up CUDA device and stream
        //
        int32_t device_count = 0;
        CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
        std::cerr << "Total GPUs visible: " << device_count << std::endl;
  
        int32_t device_id = 1;
        cudaDeviceProp prop;
        CUDA_CHECK( cudaGetDeviceProperties ( &prop, device_id ) );
        CUDA_CHECK( cudaSetDevice( device_id ) );
        std::cerr << "\t[" << device_id << "]: " << prop.name << std::endl;

        CUDA_CHECK( cudaStreamCreate( &state.stream ) );

        //
        // Set up OptiX state
        //
        initLaunchParams( state );

        Timing::startTiming("create Context");
        createContext  ( state );
        Timing::stopTiming(true);

        Timing::startTiming("create Geometry");
        createGeometry ( state );
        Timing::stopTiming(true);

        Timing::startTiming("create Pipeline");
        createPipeline ( state );
        Timing::stopTiming(true);

        Timing::startTiming("create SBT");
        createSBT      ( state );
        Timing::stopTiming(true);

        //
        // Do the work
        //

        // Free of CUDAOutputBuffer is done in the destructor.
        sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        Timing::startTiming("compute");
        state.params.limit = 1;
        sutil::CUDAOutputBuffer<unsigned int> output_buffer(
                output_buffer_type,
                state.params.numPrims * state.params.limit,
                1,
                device_id
                );
        launchSubframe( output_buffer, state );
        Timing::stopTiming(true);

        //
        // Check the work
        //

        Timing::startTiming("initial neighbor copy D2H");
        void* data = output_buffer.getHostPointer();
        Timing::stopTiming(true);

        sanityCheck( state, data );

        Timing::startTiming("sort queries");
        thrust::host_vector<unsigned int> h_vec_key(state.params.numPrims);
        thrust::host_vector<unsigned int> h_vec_val(state.params.numPrims);
        for (unsigned int i = 0; i < state.params.numPrims; i++) {
          unsigned int p = reinterpret_cast<unsigned int*>( data )[ i * state.params.limit ];
          h_vec_key[i] = p;
        }
        thrust::sequence(h_vec_val.begin(), h_vec_val.end());
        thrust::device_vector<unsigned int> d_vec_key = h_vec_key;
        thrust::device_vector<unsigned int> d_vec_val = h_vec_val;

        sortQueries( &h_vec_key, &h_vec_val, &d_vec_key, &d_vec_val );
        Timing::stopTiming(true);

        //for (unsigned int i = 0; i < h_vec_key.size(); i++) {
        //  std::cout << h_vec_val[i] << std::endl;
        //}

	// thrust can't be used in kernel code since NVRTC supports only a
	// limited subset of C++, so we would have to explicitly cast a thrust
	// device vector to its raw pointer. See the problem discussed here:
	// https://github.com/cupy/cupy/issues/3728 and
	// https://github.com/cupy/cupy/issues/3408. See how cuNSearch does it:
	// https://github.com/InteractiveComputerGraphics/cuNSearch/blob/master/src/cuNSearchDeviceData.cu#L152
        state.params.d_vec_key = thrust::raw_pointer_cast(&d_vec_key[0]);
        state.params.d_vec_val = thrust::raw_pointer_cast(&d_vec_val[0]);

        Timing::startTiming("second compute");
        state.params.limit = state.params.knn;
        sutil::CUDAOutputBuffer<unsigned int> output_buffer_2(
                output_buffer_type,
                state.params.numPrims * state.params.limit,
                1,
                device_id
                );
        launchSubframe( output_buffer_2, state );
        Timing::stopTiming(true);

        Timing::startTiming("Neighbor copy D2H");
        data = output_buffer_2.getHostPointer();
        Timing::stopTiming(true);

        sanityCheck( state, data );

        cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
