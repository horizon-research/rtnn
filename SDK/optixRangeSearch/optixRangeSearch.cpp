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

#include <GLFW/glfw3.h>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <string>

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

typedef Record<GeomData>      RayGenRecord;
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
    OptixModule                 shading_module            = 0;

    OptixProgramGroup           raygen_prog_group         = 0;
    OptixProgramGroup           radiance_miss_prog_group  = 0;
    OptixProgramGroup           occlusion_miss_prog_group = 0;
    OptixProgramGroup           radiance_glass_sphere_prog_group  = 0;
    OptixProgramGroup           occlusion_glass_sphere_prog_group = 0;
    OptixProgramGroup           radiance_metal_sphere_prog_group  = 0;
    OptixProgramGroup           occlusion_metal_sphere_prog_group = 0;
    OptixProgramGroup           radiance_floor_prog_group         = 0;
    OptixProgramGroup           occlusion_floor_prog_group        = 0;

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
//  Geometry and Camera data
//
//------------------------------------------------------------------------------

// Metal sphere, glass sphere, floor, light
const Sphere g_sphere1 = {
    { 0.0f, 0.0f, 0.0f }, // center
    2.0f                  // radius
};
const Sphere g_sphere2 = {
    { 5.0f, 0.0f, 0.0f }, // center
    2.0f                  // radius
};
const Sphere g_sphere3 = {
    { 0.0f, 5.0f, 0.0f }, // center
    2.0f                  // radius
};
const Sphere g_sphere4 = {
    { 0.0f, 0.0f, 5.0f }, // center
    2.0f                  // radius
};

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

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
  float3* points = new float3[lines];
  *N = lines;

  lines = 0;
  while (file.getline(line, 1024)) {
    double x, y, z;

    sscanf(line, "%lf,%lf,%lf\n", &x, &y, &z);
    points[lines].x = x;
    points[lines].y = y;
    points[lines].z = z;
    //std::cerr << points[lines].x << "," << points[lines].y << "," << points[lines].z << std::endl;
    lines++;
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

    state.params.max_depth = max_trace;
    state.params.scene_epsilon = 1.e-4f;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );

    state.params.handle = state.gas_handle;
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
        0,
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
    // Build Custom Primitives
    //

    // Load AABB into device memory
    //OptixAabb   aabb[OBJ_COUNT];
    OptixAabb* aabb = (OptixAabb*)malloc(state.params.numPrims * sizeof(OptixAabb));
    CUdeviceptr d_aabb;

    for(unsigned int i = 0; i < state.params.numPrims; i++) {
      sphere_bound(
          state.points[i], state.params.radius,
          reinterpret_cast<float*>(&aabb[i]));
    }

    //std::cerr << aabb[0].minX << "," << aabb[0].minY << "," << aabb[0].minZ << std::endl;
    //std::cerr << aabb[0].maxX << "," << aabb[0].maxY << "," << aabb[0].maxZ << std::endl;
    //std::cerr << aabb[1].minX << "," << aabb[1].minY << "," << aabb[1].minZ << std::endl;
    //std::cerr << aabb[1].maxX << "," << aabb[1].maxY << "," << aabb[1].maxZ << std::endl;
    //std::cerr << aabb[2].minX << "," << aabb[2].minY << "," << aabb[2].minZ << std::endl;
    //std::cerr << aabb[2].maxX << "," << aabb[2].maxY << "," << aabb[2].maxZ << std::endl;

    //sphere_bound(
    //    g_sphere1.center, g_sphere1.radius,
    //    reinterpret_cast<float*>(&aabb[0]));
    //sphere_bound(
    //    g_sphere2.center, g_sphere2.radius,
    //    reinterpret_cast<float*>(&aabb[1]));
    //sphere_bound(
    //    g_sphere3.center, g_sphere3.radius,
    //    reinterpret_cast<float*>(&aabb[2]));
    //sphere_bound(
    //    g_sphere4.center, g_sphere4.radius,
    //    reinterpret_cast<float*>(&aabb[3]));

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
        ), state.params.numPrims* sizeof( OptixAabb ) ) );
        //), OBJ_COUNT * sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_aabb ),
                //&aabb,
                aabb,
                state.params.numPrims * sizeof( OptixAabb ),
                //OBJ_COUNT * sizeof( OptixAabb ),
                cudaMemcpyHostToDevice
                ) );

    // Setup AABB build input
    //uint32_t aabb_input_flags[] = {
    //    /* flags for metal sphere */
    //    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    //    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    //    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    //    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    //};
    uint32_t* aabb_input_flags = (uint32_t*)malloc(state.params.numPrims * sizeof(uint32_t));

    //const uint32_t sbt_index[] = { 0, 1, 2, 3 };
    //const uint32_t sbt_index[4] = { 0 };
    uint32_t* sbt_index = (uint32_t*)malloc(state.params.numPrims * sizeof(uint32_t));
    for (unsigned int i = 0; i < state.params.numPrims; i++) {
      aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
      // it's important to set all these indices to 0.
      sbt_index[i] = 0;
    }
    CUdeviceptr    d_sbt_index;

    //CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), sizeof(sbt_index) ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), state.params.numPrims * sizeof(uint32_t) ) );
    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( d_sbt_index ),
        sbt_index,
        //sizeof( sbt_index ),
        state.params.numPrims * sizeof(uint32_t),
        cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    //aabb_input.customPrimitiveArray.numSbtRecords = OBJ_COUNT;
    //aabb_input.customPrimitiveArray.numSbtRecords = state.params.numPrims;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;
    //aabb_input.customPrimitiveArray.numPrimitives = OBJ_COUNT;
    aabb_input.customPrimitiveArray.numPrimitives = state.params.numPrims;
    //aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer         = d_sbt_index;
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

    {
        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "shading.cu" );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &state.shading_module ) );
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
    radiance_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__metal_radiance";
    radiance_sphere_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

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

    //OptixProgramGroup           occlusion_sphere_prog_group;
    //OptixProgramGroupOptions    occlusion_sphere_prog_group_options = {};
    //OptixProgramGroupDesc       occlusion_sphere_prog_group_desc = {};
    //occlusion_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    //    occlusion_sphere_prog_group_desc.hitgroup.moduleIS           = state.geometry_module;
    //occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
    //occlusion_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    //occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__full_occlusion";
    //occlusion_sphere_prog_group_desc.hitgroup.moduleAH               = nullptr;
    //occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    //OPTIX_CHECK_LOG( optixProgramGroupCreate(
    //    state.context,
    //    &occlusion_sphere_prog_group_desc,
    //    1,
    //    &occlusion_sphere_prog_group_options,
    //    log,
    //    &sizeof_log,
    //    &occlusion_sphere_prog_group ) );

    //program_groups.push_back(occlusion_sphere_prog_group);
    //state.occlusion_metal_sphere_prog_group = occlusion_sphere_prog_group;
}

static void createMissProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroupOptions    miss_prog_group_options = {};
    OptixProgramGroupDesc       miss_prog_group_desc = {};
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module             = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName  = "__miss__constant_bg";

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

    miss_prog_group_desc.miss = {
        nullptr,    // module
        nullptr     // entryFunctionName
    };
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        log,
        &sizeof_log,
        &state.occlusion_miss_prog_group ) );

    program_groups.push_back(state.occlusion_miss_prog_group);
}

void createPipeline( WhittedState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
        5,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
        5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
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
        //std::vector<float3> spheres;
        //for (unsigned int i = 0; i < state.params.numPrims; i++) {
        //  spheres.push_back(state.points[i]);
        //}
        //spheres.push_back(g_sphere1.center);
        //spheres.push_back(g_sphere2.center);
        //spheres.push_back(g_sphere3.center);
        //spheres.push_back(g_sphere4.center);

        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>(&state.d_spheres),
            state.params.numPrims * sizeof(float3) ) );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( state.d_spheres),
            //&spheres[0],
            state.points,
            state.params.numPrims * sizeof(float3),
            cudaMemcpyHostToDevice
        ) );
        state.params.spheres = state.d_spheres;

        RayGenRecord rg_sbt;
        optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt );
        rg_sbt.data.spheres = state.d_spheres;

        CUdeviceptr d_raygen_record;
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_raygen_record ),
            sizeof( RayGenRecord ) ) );

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

        //MissRecord ms_sbt[RAY_TYPE_COUNT];
        //optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt[0] );
        //optixSbtRecordPackHeader( state.occlusion_miss_prog_group, &ms_sbt[1] );
        //ms_sbt[1].data = ms_sbt[0].data = { 0.34f, 0.55f, 0.85f };

        //CUDA_CHECK( cudaMemcpy(
        //    reinterpret_cast<void*>( d_miss_record ),
        //    ms_sbt,
        //    sizeof_miss_record*RAY_TYPE_COUNT,
        //    cudaMemcpyHostToDevice
        //) );

        state.sbt.missRecordBase          = d_miss_record;
        //state.sbt.missRecordCount         = RAY_TYPE_COUNT;
        state.sbt.missRecordCount         = 1;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record
    {
        HitGroupRecord hit_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.radiance_metal_sphere_prog_group,
            &hit_sbt ) );
        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_hitgroup_records ),
            sizeof_hitgroup_record
        ) );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_hitgroup_records ),
            &hit_sbt,
            sizeof_hitgroup_record,
            cudaMemcpyHostToDevice
        ) );

        //const size_t count_records = 1;
        //const size_t count_records = RAY_TYPE_COUNT * OBJ_COUNT;
        //const size_t count_records = RAY_TYPE_COUNT * state.params.numPrims;
        //std::cerr << count_records << std::endl;
        //HitGroupRecord hitgroup_records[count_records];

        // Note: MUST fill SBT record array same order like AS is built. Fill
        // different ray types for a primitive first then move to the next
        // primitive. See the table here:
        // https://raytracing-docs.nvidia.com/optix7/guide/index.html#shader_binding_table#shader-binding-table
        //int sbt_idx = 0;

        // Metal Sphere
        // The correct way of thinking about optixSbtRecordPackHeader is that
        // it assigns a program group to a sbt record, not the other way
        // around. When we find a hit we will have to decide what program to
        // run and what data to use for that program; given the parameters
        // passed into optixTrace, we can calculate the entry of the SBT
        // associated with that ray; that entry will have an assigned program
        // group and the data. The way to calculate the exact entry in the SBT
        // can be found here:
        // https://www.willusher.io/graphics/2019/11/20/the-sbt-three-ways. The
        // key is to think with a SBT-centric mindset: the programs are just
        // attached to the SBT (each SBT entry points to a program that will be
        // executed when we find the entry).
        //OPTIX_CHECK( optixSbtRecordPackHeader(
        //    state.radiance_metal_sphere_prog_group,
        //    &hitgroup_records[sbt_idx] ) );
        //hitgroup_records[ sbt_idx ].data.geometry.sphere = g_sphere1;
        //hitgroup_records[ sbt_idx ].data.shading.metal = {
        //    { 0.2f, 0.5f, 0.5f },   // Ka
        //    { 0.2f, 0.7f, 0.8f },   // Kd
        //    { 0.9f, 0.9f, 0.9f },   // Ks
        //    { 0.5f, 0.5f, 0.5f },   // Kr
        //    64,                     // phong_exp
        //};
        //sbt_idx ++;

        //OPTIX_CHECK( optixSbtRecordPackHeader(
        //    state.occlusion_metal_sphere_prog_group,
        //    &hitgroup_records[sbt_idx] ) );
        //hitgroup_records[ sbt_idx ].data.geometry.sphere = g_sphere1;
        //sbt_idx ++;

        //OPTIX_CHECK( optixSbtRecordPackHeader(
        //    state.radiance_metal_sphere_prog_group,
        //    &hitgroup_records[sbt_idx] ) );
        //hitgroup_records[ sbt_idx ].data.geometry.sphere = g_sphere2;
        //hitgroup_records[ sbt_idx ].data.shading.metal = {
        //    { 0.2f, 0.5f, 0.5f },   // Ka
        //    { 0.2f, 0.7f, 0.8f },   // Kd
        //    { 0.9f, 0.9f, 0.9f },   // Ks
        //    { 0.5f, 0.5f, 0.5f },   // Kr
        //    64,                     // phong_exp
        //};
        //sbt_idx ++;

        //OPTIX_CHECK( optixSbtRecordPackHeader(
        //    state.occlusion_metal_sphere_prog_group,
        //    &hitgroup_records[sbt_idx] ) );
        //hitgroup_records[ sbt_idx ].data.geometry.sphere = g_sphere2;
        //sbt_idx ++;

        // Glass Sphere
        //OPTIX_CHECK( optixSbtRecordPackHeader(
        //    state.radiance_glass_sphere_prog_group,
        //    &hitgroup_records[sbt_idx] ) );
        //hitgroup_records[ sbt_idx ].data.geometry.sphere_shell = g_sphere_shell;
        //hitgroup_records[ sbt_idx ].data.shading.glass = {
        //    1e-2f,                                  // importance_cutoff
        //    { 0.034f, 0.055f, 0.085f },             // cutoff_color
        //    3.0f,                                   // fresnel_exponent
        //    0.1f,                                   // fresnel_minimum
        //    1.0f,                                   // fresnel_maximum
        //    1.4f,                                   // refraction_index
        //    { 1.0f, 1.0f, 1.0f },                   // refraction_color
        //    { 1.0f, 1.0f, 1.0f },                   // reflection_color
        //    { logf(.83f), logf(.83f), logf(.83f) }, // extinction_constant
        //    { 0.6f, 0.6f, 0.6f },                   // shadow_attenuation
        //    10,                                     // refraction_maxdepth
        //    5                                       // reflection_maxdepth
        //};
        //sbt_idx ++;

        //OPTIX_CHECK( optixSbtRecordPackHeader(
        //    state.occlusion_glass_sphere_prog_group,
        //    &hitgroup_records[sbt_idx] ) );
        //hitgroup_records[ sbt_idx ].data.geometry.sphere_shell = g_sphere_shell;
        //hitgroup_records[ sbt_idx ].data.shading.glass.shadow_attenuation = { 0.6f, 0.6f, 0.6f };
        //sbt_idx ++;

        //// Floor
        //OPTIX_CHECK( optixSbtRecordPackHeader(
        //    state.radiance_floor_prog_group,
        //    &hitgroup_records[sbt_idx] ) );
        //hitgroup_records[ sbt_idx ].data.geometry.parallelogram = g_floor;
        //hitgroup_records[ sbt_idx ].data.shading.checker = {
        //    { 0.8f, 0.3f, 0.15f },      // Kd1
        //    { 0.9f, 0.85f, 0.05f },     // Kd2
        //    { 0.8f, 0.3f, 0.15f },      // Ka1
        //    { 0.9f, 0.85f, 0.05f },     // Ka2
        //    { 0.0f, 0.0f, 0.0f },       // Ks1
        //    { 0.0f, 0.0f, 0.0f },       // Ks2
        //    { 0.0f, 0.0f, 0.0f },       // Kr1
        //    { 0.0f, 0.0f, 0.0f },       // Kr2
        //    0.0f,                       // phong_exp1
        //    0.0f,                       // phong_exp2
        //    { 32.0f, 16.0f }            // inv_checker_size
        //};
        //sbt_idx++;

        //OPTIX_CHECK( optixSbtRecordPackHeader(
        //    state.occlusion_floor_prog_group,
        //    &hitgroup_records[sbt_idx] ) );
        //hitgroup_records[ sbt_idx ].data.geometry.parallelogram = g_floor;

        //CUdeviceptr d_hitgroup_records;
        //size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        //CUDA_CHECK( cudaMalloc(
        //    reinterpret_cast<void**>( &d_hitgroup_records ),
        //    sizeof_hitgroup_record*count_records
        //) );

        //CUDA_CHECK( cudaMemcpy(
        //    reinterpret_cast<void*>( d_hitgroup_records ),
        //    hitgroup_records,
        //    sizeof_hitgroup_record*count_records,
        //    cudaMemcpyHostToDevice
        //) );

        state.sbt.hitgroupRecordBase            = d_hitgroup_records;
        //state.sbt.hitgroupRecordCount           = count_records;
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

//
//
//

void launchSubframe( sutil::CUDAOutputBuffer<unsigned int>& output_buffer, WhittedState& state )
{
    // Launch
    // this map() thing basically returns the cudaMalloc-ed device pointer.
    unsigned int* result_buffer_data = output_buffer.map();

    // need to manually set the cuda-malloced device memory. note the semantics
    // of cudamemset: it sets #count number of BYTES to value; literally think
    // about what each byte have to be.
    CUDA_CHECK( cudaMemset ( result_buffer_data, 0xFF, state.params.numPrims*state.params.knn*sizeof(unsigned int) ) );
    state.params.frame_buffer = result_buffer_data;

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
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void cleanupState( WhittedState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_metal_sphere_prog_group ) );
    //OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.shading_module          ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.camera_module           ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_spheres                    ) ) );
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
        // Set up OptiX state
        //
        int32_t device_count = 0;
        CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
        //state.resize( device_count );
        std::cout << "Total GPUs visible: " << device_count << std::endl;
  
        int32_t device_id = 1;
        cudaDeviceProp prop;
        CUDA_CHECK( cudaGetDeviceProperties ( &prop, device_id ) );
        CUDA_CHECK( cudaSetDevice( device_id ) );
        std::cout << "\t[" << device_id << "]: " << prop.name << std::endl;

        Timing::startTiming("creat Context");
        createContext  ( state );
        Timing::stopTiming(true);

        Timing::startTiming("creat Geometry");
        createGeometry ( state );
        Timing::stopTiming(true);

        Timing::startTiming("creat Pipeline");
        createPipeline ( state );
        Timing::stopTiming(true);

        Timing::startTiming("creat SBT");
        createSBT      ( state );
        Timing::stopTiming(true);

        Timing::startTiming("init Launch Params");
        initLaunchParams( state );
        Timing::stopTiming(true);

        {
            sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
            Timing::startTiming("optixLaunch compute time");

            sutil::CUDAOutputBuffer<unsigned int> output_buffer(
                    output_buffer_type,
                    state.params.numPrims*state.params.knn,
                    1,
                    device_id
                    );

            launchSubframe( output_buffer, state );
            Timing::stopTiming(true);

            Timing::startTiming("Neighbor copy from device to host");
            void* data = output_buffer.getHostPointer();
            Timing::stopTiming(true);

            unsigned int totalNeighbors = 0;
            for (unsigned int i = 0; i < state.params.numPrims; i++) {
              for (unsigned int j = 0; j < state.params.knn; j++) {
                unsigned int p = reinterpret_cast<unsigned int*>( data )[ i * state.params.knn + j ];
                if (p == UINT_MAX) break;
                else {
                  totalNeighbors++;
                  float3 diff = state.points[p] - state.points[i];
                  float dists = dot(diff, diff);
                  if (dists > state.params.radius*state.params.radius) {
                    fprintf(stderr, "Point %u [%f, %f, %f] is not a neighbor of query %u [%f, %f, %f]. Dist is %lf.\n", p, state.points[i].x, state.points[i].y, state.points[i].z, i, state.points[p].x, state.points[p].y, state.points[p].z, sqrt(dists));
                    exit(1);
                  }
                }
                //std::cout << p << " ";
              }
              //std::cout << "\n";
            }
            std::cerr << "Sanity check done. Avg " << totalNeighbors/state.params.numPrims << " neighbors" << std::endl;
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
