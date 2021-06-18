//
// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>

#include <GLFW/glfw3.h>
#include <cstring>
#include <iomanip>

#include <cuda/whitted.h>

#include "optixCallablePrograms.h"

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

bool resize_dirty = false;
bool minimized    = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Shading state
bool         shading_changed = false;
unsigned int dc_index        = 0;

// Mouse state
int32_t mouse_button = -1;

//------------------------------------------------------------------------------
//
// Local types
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<EmptyData>    RayGenRecord;
typedef Record<EmptyData>    MissRecord;
typedef Record<HitGroupData> HitGroupRecord;
typedef Record<EmptyData>    CallablesRecord;

struct CallableProgramsState
{
    OptixDeviceContext          context                  = 0;
    OptixTraversableHandle      gas_handle               = 0;
    CUdeviceptr                 d_gas_output_buffer      = 0;

    OptixModule                 camera_module            = 0;
    OptixModule                 geometry_module          = 0;
    OptixModule                 shading_module           = 0;

    OptixProgramGroup           raygen_prog_group        = 0;
    OptixProgramGroup           miss_prog_group          = 0;
    OptixProgramGroup           hitgroup_prog_group      = 0;
    OptixProgramGroup           callable_prog_groups[3]  = {};

    OptixPipeline               pipeline                 = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};

    CUstream                    stream                   = 0;
    whitted::LaunchParams       params                   = {};
    whitted::LaunchParams*      d_params                 = 0;
    OptixShaderBindingTable     sbt                      = {};
};

//------------------------------------------------------------------------------
//
//  Geometry data
//
//------------------------------------------------------------------------------

const GeometryData::Sphere g_sphere = {
    {0.f, 0.f, 0.f},  // center
    1.0f              // radius
};

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking( static_cast<int>( xpos ), static_cast<int>( ypos ) );
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    whitted::LaunchParams* params = static_cast<whitted::LaunchParams*>( glfwGetWindowUserPointer( window ) );

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    whitted::LaunchParams* params = static_cast<whitted::LaunchParams*>( glfwGetWindowUserPointer( window ) );
    params->width                 = res_x;
    params->height                = res_y;
    camera_changed                = true;
    resize_dirty                  = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_SPACE )
    {
        shading_changed = true;
        dc_index        = ( dc_index + 1 ) % 3;
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( (int)yscroll ) )
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

void initLaunchParams( CallableProgramsState& state )
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.params.accum_buffer ),
                            state.params.width * state.params.height * sizeof( float4 ) ) );
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    state.params.subframe_index = 0u;

    // Set ambient light color and point light position
    std::vector<Light> lights( 2 );
    lights[0].type            = Light::Type::AMBIENT;
    lights[0].ambient.color   = make_float3( 0.4f, 0.4f, 0.4f );
    lights[1].type            = Light::Type::POINT;
    lights[1].point.color     = make_float3( 1.0f, 1.0f, 1.0f );
    lights[1].point.intensity = 1.0f;
    lights[1].point.position  = make_float3( 10.0f, 10.0f, -10.0f );
    lights[1].point.falloff   = Light::Falloff::QUADRATIC;

    state.params.lights.count = static_cast<unsigned int>( lights.size() );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.params.lights.data ), lights.size() * sizeof( Light ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.params.lights.data ), lights.data(),
                            lights.size() * sizeof( Light ), cudaMemcpyHostToDevice ) );

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( whitted::LaunchParams ) ) );

    state.params.handle = state.gas_handle;
}

static void sphere_bound( float3 center, float radius, float result[6] )
{
    OptixAabb* aabb = reinterpret_cast<OptixAabb*>( result );

    float3 m_min = center - radius;
    float3 m_max = center + radius;

    *aabb = {m_min.x, m_min.y, m_min.z, m_max.x, m_max.y, m_max.z};
}

static void buildGas( const CallableProgramsState&  state,
                      const OptixAccelBuildOptions& accel_options,
                      const OptixBuildInput&        build_input,
                      OptixTraversableHandle&       gas_handle,
                      CUdeviceptr&                  d_gas_output_buffer )
{
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr           d_temp_buffer_gas;

    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &build_input, 1, &gas_buffer_sizes ) );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ), compactedSizeOffset + 8 ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild( state.context, 0, &accel_options, &build_input, 1, d_temp_buffer_gas,
                                  gas_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_gas_and_compacted_size,
                                  gas_buffer_sizes.outputSizeInBytes, &gas_handle, &emitProperty, 1 ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

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

void createGeometry( CallableProgramsState& state )
{
    //
    // Build Custom Primitive for Sphere
    //

    // Load AABB into device memory
    OptixAabb   aabb;
    CUdeviceptr d_aabb;

    sphere_bound( g_sphere.center, g_sphere.radius, reinterpret_cast<float*>( &aabb ) );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb ), sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_aabb ), &aabb, sizeof( OptixAabb ), cudaMemcpyHostToDevice ) );

    // Setup AABB build input
    uint32_t aabb_input_flags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};

    const uint32_t sbt_index[1] = {0};
    CUdeviceptr    d_sbt_index;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), sizeof( uint32_t ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_sbt_index ), sbt_index, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input                                = {};
    aabb_input.type                                           = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers               = &d_aabb;
    aabb_input.customPrimitiveArray.flags                     = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords             = 1;
    aabb_input.customPrimitiveArray.numPrimitives             = 1;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer      = d_sbt_index;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof( uint32_t );
    aabb_input.customPrimitiveArray.primitiveIndexOffset      = 0;

    OptixAccelBuildOptions accel_options = {
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags
        OPTIX_BUILD_OPERATION_BUILD         // operation
    };

    buildGas( state, accel_options, aabb_input, state.gas_handle, state.d_gas_output_buffer );

    CUDA_CHECK( cudaFree( (void*)d_aabb ) );
    CUDA_CHECK( cudaFree( (void*)d_sbt_index ) );
}

void createModules( CallableProgramsState& state )
{
    OptixModuleCompileOptions module_compile_options = {
        100,                                 // maxRegisterCount
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,  // optLevel
        OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO   // debugLevel
    };
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    {
        const std::string ptx = sutil::getPtxString( nullptr, nullptr, "whitted.cu" );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX( state.context, &module_compile_options, &state.pipeline_compile_options,
                                                   ptx.c_str(), ptx.size(), log, &sizeof_log, &state.camera_module ) );
    }

    {
        const std::string ptx = sutil::getPtxString( nullptr, nullptr, "sphere.cu" );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX( state.context, &module_compile_options, &state.pipeline_compile_options,
                                                   ptx.c_str(), ptx.size(), log, &sizeof_log, &state.geometry_module ) );
    }

    {
        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixCallablePrograms.cu" );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX( state.context, &module_compile_options, &state.pipeline_compile_options,
                                                   ptx.c_str(), ptx.size(), log, &sizeof_log, &state.shading_module ) );
    }
}

static void createCameraProgram( CallableProgramsState& state, std::vector<OptixProgramGroup>& program_groups )
{
    OptixProgramGroup        cam_prog_group;
    OptixProgramGroupOptions cam_prog_group_options = {};
    OptixProgramGroupDesc    cam_prog_group_desc    = {};
    cam_prog_group_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module               = state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName    = "__raygen__pinhole";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &cam_prog_group_desc, 1, &cam_prog_group_options, log,
                                              &sizeof_log, &cam_prog_group ) );

    program_groups.push_back( cam_prog_group );
    state.raygen_prog_group = cam_prog_group;
}

static void createSphereProgram( CallableProgramsState& state, std::vector<OptixProgramGroup>& program_groups )
{
    OptixProgramGroup        hitgroup_prog_group;
    OptixProgramGroupOptions hitgroup_prog_group_options  = {};
    OptixProgramGroupDesc    hitgroup_prog_group_desc     = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    hitgroup_prog_group_desc.hitgroup.moduleIS            = state.geometry_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    hitgroup_prog_group_desc.hitgroup.moduleCH            = state.shading_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &hitgroup_prog_group_desc, 1, &hitgroup_prog_group_options,
                                              log, &sizeof_log, &hitgroup_prog_group ) );

    program_groups.push_back( hitgroup_prog_group );
    state.hitgroup_prog_group = hitgroup_prog_group;

    // Callable programs
    OptixProgramGroupOptions callable_prog_group_options  = {};
    OptixProgramGroupDesc    callable_prog_group_descs[3] = {};

    callable_prog_group_descs[0].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_descs[0].callables.moduleDC            = state.shading_module;
    callable_prog_group_descs[0].callables.entryFunctionNameDC = "__direct_callable__phong_shade";
    callable_prog_group_descs[0].callables.moduleCC            = state.shading_module;
    callable_prog_group_descs[0].callables.entryFunctionNameCC = "__continuation_callable__raydir_shade";

    callable_prog_group_descs[1].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_descs[1].callables.moduleDC            = state.shading_module;
    callable_prog_group_descs[1].callables.entryFunctionNameDC = "__direct_callable__checkered_shade";

    callable_prog_group_descs[2].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_descs[2].callables.moduleDC            = state.shading_module;
    callable_prog_group_descs[2].callables.entryFunctionNameDC = "__direct_callable__normal_shade";

    OPTIX_CHECK( optixProgramGroupCreate( state.context, callable_prog_group_descs, 3, &callable_prog_group_options,
                                          log, &sizeof_log, state.callable_prog_groups ) );

    program_groups.push_back( state.callable_prog_groups[0] );
    program_groups.push_back( state.callable_prog_groups[1] );
    program_groups.push_back( state.callable_prog_groups[2] );
}

static void createMissProgram( CallableProgramsState& state, std::vector<OptixProgramGroup>& program_groups )
{
    OptixProgramGroupOptions miss_prog_group_options = {};
    OptixProgramGroupDesc    miss_prog_group_desc    = {};
    miss_prog_group_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module                 = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName      = "__miss__raydir_shade";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc, 1, &miss_prog_group_options, log,
                                              &sizeof_log, &state.miss_prog_group ) );

    program_groups.push_back( state.miss_prog_group );
}

void createPipeline( CallableProgramsState& state )
{
    const uint32_t max_trace_depth     = 1;
    const uint32_t max_cc_depth        = 1;
    const uint32_t max_dc_depth        = 1;
    const uint32_t max_traversal_depth = 1;

    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                          // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,  // traversableGraphFlags
        whitted::NUM_PAYLOAD_VALUES,                    // numPayloadValues
        3,                                              // numAttributeValues
        OPTIX_EXCEPTION_FLAG_NONE,                      // exceptionFlags
        "params"                                        // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createModules( state );
    createCameraProgram( state, program_groups );
    createSphereProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
        max_trace_depth,                // maxTraceDepth
        OPTIX_COMPILE_DEBUG_LEVEL_FULL  // debugLevel
    };
    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate( state.context, &state.pipeline_compile_options, &pipeline_link_options,
                                          program_groups.data(), static_cast<unsigned int>( program_groups.size() ),
                                          log, &sizeof_log, &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth, max_cc_depth, max_dc_depth, &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size, max_traversal_depth ) );
}

void syncDCShaderIndexToSbt( CallableProgramsState& state )
{
    // Update the dc_index in HitGroupData so that the closest hit program invokes the correct DC for shading
    HitGroupRecord hitgroup_record;
    OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_group, &hitgroup_record ) );
    hitgroup_record.data.dc_index = dc_index;

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase
                                                     + ( sizeof( hitgroup_record.header ) + sizeof( GeometryData::Sphere ) ) ),
                            &hitgroup_record.data.dc_index, sizeof( unsigned int ), cudaMemcpyHostToDevice ) );
}

void createSBT( CallableProgramsState& state )
{
    // Raygen program record
    {
        RayGenRecord raygen_record;
        OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &raygen_record ) );

        CUdeviceptr d_raygen_record;
        size_t      sizeof_raygen_record = sizeof( RayGenRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen_record ), sizeof_raygen_record ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_raygen_record ), &raygen_record, sizeof_raygen_record, cudaMemcpyHostToDevice ) );

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        MissRecord miss_record;
        OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &miss_record ) );

        CUdeviceptr d_miss_record;
        size_t      sizeof_miss_record = sizeof( MissRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss_record ), sizeof_miss_record ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_miss_record ), &miss_record, sizeof_miss_record, cudaMemcpyHostToDevice ) );

        state.sbt.missRecordBase          = d_miss_record;
        state.sbt.missRecordCount         = 1;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record
    {
        HitGroupRecord hitgroup_record;
        OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_group, &hitgroup_record ) );
        hitgroup_record.data.sphere = g_sphere;

        CUdeviceptr d_hitgroup_record;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup_record ), sizeof_hitgroup_record ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_hitgroup_record ), &hitgroup_record, sizeof_hitgroup_record,
                                cudaMemcpyHostToDevice ) );

        state.sbt.hitgroupRecordBase          = d_hitgroup_record;
        state.sbt.hitgroupRecordCount         = 1;
        state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( sizeof_hitgroup_record );
    }

    // Callables program record
    {
        CallablesRecord callable_records[3];
        OPTIX_CHECK( optixSbtRecordPackHeader( state.callable_prog_groups[0], &callable_records[0] ) );
        OPTIX_CHECK( optixSbtRecordPackHeader( state.callable_prog_groups[1], &callable_records[1] ) );
        OPTIX_CHECK( optixSbtRecordPackHeader( state.callable_prog_groups[2], &callable_records[2] ) );

        CUdeviceptr d_callable_records;
        size_t      sizeof_callable_record = sizeof( CallablesRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_callable_records ), sizeof_callable_record * 3 ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_callable_records ), callable_records,
                                sizeof_callable_record * 3, cudaMemcpyHostToDevice ) );

        state.sbt.callablesRecordBase          = d_callable_records;
        state.sbt.callablesRecordCount         = 3;
        state.sbt.callablesRecordStrideInBytes = static_cast<unsigned int>( sizeof_callable_record );
    }
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

void createContext( CallableProgramsState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}

//
// Handle updates
//

void initCameraState()
{
    camera.setEye( make_float3( 0.0f, 0.0f, -3.0f ) );
    camera.setLookat( make_float3( 0.0f, 0.0f, 0.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 60.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock( true );
}

void handleCameraUpdate( CallableProgramsState& state )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( state.params.width ) / static_cast<float>( state.params.height ) );
    state.params.eye = camera.eye();
    camera.UVWFrame( state.params.U, state.params.V, state.params.W );
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, whitted::LaunchParams& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &params.accum_buffer ), params.width * params.height * sizeof( float4 ) ) );
}

void handleShading( CallableProgramsState& state )
{
    if( !shading_changed )
        return;
    shading_changed = false;

    syncDCShaderIndexToSbt( state );
}

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, CallableProgramsState& state )
{
    // Update params on device
    if( camera_changed || resize_dirty || shading_changed )
        state.params.subframe_index = 0;

    handleCameraUpdate( state );
    handleResize( output_buffer, state.params );
    handleShading( state );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, CallableProgramsState& state )
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer  = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ), &state.params,
                                 sizeof( whitted::LaunchParams ), cudaMemcpyHostToDevice, state.stream ) );

    OPTIX_CHECK( optixLaunch( state.pipeline, state.stream, reinterpret_cast<CUdeviceptr>( state.d_params ),
                              sizeof( whitted::LaunchParams ), &state.sbt,
                              state.params.width,   // launch width
                              state.params.height,  // launch height
                              1                     // launch depth
                              ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display( output_buffer.width(), output_buffer.height(), framebuf_res_x, framebuf_res_y, output_buffer.getPBO() );
}


void cleanupState( CallableProgramsState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.hitgroup_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.callable_prog_groups[0] ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.callable_prog_groups[1] ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.callable_prog_groups[2] ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );
    OPTIX_CHECK( optixModuleDestroy( state.shading_module ) );
    OPTIX_CHECK( optixModuleDestroy( state.geometry_module ) );
    OPTIX_CHECK( optixModuleDestroy( state.camera_module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );


    CUDA_CHECK( cudaStreamDestroy( state.stream ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.callablesRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.lights.data ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params ) ) );
}

int main( int argc, char* argv[] )
{
    CallableProgramsState state;
    state.params.width                             = 768;
    state.params.height                            = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int               w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        initCameraState();

        //
        // Set up OptiX state
        //
        createContext( state );
        createGeometry( state );
        createPipeline( state );
        createSBT( state );

        initLaunchParams( state );

        //
        // Render loop
        //
        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixCallablePrograms", state.params.width, state.params.height );
            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback( window, cursorPosCallback );
            glfwSetWindowSizeCallback( window, windowSizeCallback );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback( window, keyCallback );
            glfwSetScrollCallback( window, scrollCallback );
            glfwSetWindowUserPointer( window, &state.params );

            {
                // output_buffer needs to be destroyed before cleanupUI is called
                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, state.params.width, state.params.height );

                output_buffer.setStream( state.stream );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, state );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, state );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers( window );

                    ++state.params.subframe_index;
                } while( !glfwWindowShouldClose( window ) );
            }
            sutil::cleanupUI( window );
        }
        else
        {
            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, state.params.width, state.params.height );

            handleCameraUpdate( state );
            handleResize( output_buffer, state.params );
            handleShading( state );
            launchSubframe( output_buffer, state );

            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = output_buffer.width();
            buffer.height       = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            sutil::saveImage( outfile.c_str(), buffer, false );

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
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
