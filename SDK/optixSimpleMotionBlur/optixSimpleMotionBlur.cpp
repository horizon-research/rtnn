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
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <sampleConfig.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>

#include "optixSimpleMotionBlur.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

bool              resize_dirty  = false;
bool              minimized     = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int2              mouse_prev_pos;
int32_t           mouse_button = -1;

int32_t           samples_per_launch = 16;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>         RayGenRecord;
typedef Record<MissData>           MissRecord;
typedef Record<HitGroupData>       HitGroupRecord;


enum InstanceType
{
    SPHERE=0,
    TRI,
    COUNT
};


struct Vertex
{
    float x, y, z, pad;
};


struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};


struct Instance
{
    float transform[12];
};


struct SimpleMotionBlurState
{
    OptixDeviceContext           context                        = 0;

    OptixTraversableHandle       tri_gas_handle                 = 0;   // Traversable handle for triangle AS
    CUdeviceptr                  d_tri_gas_output_buffer        = 0;   // Triangle AS memory

    OptixTraversableHandle       sphere_gas_handle              = 0;   // Traversable handle for sphere
    CUdeviceptr                  d_sphere_gas_output_buffer     = 0;   // Sphere AS memory
    OptixTraversableHandle       sphere_motion_transform_handle = 0;
    CUdeviceptr                  d_sphere_motion_transform      = 0;

    OptixTraversableHandle       ias_handle                     = 0;   // Traversable handle for instance AS
    CUdeviceptr                  d_ias_output_buffer            = 0;   // Instance AS memory

    OptixModule                  ptx_module                     = 0;
    OptixPipelineCompileOptions  pipeline_compile_options       = {};
    OptixPipeline                pipeline                       = 0;

    OptixProgramGroup            raygen_prog_group              = 0;
    OptixProgramGroup            miss_group                     = 0;
    OptixProgramGroup            tri_hit_group                  = 0;
    OptixProgramGroup            sphere_hit_group               = 0;

    CUstream                     stream                         = 0;
    Params                       params;
    Params*                      d_params;

    OptixShaderBindingTable      sbt                            = {};
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
        trackball.startTracking(static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );

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

    Params* params  = static_cast<Params*>( glfwGetWindowUserPointer( window ) );
    params->width   = res_x;
    params->height  = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q ||
            key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
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


void initLaunchParams( SimpleMotionBlurState& state )
{
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.params.accum_buffer ),
                state.params.width*state.params.height*sizeof(float4)
                ) );

    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.subframe_index = 0u;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );

    state.params.handle = state.ias_handle;
}


void handleCameraUpdate( Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( params.width ) / static_cast<float>( params.height ) );
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.accum_buffer ),
                params.width*params.height*sizeof(float4)
                ) );
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, SimpleMotionBlurState& state )
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
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
                state.params.width,  // launch width
                state.params.height, // launch height
                1                    // launch depth
                ) );
    output_buffer.unmap();
}


void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}


void initCameraState()
{
    camera.setEye( make_float3( 0.0f, 0.0f, 5.0f ) );
    camera.setLookat( make_float3( 0.0f, 0.0f, 0.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 35.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock(true);
}


void createContext( SimpleMotionBlurState& state )
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


void buildGAS( SimpleMotionBlurState& state )
{
    //
    // Build triangle GAS
    //
    {
        const int NUM_KEYS  = 3;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags            = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
        accel_options.motionOptions.numKeys   = NUM_KEYS;
        accel_options.motionOptions.timeBegin = 0.0f;
        accel_options.motionOptions.timeEnd   = 1.0f;
        accel_options.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;

        //
        // copy triangle mesh data to device
        //
        const int NUM_VERTS = 3;
        const std::array<Vertex, NUM_VERTS*NUM_KEYS> tri_vertices =
        { {
              {  0.0f,  0.0f, 0.0f, 0.0f },  //
              {  1.0f,  0.0f, 0.0f, 0.0f },  // Motion key 0
              {  0.5f,  1.0f, 0.0f, 0.0f },  //

              {  0.5f,  0.0f, 0.0f, 0.0f },  //
              {  1.5f,  0.0f, 0.0f, 0.0f },  // Motion key 1
              {  1.0f,  1.0f, 0.0f, 0.0f },  //

              {  0.5f, -0.5f, 0.0f, 0.0f },  //
              {  1.5f, -0.5f, 0.0f, 0.0f },  // Motion key 2
              {  1.0f,  0.5f, 0.0f, 0.0f },  //

        } };

        const size_t vertices_size_in_bytes = NUM_VERTS*NUM_KEYS*sizeof( Vertex );
        CUdeviceptr d_tri_vertices;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_tri_vertices ), vertices_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_tri_vertices ),
                    tri_vertices.data(),
                    vertices_size_in_bytes,
                    cudaMemcpyHostToDevice
                    ) );

        uint32_t triangle_input_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        CUdeviceptr vertex_buffer_ptrs[ NUM_KEYS ];
        for( int i = 0; i < NUM_KEYS; ++i )
            vertex_buffer_ptrs[i] = d_tri_vertices + i*NUM_VERTS*sizeof(Vertex);


        OptixBuildInput triangle_input = {};
        triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
        triangle_input.triangleArray.numVertices                 = NUM_VERTS;
        triangle_input.triangleArray.vertexBuffers               = vertex_buffer_ptrs;
        triangle_input.triangleArray.flags                       = &triangle_input_flag;
        triangle_input.triangleArray.numSbtRecords               = 1;
        triangle_input.triangleArray.sbtIndexOffsetBuffer        = 0;


        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage(
                    state.context,
                    &accel_options,
                    &triangle_input,
                    1,  // num_build_inputs
                    &gas_buffer_sizes
                    ) );

        CUdeviceptr d_temp_buffer;
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_temp_buffer ),
                    gas_buffer_sizes.tempSizeInBytes
                    ) );

        // non-compacted output
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
                    0,                  // CUDA stream
                    &accel_options,
                    &triangle_input,
                    1,                  // num build inputs
                    d_temp_buffer,
                    gas_buffer_sizes.tempSizeInBytes,
                    d_buffer_temp_output_gas_and_compacted_size,
                    gas_buffer_sizes.outputSizeInBytes,
                    &state.tri_gas_handle,
                    &emitProperty,      // emitted property list
                    1                   // num emitted properties
                    ) );

        CUDA_CHECK( cudaFree( (void*)d_temp_buffer ) );
        CUDA_CHECK( cudaFree( (void*)d_tri_vertices ) );

        size_t compacted_gas_size;
        CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_tri_gas_output_buffer ), compacted_gas_size ) );

            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( state.context, 0, state.tri_gas_handle, state.d_tri_gas_output_buffer,
                                            compacted_gas_size, &state.tri_gas_handle ) );

            CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
        }
        else
        {
            state.d_tri_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }

    //
    // Build sphere GAS
    //
    {
        OptixAccelBuildOptions accel_options  = {};
        accel_options.buildFlags              = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation               = OPTIX_BUILD_OPERATION_BUILD;

        // AABB build input
        OptixAabb   aabb = { -1.5f, -1.0f, -0.5f,
                             -0.5f,  0.0f,  0.5f};
        CUdeviceptr d_aabb_buffer;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_aabb_buffer ),
                    &aabb,
                    sizeof( OptixAabb ),
                    cudaMemcpyHostToDevice
                    ) );

        uint32_t sphere_input_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixBuildInput sphere_input = {};
        sphere_input.type                                = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        sphere_input.customPrimitiveArray.aabbBuffers    = &d_aabb_buffer;
        sphere_input.customPrimitiveArray.numPrimitives  = 1;
        sphere_input.customPrimitiveArray.flags          = &sphere_input_flag;
        sphere_input.customPrimitiveArray.numSbtRecords  = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &sphere_input,
                                                   1,  // num_build_inputs
                                                   &gas_buffer_sizes ) );

        CUdeviceptr d_temp_buffer;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                    compactedSizeOffset + 8
                    ) );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

        OPTIX_CHECK( optixAccelBuild( state.context,
                                      0,                  // CUDA stream
                                      &accel_options,
                                      &sphere_input,
                                      1,                  // num build inputs
                                      d_temp_buffer,
                                      gas_buffer_sizes.tempSizeInBytes,
                                      d_buffer_temp_output_gas_and_compacted_size,
                                      gas_buffer_sizes.outputSizeInBytes,
                                      &state.sphere_gas_handle,
                                      &emitProperty,      // emitted property list
                                      1                   // num emitted properties
                                      ) );

        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_aabb_buffer ) ) );

        size_t compacted_gas_size;
        CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_sphere_gas_output_buffer ), compacted_gas_size ) );

            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( state.context, 0, state.sphere_gas_handle, state.d_sphere_gas_output_buffer,
                                            compacted_gas_size, &state.sphere_gas_handle ) );

            CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
        }
        else
        {
            state.d_sphere_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }

        {
            const float motion_matrix_keys[2][12] =
            {
                {
                    1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f
                },
                {
                    1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.5f,
                    0.0f, 0.0f, 1.0f, 0.0f
                }
            };

            OptixMatrixMotionTransform motion_transform = {};
            motion_transform.child                      = state.sphere_gas_handle;
            motion_transform.motionOptions.numKeys      = 2;
            motion_transform.motionOptions.timeBegin    = 0.0f;
            motion_transform.motionOptions.timeEnd      = 1.0f;
            motion_transform.motionOptions.flags        = OPTIX_MOTION_FLAG_NONE;
            memcpy( motion_transform.transform, motion_matrix_keys, 2 * 12 * sizeof( float ) );

            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &state.d_sphere_motion_transform),
                        sizeof( OptixMatrixMotionTransform )
                        ) );

            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( state.d_sphere_motion_transform ),
                        &motion_transform,
                        sizeof( OptixMatrixMotionTransform ),
                        cudaMemcpyHostToDevice
                        ) );

            OPTIX_CHECK( optixConvertPointerToTraversableHandle(
                        state.context,
                        state.d_sphere_motion_transform,
                        OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
                        &state.sphere_motion_transform_handle
                        ) );
        }
    }
}


void buildInstanceAccel( SimpleMotionBlurState& state )
{
    Instance instance = { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 } };

    const size_t instance_size_in_bytes = sizeof( OptixInstance ) * InstanceType::COUNT;
    OptixInstance optix_instances[ InstanceType::COUNT ];
    memset( optix_instances, 0, instance_size_in_bytes );

    optix_instances[InstanceType::SPHERE].flags             = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[InstanceType::SPHERE].instanceId        = 1;
    optix_instances[InstanceType::SPHERE].sbtOffset         = 0;
    optix_instances[InstanceType::SPHERE].visibilityMask    = 1;
    optix_instances[InstanceType::SPHERE].traversableHandle = state.sphere_motion_transform_handle;
    memcpy( optix_instances[InstanceType::SPHERE].transform, instance.transform, sizeof( float ) * 12 );

    optix_instances[InstanceType::TRI].flags                = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[InstanceType::TRI].instanceId           = 0;
    optix_instances[InstanceType::TRI].sbtOffset            = 1; // Prefix sum of previous instance numSBT records
    optix_instances[InstanceType::TRI].visibilityMask       = 1;
    optix_instances[InstanceType::TRI].traversableHandle    = state.tri_gas_handle;
    memcpy( optix_instances[InstanceType::TRI].transform, instance.transform, sizeof( float ) * 12 );

    OptixAabb aabbs[2] =
    {
        // NOTE: instead of using a motion IAS, we are using a static IAS with expanded AABBs
        //       to enclose the motion path.  See comments below on the IAS's accel options.
        { -1.5f, -1.0f, -0.5f,
          -0.5f,  0.5f,  0.5f  },
        {  0.5f,  0.0f, -0.01f,   // NOTE: the bbox for the triangle geom is being ignored.  Should it be???
           1.5f,  1.5f,  0.01f }  //
    };
    CUdeviceptr  d_aabbs;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabbs), 2*sizeof(OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_aabbs),
                aabbs,
                2*sizeof(OptixAabb),
                cudaMemcpyHostToDevice
                ) );


    CUdeviceptr  d_instances;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_instances ), instance_size_in_bytes) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_instances ),
                optix_instances,
                instance_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances;
    instance_input.instanceArray.numInstances = InstanceType::COUNT;
    instance_input.instanceArray.aabbs        = d_aabbs;
    instance_input.instanceArray.numAabbs     = InstanceType::COUNT; // * NUM_KEYS; NOTE: need AABB per key per instance if using motion Accel

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags              = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation               = OPTIX_BUILD_OPERATION_BUILD;
    // Note: Instead of using padded AABBs above, we could make the IAS into a motion BVH and, provide AABBs
    //       for each of the motion keys
    /*
    accel_options.motionOptions.numKeys   = 2;
    accel_options.motionOptions.timeBegin = 0.0f;
    accel_options.motionOptions.timeEnd   = 1.0f;
    accel_options.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
    */

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &instance_input,
                1, // num build inputs
                &ias_buffer_sizes
                ) );

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_temp_buffer ),
                ias_buffer_sizes.tempSizeInBytes
                ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.d_ias_output_buffer ),
                ias_buffer_sizes.outputSizeInBytes
                ) );

    OPTIX_CHECK( optixAccelBuild(
                state.context,
                0,                  // CUDA stream
                &accel_options,
                &instance_input,
                1,                  // num build inputs
                d_temp_buffer,
                ias_buffer_sizes.tempSizeInBytes,
                state.d_ias_output_buffer,
                ias_buffer_sizes.outputSizeInBytes,
                &state.ias_handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_instances   ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_aabbs       ) ) );
}


void createModule( SimpleMotionBlurState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY; // IMPORTANT: if not set to 'ANY', instance traversables will not work
    state.pipeline_compile_options.numPayloadValues          = 3;
    state.pipeline_compile_options.numAttributeValues        = 3;
    state.pipeline_compile_options.usesMotionBlur            = true;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixSimpleMotionBlur.cu" );
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                state.context,
                &module_compile_options,
                &state.pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &state.ptx_module
                ) );
}


void createProgramGroups( SimpleMotionBlurState& state )
{
    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                state.context,
                &raygen_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &state.raygen_prog_group
                )
            );

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__camera";
    sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                state.context,
                &miss_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &state.miss_group
                )
            );

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__camera";
    sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG(
            optixProgramGroupCreate(
                state.context,
                &hit_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &state.tri_hit_group
                )
            );

    hit_prog_group_desc.hitgroup.moduleIS            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG(
            optixProgramGroupCreate(
                state.context,
                &hit_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &state.sphere_hit_group
                )
            );
}


void createPipeline( SimpleMotionBlurState& state )
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.miss_group,
        state.sphere_hit_group,
        state.tri_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = 2;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                state.context,
                &state.pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                log,
                &sizeof_log,
                &state.pipeline
                ) );

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stackSizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.raygen_prog_group, &stackSizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.miss_group, &stackSizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.sphere_hit_group, &stackSizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.tri_hit_group, &stackSizes ) );

    unsigned int maxTraceDepth = 1;
    unsigned int maxCCDepth = 0;
    unsigned int maxDCDepth = 0;
    unsigned int directCallableStackSizeFromTraversal;
    unsigned int directCallableStackSizeFromState;
    unsigned int continuationStackSize;
    OPTIX_CHECK( optixUtilComputeStackSizes(
                &stackSizes,
                maxTraceDepth,
                maxCCDepth,
                maxDCDepth,
                &directCallableStackSizeFromTraversal,
                &directCallableStackSizeFromState,
                &continuationStackSize
                ) );

    // This is 3 since the largest depth is IAS->MT->GAS
    unsigned int maxTraversalDepth = 3;

    OPTIX_CHECK( optixPipelineSetStackSize(
                state.pipeline,
                directCallableStackSizeFromTraversal,
                directCallableStackSizeFromState,
                continuationStackSize,
                maxTraversalDepth
                ) );
}


void createSBT( SimpleMotionBlurState& state )
{
    CUdeviceptr   d_raygen_record;
    const size_t  raygen_record_size = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen_record ), raygen_record_size ) );

    RayGenRecord rg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr   d_miss_records;
    const size_t  miss_record_size = sizeof( MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss_records ), miss_record_size) );

    MissRecord ms_sbt[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_group, &ms_sbt[0] ) );
    ms_sbt[0].data.color = {0.1f, 0.1f, 0.1f};

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_miss_records ),
                ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr   d_hitgroup_records;
    const size_t  hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_hitgroup_records ),
                hitgroup_record_size*InstanceType::COUNT
                ) );
    HitGroupRecord hitgroup_records[ InstanceType::COUNT ];

    //
    // Hit groups
    //
    OPTIX_CHECK( optixSbtRecordPackHeader( state.sphere_hit_group, &hitgroup_records[ InstanceType::SPHERE ] ) );
    hitgroup_records[ InstanceType::SPHERE ].data.color  = make_float3(  0.9f,  0.1f, 0.1f );
    hitgroup_records[ InstanceType::SPHERE ].data.center = make_float3( -1.0f, -0.5f, 0.0f );
    hitgroup_records[ InstanceType::SPHERE ].data.radius =  0.5f;


    OPTIX_CHECK( optixSbtRecordPackHeader( state.sphere_hit_group, &hitgroup_records[ InstanceType::SPHERE ] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( state.tri_hit_group, &hitgroup_records[ InstanceType::TRI ] ) );
    hitgroup_records[ InstanceType::TRI    ].data.color  = make_float3( 0.1f, 0.1f, 0.9f );
    hitgroup_records[ InstanceType::TRI    ].data.center = make_float3( 0.0f ); // Not used
    hitgroup_records[ InstanceType::TRI    ].data.radius =  0.0f;               // Not used

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_hitgroup_records ),
                hitgroup_records,
                hitgroup_record_size*InstanceType::COUNT,
                cudaMemcpyHostToDevice
                ) );

    state.sbt.raygenRecord                = d_raygen_record;
    state.sbt.missRecordBase              = d_miss_records;
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    state.sbt.missRecordCount             = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase          = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    state.sbt.hitgroupRecordCount         = InstanceType::COUNT;
}


void cleanupState( SimpleMotionBlurState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline          ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.miss_group        ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.tri_hit_group     ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.ptx_module        ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context           ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord         ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase   ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_tri_gas_output_buffer  ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_sphere_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer      ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params                 ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_ias_output_buffer      ) ) );
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    SimpleMotionBlurState state;
    state.params.width  = 768;
    state.params.height = 768;
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
            int w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else if( arg == "--launch-samples" || arg == "-s" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            samples_per_launch = atoi( argv[++i] );
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
        createContext      ( state );
        buildGAS           ( state );
        buildInstanceAccel ( state );
        createModule       ( state );
        createProgramGroups( state );
        createPipeline     ( state );
        createSBT          ( state );

        initLaunchParams( state );

        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixSimpleMotionBlur", state.params.width, state.params.height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetWindowUserPointer    ( window, &state.params         );

            {
                // output_buffer needs to be destroyed before cleanupUI is called
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                        output_buffer_type,
                        state.params.width,
                        state.params.height
                        );

                output_buffer.setStream( state.stream );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                //
                // Render loop
                //
                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, state.params );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, state );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1-t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    ++state.params.subframe_index;
                }
                while( !glfwWindowShouldClose( window ) );
            }

            sutil::cleanupUI( window );
        }
        else
        {
            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW(); // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                    );

            handleCameraUpdate( state.params );
            handleResize( output_buffer, state.params );
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
