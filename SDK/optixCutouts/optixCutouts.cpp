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

#include <GLFW/glfw3.h>

#include "optixCutouts.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>


bool             use_pbo      = true;
bool             resize_dirty = false;
bool             minimized    = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int2             mouse_prev_pos;
int32_t          mouse_button = -1;

int32_t          samples_per_launch = 16;

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

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;


struct Vertex
{
    float x, y, z, pad;
};


struct Instance
{
    float transform[12];
};


struct CutoutsState
{
    OptixDeviceContext          context                      = 0;

    OptixTraversableHandle      triangle_gas_handle          = 0;  // Traversable handle for triangle AS
    CUdeviceptr                 d_triangle_gas_output_buffer = 0;  // Triangle AS memory
    CUdeviceptr                 d_vertices                   = 0;
    CUdeviceptr                 d_tex_coords                 = 0;

    OptixTraversableHandle      sphere_gas_handle            = 0;  // Traversable handle for sphere AS
    CUdeviceptr                 d_sphere_gas_output_buffer   = 0;  // Sphere AS memory

    OptixTraversableHandle      ias_handle                   = 0;  // Traversable handle for instance AS
    CUdeviceptr                 d_ias_output_buffer          = 0;  // Instance AS memory

    OptixModule                 ptx_module                   = 0;

    OptixPipelineCompileOptions pipeline_compile_options     = {};
    OptixPipeline               pipeline                     = 0;

    OptixProgramGroup           raygen_prog_group            = 0;
    OptixProgramGroup           radiance_miss_group          = 0;
    OptixProgramGroup           occlusion_miss_group         = 0;
    OptixProgramGroup           radiance_hit_group           = 0;
    OptixProgramGroup           occlusion_hit_group          = 0;

    CUstream                    stream                       = 0;
    Params                      params;
    Params*                     d_params;

    OptixShaderBindingTable     sbt = {};
};


//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

const int32_t TRIANGLE_COUNT     = 32;
const int32_t TRIANGLE_MAT_COUNT = 5;
const int32_t SPHERE_COUNT       = 1;
const int32_t SPHERE_MAT_COUNT   = 1;

const static std::array<Vertex, TRIANGLE_COUNT*3> g_vertices =
{ {
    // Floor  -- white lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },

    {    0.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,    0.0f, 0.0f },

    // Ceiling -- white lambert
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    // Back wall -- white lambert
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },

    // Right wall -- green lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },

    // Left wall -- red lambert
    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },

    // Short block -- white lambert
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },

    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {  240.0f,    0.0f,  272.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  290.0f,    0.0f,  114.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  130.0f,    0.0f,   65.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {   82.0f,    0.0f,  225.0f, 0.0f },

    // Tall block -- white lambert
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },

    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  472.0f,    0.0f,  406.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  314.0f,    0.0f,  456.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  265.0f,    0.0f,  296.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  423.0f,    0.0f,  247.0f, 0.0f },

    // Ceiling light -- emmissive
    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },

    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },
    {  343.0f,  548.6f,  332.0f, 0.0f }
} };


static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices =
{ {
    0, 0,                          // Floor         -- white lambert
    0, 0,                          // Ceiling       -- white lambert
    0, 0,                          // Back wall     -- white lambert
    1, 1,                          // Right wall    -- green lambert
    2, 2,                          // Left wall     -- red lambert
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  // Short block   -- cutout
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
    3, 3                           // Ceiling light -- emmissive
} };


const std::array<float3, TRIANGLE_MAT_COUNT> g_emission_colors =
{ {
    {  0.0f,  0.0f, 0.0f },
    {  0.0f,  0.0f, 0.0f },
    {  0.0f,  0.0f, 0.0f },
    { 15.0f, 15.0f, 5.0f },
    {  0.0f,  0.0f, 0.0f }
} };


const std::array<float3, TRIANGLE_MAT_COUNT> g_diffuse_colors =
{ {
    { 0.80f, 0.80f, 0.80f },
    { 0.05f, 0.80f, 0.05f },
    { 0.80f, 0.05f, 0.05f },
    { 0.50f, 0.00f, 0.00f },
    { 0.70f, 0.25f, 0.00f }
} };


// NB: Some UV scaling is baked into the coordinates for the short block, since
//     the coordinates are used for the cutout texture.
const std::array<float2, TRIANGLE_COUNT* 3> g_tex_coords =
{ {
    // Floor
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Ceiling
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Back wall
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Right wall
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Left wall
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Short Block
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },

    // Tall Block
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },

    // Ceiling light
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f }
} };


const Sphere g_sphere                = { 410.0f, 90.0f, 110.0f, 90.0f };
const float3 g_sphere_emission_color = { 0.0f };
const float3 g_sphere_diffuse_color  = { 0.1f, 0.2f, 0.8f };

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

    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );
    params->width  = res_x;
    params->height = res_y;
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
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if(trackball.wheelEvent((int)yscroll))
        camera_changed = true;
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
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}


void initLaunchParams( CutoutsState& state )
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.params.accum_buffer ),
                            state.params.width*state.params.height*sizeof(float4) ) );
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;

    state.params.light.emission = make_float3(   15.0f,  15.0f,   5.0f );
    state.params.light.corner   = make_float3(  343.0f, 548.5f, 227.0f );
    state.params.light.v1       = make_float3(    0.0f,   0.0f, 105.0f );
    state.params.light.v2       = make_float3( -130.0f,   0.0f,   0.0f );
    state.params.light.normal   = normalize  ( cross( state.params.light.v1,  state.params.light.v2) );

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
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &params.accum_buffer ),
                            params.width*params.height*sizeof(float4) ) );
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, CutoutsState& state )
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
    CUDA_SYNC_CHECK();
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
    camera.setEye( make_float3( 278.0f, 273.0f, -900.0f ) );
    camera.setLookat( make_float3( 278.0f, 273.0f, 330.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 35.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ),
                                 make_float3( 0.0f, 0.0f, 1.0f ),
                                 make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock(true);
}


void createContext( CutoutsState& state )
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


void buildGeomAccel( CutoutsState& state )
{
    //
    // Build triangle GAS
    //
    {
        const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_vertices ), vertices_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.d_vertices ), g_vertices.data(), vertices_size_in_bytes,
                                cudaMemcpyHostToDevice ) );

        CUdeviceptr  d_mat_indices             = 0;
        const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_mat_indices ), mat_indices_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_mat_indices ), g_mat_indices.data(),
                                mat_indices_size_in_bytes, cudaMemcpyHostToDevice ) );

        const size_t tex_coords_size_in_bytes = g_tex_coords.size() * sizeof( float2 );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_tex_coords ), tex_coords_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.d_tex_coords ), g_tex_coords.data(),
                                tex_coords_size_in_bytes, cudaMemcpyHostToDevice ) );

        uint32_t triangle_input_flags[TRIANGLE_MAT_COUNT] = {
            // One per SBT record for this build input
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            // Do not disable anyhit on the cutout material for the short block
            OPTIX_GEOMETRY_FLAG_NONE
        };

        OptixBuildInput triangle_input                           = {};
        triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
        triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( g_vertices.size() );
        triangle_input.triangleArray.vertexBuffers               = &state.d_vertices;
        triangle_input.triangleArray.flags                       = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords               = TRIANGLE_MAT_COUNT;
        triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_mat_indices;
        triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
        triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &triangle_input,
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
        emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

        OPTIX_CHECK( optixAccelBuild(
                    state.context,
                    0,              // CUDA stream
                    &accel_options,
                    &triangle_input,
                    1,              // num build inputs
                    d_temp_buffer,
                    gas_buffer_sizes.tempSizeInBytes,
                    d_buffer_temp_output_gas_and_compacted_size,
                    gas_buffer_sizes.outputSizeInBytes,
                    &state.triangle_gas_handle,
                    &emitProperty,  // emitted property list
                    1               // num emitted properties
                    ) );

        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_mat_indices ) ) );

        size_t compacted_gas_size;
        CUDA_CHECK( cudaMemcpy(
                    &compacted_gas_size,
                    (void*)emitProperty.result,
                    sizeof( size_t ),
                    cudaMemcpyDeviceToHost
                    ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_triangle_gas_output_buffer ), compacted_gas_size ) );

            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( state.context, 0, state.triangle_gas_handle, state.d_triangle_gas_output_buffer, compacted_gas_size, &state.triangle_gas_handle ) );

            CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
        }
        else
        {
            state.d_triangle_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }

    //
    // Build sphere GAS
    //
    {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        // AABB build input
        OptixAabb   aabb = g_sphere.getAabb();
        CUdeviceptr d_aabb_buffer;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabb, sizeof( OptixAabb ),
                                cudaMemcpyHostToDevice ) );

        uint32_t sphere_input_flag = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput sphere_input                    = {};
        sphere_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        sphere_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
        sphere_input.customPrimitiveArray.numPrimitives = 1;
        sphere_input.customPrimitiveArray.flags         = &sphere_input_flag;
        sphere_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context,
                                                   &accel_options,
                                                   &sphere_input,
                                                   1,  // num_build_inputs
                                                   &gas_buffer_sizes ) );

        CUdeviceptr d_temp_buffer;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ), compactedSizeOffset + 8 ) );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

        OPTIX_CHECK( optixAccelBuild( state.context,
                                      0,        // CUDA stream
                                      &accel_options,
                                      &sphere_input,
                                      1,        // num build inputs
                                      d_temp_buffer,
                                      gas_buffer_sizes.tempSizeInBytes,
                                      d_buffer_temp_output_gas_and_compacted_size,
                                      gas_buffer_sizes.outputSizeInBytes,
                                      &state.sphere_gas_handle,
                                      &emitProperty,  // emitted property list
                                      1 ) );          // num emitted properties

        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_aabb_buffer ) ) );

        size_t compacted_gas_size;
        CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_sphere_gas_output_buffer ), compacted_gas_size ) );

            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( state.context, 0, state.sphere_gas_handle, state.d_sphere_gas_output_buffer, compacted_gas_size, &state.sphere_gas_handle ) );

            CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
        }
        else
        {
            state.d_sphere_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }
}


void buildInstanceAccel( CutoutsState& state )
{
    CUdeviceptr d_instances;
    size_t      instance_size_in_bytes = sizeof( OptixInstance ) * 2;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_instances ), instance_size_in_bytes ) );

    OptixBuildInput instance_input = {};

    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances;
    instance_input.instanceArray.numInstances = 2;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &instance_input,
                                               1,  // num build inputs
                                               &ias_buffer_sizes ) );

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), ias_buffer_sizes.tempSizeInBytes ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_ias_output_buffer ), ias_buffer_sizes.outputSizeInBytes ) );

    // Use the identity matrix for the instance transform
    Instance instance = { { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 } };

    OptixInstance optix_instances[2];
    memset( optix_instances, 0, instance_size_in_bytes );

    optix_instances[0].traversableHandle = state.triangle_gas_handle;
    optix_instances[0].flags             = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[0].instanceId        = 0;
    optix_instances[0].sbtOffset         = 0;
    optix_instances[0].visibilityMask    = 1;
    memcpy( optix_instances[0].transform, instance.transform, sizeof( float ) * 12 );

    optix_instances[1].traversableHandle = state.sphere_gas_handle;
    optix_instances[1].flags             = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[1].instanceId        = 1;
    optix_instances[1].sbtOffset         = TRIANGLE_MAT_COUNT*RAY_TYPE_COUNT;
    optix_instances[1].visibilityMask    = 1;
    memcpy( optix_instances[1].transform, instance.transform, sizeof( float ) * 12 );

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_instances ), &optix_instances, instance_size_in_bytes,
                            cudaMemcpyHostToDevice ) );

    OPTIX_CHECK( optixAccelBuild( state.context,
                                  0,  // CUDA stream
                                  &accel_options,
                                  &instance_input,
                                  1,  // num build inputs
                                  d_temp_buffer,
                                  ias_buffer_sizes.tempSizeInBytes,
                                  state.d_ias_output_buffer,
                                  ias_buffer_sizes.outputSizeInBytes,
                                  &state.ias_handle,
                                  nullptr,  // emitted property list
                                  0         // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_instances   ) ) );
}


void createModule( CutoutsState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount  = 100;

    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur            = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    state.pipeline_compile_options.numPayloadValues          = 2;
    state.pipeline_compile_options.numAttributeValues        = 4;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixCutouts.cu" );
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


void createProgramGroups( CutoutsState& state )
{
    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc    = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context,
                                              &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options,
                                              log, &sizeof_log,
                                              &state.raygen_prog_group ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
    sizeof_log                                  = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context,
                                              &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options,
                                              log, &sizeof_log,
                                              &state.radiance_miss_group ) );

    memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = nullptr;  // NULL miss program for occlusion rays
    miss_prog_group_desc.miss.entryFunctionName = nullptr;
    sizeof_log                                  = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.occlusion_miss_group ) );

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hit_prog_group_desc.hitgroup.moduleAH            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    hit_prog_group_desc.hitgroup.moduleIS            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context,
                                              &hit_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options,
                                              log,
                                              &sizeof_log,
                                              &state.radiance_hit_group ) );

    memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
    hit_prog_group_desc.hitgroup.moduleAH            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    hit_prog_group_desc.hitgroup.moduleIS            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    sizeof_log = sizeof( log );
    OPTIX_CHECK( optixProgramGroupCreate( state.context,
                                          &hit_prog_group_desc,
                                          1,  // num program groups
                                          &program_group_options,
                                          log,
                                          &sizeof_log,
                                          &state.occlusion_hit_group ) );
}


void createPipeline( CutoutsState& state )
{
    const uint32_t    max_trace_depth = 2;
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.radiance_miss_group,
        state.occlusion_miss_group,
        state.radiance_hit_group,
        state.occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace_depth;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate( state.context,
                                          &state.pipeline_compile_options,
                                          &pipeline_link_options,
                                          program_groups,
                                          sizeof( program_groups ) / sizeof( program_groups[0] ),
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
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                             0,  // maxCCDepth
                                             0,  // maxDCDEpth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}


void createSBT( CutoutsState& state )
{
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen_record ), raygen_record_size ) );

    RayGenRecord rg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );
    rg_sbt.data = {1.0f, 0.f, 0.f};

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_raygen_record ), &rg_sbt, raygen_record_size,
                            cudaMemcpyHostToDevice ) );

    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof( MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss_records ), miss_record_size * RAY_TYPE_COUNT ) );

    MissRecord ms_sbt[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_miss_group, &ms_sbt[0] ) );
    ms_sbt[0].data = {0.0f, 0.0f, 0.0f};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.occlusion_miss_group, &ms_sbt[1] ) );
    ms_sbt[1].data = {0.0f, 0.0f, 0.0f};

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_miss_records ), ms_sbt, miss_record_size * RAY_TYPE_COUNT,
                            cudaMemcpyHostToDevice ) );

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup_records ),
                            hitgroup_record_size * ( RAY_TYPE_COUNT * ( TRIANGLE_MAT_COUNT + SPHERE_MAT_COUNT ) ) ) );

    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * ( TRIANGLE_MAT_COUNT + SPHERE_MAT_COUNT )];

    // Set up the HitGroupRecords for the triangle materials
    for( int i = 0; i < TRIANGLE_MAT_COUNT; ++i )
    {
        {
            const int sbt_idx = i*RAY_TYPE_COUNT+0; // SBT for radiance ray-type for ith material

            OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[ sbt_idx ].data.emission_color = g_emission_colors[i];
            hitgroup_records[ sbt_idx ].data.diffuse_color  = g_diffuse_colors[i];
            hitgroup_records[ sbt_idx ].data.vertices       = reinterpret_cast<float4*>(state.d_vertices);
            hitgroup_records[ sbt_idx ].data.tex_coords     = reinterpret_cast<float2*>(state.d_tex_coords);
        }

        {
            const int sbt_idx = i*RAY_TYPE_COUNT+1; // SBT for occlusion ray-type for ith material
            memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

            OPTIX_CHECK( optixSbtRecordPackHeader( state.occlusion_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[ sbt_idx ].data.vertices   = reinterpret_cast<float4*>(state.d_vertices);
            hitgroup_records[ sbt_idx ].data.tex_coords = reinterpret_cast<float2*>(state.d_tex_coords);
        }
    }

    // Set up the HitGroupRecords for the sphere material
    {
        const int sbt_idx = TRIANGLE_MAT_COUNT * RAY_TYPE_COUNT+0; // SBT for radiance ray-type for sphere material

        OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_hit_group, &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.emission_color = g_sphere_emission_color;
        hitgroup_records[ sbt_idx ].data.diffuse_color  = g_sphere_diffuse_color;
        hitgroup_records[ sbt_idx ].data.sphere         = g_sphere;
    }

    {
        const int sbt_idx = TRIANGLE_MAT_COUNT * RAY_TYPE_COUNT+1; // SBT for occlusion ray-type for sphere material
        memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

        OPTIX_CHECK( optixSbtRecordPackHeader( state.occlusion_hit_group, &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.sphere = g_sphere;
    }

    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_hitgroup_records ), hitgroup_records,
                            hitgroup_record_size * ( RAY_TYPE_COUNT * ( TRIANGLE_MAT_COUNT + SPHERE_MAT_COUNT ) ),
                            cudaMemcpyHostToDevice ) );

    state.sbt.raygenRecord                = d_raygen_record;
    state.sbt.missRecordBase              = d_miss_records;
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    state.sbt.missRecordCount             = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase          = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    state.sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * ( TRIANGLE_MAT_COUNT + SPHERE_MAT_COUNT );
}


void cleanupState( CutoutsState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_miss_group ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_vertices ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_tex_coords ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_triangle_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_sphere_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_ias_output_buffer ) ) );
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    CutoutsState state;
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
            use_pbo = false;
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
        buildGeomAccel     ( state );
        buildInstanceAccel ( state );
        createModule       ( state );
        createProgramGroups( state );
        createPipeline     ( state );
        createSBT          ( state );
        initLaunchParams( state );


        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixCutouts", state.params.width, state.params.height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            glfwSetWindowUserPointer    ( window, &state.params         );

            //
            // Render loop
            //
            {
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
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers(window);

                    ++state.params.subframe_index;
                }
                while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        }
        else
        {
            if( use_pbo)
            {
                sutil::initGLFW(); // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, state.params.width, state.params.height);
            handleCameraUpdate(state.params);
            handleResize(output_buffer, state.params);
            launchSubframe(output_buffer, state);

            sutil::ImageBuffer buffer;
            buffer.data = output_buffer.getHostPointer();
            buffer.width = output_buffer.width();
            buffer.height = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::saveImage(outfile.c_str(), buffer, false);

            glfwTerminate();
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
