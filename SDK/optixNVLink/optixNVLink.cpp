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

/*
Sample Description:

If mutliple GPUs are present with nvlink, or other peer-to-peer access, between
them, then this connectivity can be used in several ways. This sample
demonstrates two of these:

Read-only buffers: nvlink capability divides the available GPUs into
nvlink "islands". Since the connection is so fast, some readable resources
can be shared so that one copy is held per island.  Typically, these
resources will be spread across the GPUs of an island so that the memory
burden is shared. This sample shares textures so that only one copy is
held per nvlink island.

The Frame buffer: (1) In the single-GPU case, a gl interop buffer can
be used to avoid data copies to OpenGL for display. (2) When multiple GPUs are
used which are not all nvlink connected, zero-copy memory, which transfers data
through the host, is typically used for the frame buffer. (3) When all of
GPUs reside in a single nvlink island, the link can be used to transfer
frame buffer data so that it does not need to be transferred through the
host.  This sample demonstrates all three of these techniques.
*/

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <nvml_configure.h> // configured file to tell if we have nvml
#if OPTIX_USE_NVML
#include <nvml.h>
#endif

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
#include <sutil/WorkDistribution.h>

#include <GLFW/glfw3.h>

#include "optixNVLink.h"

//------------------------------------------------------------------------------
//
//  Variables related to display
//
//------------------------------------------------------------------------------

bool              resize_dirty  = false;
bool              minimized     = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int2              mouse_prev_pos;
int32_t           mouse_button = -1;

int32_t           width  = 768;
int32_t           height = 768;
int32_t           samples_per_launch = 8;

// Output file name (empty means do not output a file)
std::string       g_outfile = "";

// How to scale the device color overlay on the image (0 means do not show)
float             g_device_color_scale = 1.0f;

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" void fillSamplesCUDA(
        int32_t  num_samples,
        cudaStream_t stream,
        int32_t  gpu_idx,
        int32_t  num_gpus,
        int32_t  width,
        int32_t  height,
        int2*    samples );

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

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;


struct Vertex
{
    float x, y, z, pad;
};


struct TexCoord
{
    float s, t;
};


struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};


struct Instance
{
    float transform[12];
};


struct PerDeviceSampleState
{
    int32_t                      device_idx               = -1;
    OptixDeviceContext           context                  = 0;

    OptixTraversableHandle       gas_handle               = 0;   // Traversable handle for triangle AS
    CUdeviceptr                  d_gas_output_buffer      = 0;   // Triangle AS memory
    CUdeviceptr                  d_vertices               = 0;
    CUdeviceptr                  d_tex_coords             = 0;

    OptixModule                  ptx_module               = 0;
    OptixPipelineCompileOptions  pipeline_compile_options = {};
    OptixPipeline                pipeline                 = 0;

    OptixProgramGroup            raygen_prog_group        = 0;
    OptixProgramGroup            radiance_miss_group      = 0;
    OptixProgramGroup            occlusion_miss_group     = 0;
    OptixProgramGroup            radiance_hit_group       = 0;
    OptixProgramGroup            occlusion_hit_group      = 0;

    OptixShaderBindingTable      sbt                      = {};

    int32_t                      num_samples              = 0;
    int2*                        d_sample_indices         = 0;
    float4*                      d_sample_accum           = 0;

    Params                       params;
    Params*                      d_params;

    CUstream                     stream                   = 0;

    uint32_t                     peers                    = 0;
};


//------------------------------------------------------------------------------
//
// Forward declarations
//
//------------------------------------------------------------------------------

cudaTextureObject_t getDiffuseTextureObject( int material_id, PerDeviceSampleState& pd_state );


//------------------------------------------------------------------------------
//
// Load NVML dynamically
//
//------------------------------------------------------------------------------
#if OPTIX_USE_NVML

bool g_nvmlLoaded = false;

#ifdef WIN32
#define APIFUNC FAR WINAPI
#else
#define APIFUNC
#endif

typedef nvmlReturn_t (APIFUNC *NVML_INIT_TYPE)();
NVML_INIT_TYPE nvmlInit_p;

typedef nvmlReturn_t (APIFUNC *NVML_DEVICE_GET_HANDLE_BY_INDEX_TYPE)( unsigned int, nvmlDevice_t* );
NVML_DEVICE_GET_HANDLE_BY_INDEX_TYPE nvmlDeviceGetHandleByIndex_p;

typedef nvmlReturn_t (APIFUNC *NVML_DEVICE_GET_PCI_INFO_TYPE)( nvmlDevice_t, nvmlPciInfo_t* );
NVML_DEVICE_GET_PCI_INFO_TYPE nvmlDeviceGetPciInfo_p;

typedef nvmlReturn_t (APIFUNC *NVML_DEVICE_GET_NVLINK_CAPABILITY_TYPE)( nvmlDevice_t, unsigned int, nvmlNvLinkCapability_t, unsigned int* );
NVML_DEVICE_GET_NVLINK_CAPABILITY_TYPE nvmlDeviceGetNvLinkCapability_p;

typedef nvmlReturn_t (APIFUNC *NVML_DEVICE_GET_NVLINK_STATE_TYPE)( nvmlDevice_t, unsigned int, nvmlEnableState_t* );
NVML_DEVICE_GET_NVLINK_STATE_TYPE nvmlDeviceGetNvLinkState_p;

typedef nvmlReturn_t (APIFUNC *NVML_DEVICE_GET_NVLINK_REMOTE_PCI_INFO_TYPE)( nvmlDevice_t, unsigned int, nvmlPciInfo_t* );
NVML_DEVICE_GET_NVLINK_REMOTE_PCI_INFO_TYPE nvmlDeviceGetNvLinkRemotePciInfo_p;

typedef nvmlReturn_t (APIFUNC *NVML_SYSTEM_GET_DRIVER_VERSION_TYPE)( char*, unsigned int );
NVML_SYSTEM_GET_DRIVER_VERSION_TYPE nvmlSystemGetDriverVersion_p;

#ifdef WIN32
void* loadDllHandle( const char* dllName )
{
    void* dllHandle = optixLoadWindowsDllFromName( dllName );
    return dllHandle;
}

void* getProcedureAddress( void* dllHandle, const char* funcName )
{
    void* proc = GetProcAddress( (HMODULE)dllHandle, funcName );
    if ( proc == NULL )
        std::cerr << funcName << " not found\n";
    return proc;
}

#else
void* loadSharedObjectHandle( const char* soName )
{
    void* soHandle = dlopen( soName, RTLD_NOW );
    return soHandle;
}

void* getProcedureAddress( void* handlePtr, const char* funcName )
{
    void* proc = dlsym( handlePtr, funcName );
    if( !proc )
        std::cerr << funcName << " not found\n";
    return proc;
}
#endif

static bool loadNvmlFunctions()
{
    // Load the library
#ifdef WIN32
    const char* soName = "nvml.dll";
    void* handle = loadDllHandle( soName );
#else
    const char* soName = "libnvidia-ml.so";
    void* handle = loadSharedObjectHandle( soName );
#endif

    if ( !handle )
    {
        std::cout << "UNABLE TO LOAD " << soName << "\n";
        return false;
    }

    // Set the individual _ptions we are using
    nvmlInit_p = reinterpret_cast<NVML_INIT_TYPE>( getProcedureAddress( handle, "nvmlInit" ) );
    nvmlDeviceGetHandleByIndex_p = reinterpret_cast<NVML_DEVICE_GET_HANDLE_BY_INDEX_TYPE>( getProcedureAddress( handle, "nvmlDeviceGetHandleByIndex" ) );
    nvmlDeviceGetPciInfo_p = reinterpret_cast<NVML_DEVICE_GET_PCI_INFO_TYPE>( getProcedureAddress( handle, "nvmlDeviceGetPciInfo" ) );
    nvmlDeviceGetNvLinkCapability_p = reinterpret_cast<NVML_DEVICE_GET_NVLINK_CAPABILITY_TYPE>( getProcedureAddress( handle, "nvmlDeviceGetNvLinkCapability" ) );
    nvmlDeviceGetNvLinkState_p = reinterpret_cast<NVML_DEVICE_GET_NVLINK_STATE_TYPE>( getProcedureAddress( handle, "nvmlDeviceGetNvLinkState" ) );
    nvmlDeviceGetNvLinkRemotePciInfo_p = reinterpret_cast<NVML_DEVICE_GET_NVLINK_REMOTE_PCI_INFO_TYPE>( getProcedureAddress( handle, "nvmlDeviceGetNvLinkRemotePciInfo" ) );
    nvmlSystemGetDriverVersion_p = reinterpret_cast<NVML_SYSTEM_GET_DRIVER_VERSION_TYPE>( getProcedureAddress( handle, "nvmlSystemGetDriverVersion" ) );

    std::cout << "LOADED " << soName << "\n";
    return true;
}

#endif // OPTIX_USE_NVML

//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

const int32_t  TRIANGLE_COUNT  = 32;
const int32_t  MAT_COUNT       = 4;

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


const static std::array<TexCoord, TRIANGLE_COUNT*3> g_tex_coords =
{ {
    // Floor  -- white lambert
    {    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

    // Ceiling -- white lambert
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

    // Back wall -- white lambert
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

    // Right wall -- green lambert
	{    0.0f,    0.0f },
	{    0.0f,    1.0f },
	{    1.0f,    1.0f },

	{    0.0f,    0.0f },
	{    1.0f,    1.0f },
	{    1.0f,    0.0f },

    // Left wall -- red lambert
	{    0.0f,    0.0f },
	{    1.0f,    0.0f },
	{    1.0f,    1.0f },

	{    0.0f,    0.0f },
	{    1.0f,    1.0f },
	{    0.0f,    1.0f },


    // Short block -- white lambert
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

    // Tall block -- white lambert
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

    // Ceiling light -- emmissive
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f },

	{    0.0f,    0.0f },
	{    0.0f,    0.0f },
	{    0.0f,    0.0f }
} };


static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices =
{ {
    0, 0,                           // Floor         -- white lambert
    0, 0,                           // Ceiling       -- white lambert
    0, 0,                           // Back wall     -- white lambert
    1, 1,                           // Right wall    -- green lambert
    2, 2,                           // Left wall     -- red lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   // Short block   -- white lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   // Tall block    -- white lambert
    3, 3                            // Ceiling light -- emmissive
} };


const std::array<float3, MAT_COUNT> g_emission_colors =
{ {
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    { 15.0f, 15.0f,  5.0f }

} };


const std::array<float3, MAT_COUNT> g_diffuse_colors =
{ {
    { 0.80f, 0.80f, 0.80f },
    { 0.05f, 0.80f, 0.05f },
    { 0.80f, 0.05f, 0.05f },
    { 0.50f, 0.00f, 0.00f }
} };


//------------------------------------------------------------------------------
//
// Texture tracking
//
//------------------------------------------------------------------------------

// Materials for which textures will be made
std::array<int, MAT_COUNT> g_make_diffuse_textures = {0, 1, 1, 0};

// Backing storage for the textures. These will be shared per
// P2P island when sharing is enabled.
std::array<std::vector<cudaArray_t>, MAT_COUNT> g_diffuse_texture_data;

// Texture objects. Each device must have a texture object for
// each texture, but the backing stores can be shared.
std::array<std::vector<cudaTextureObject_t>, MAT_COUNT> g_diffuse_textures;

// Texture memory usage per device
std::vector<float> g_device_tex_usage;

// Kinds of connections between devices to accept as peers
const int PEERS_NONE   = 0;
const int PEERS_NVLINK = 1;
const int PEERS_ALL    = 2;

// Configuration decisions
int g_peer_usage = PEERS_NVLINK;
bool g_share_textures = true;
bool g_optimize_framebuffer = true;

const int TEXTURE_WIDTH = 1024;

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
    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
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

    width          = res_x;
    height         = res_y;
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
//
//------------------------------------------------------------------------------


void printUsageAndExit( const char* argv0 )
{
    std::cout <<  "Usage  : " << argv0 << " [options]\n";
    std::cout <<  "Options: --launch-samples | -s         Number of samples per pixel per launch (default 8)\n";
    std::cout <<  "         --file | -f                   Output file name\n";
    std::cout <<  "         --device-color-scale | -d     Device color overlay scale (default 1.0)\n";
    std::cout <<  "         --peers | -p                  P2P connections to include [none, nvlink, all] (default nvlink)\n";
    std::cout <<  "         --optimize-framebuffer | -o   Optimize the framebuffer for speed [true, false] (default true)\n";
    std::cout <<  "         --share-textures | -t         Share textures if allowed [true, false] (default true)\n";
    std::cout <<  "         --help | -h                   Print this usage message\n";

    exit( 0 );
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
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock(true);
}


void initLaunchParams( PerDeviceSampleState& pd_state )
{
    pd_state.params.subframe_index     = 0u;
    pd_state.params.width              = width;
    pd_state.params.height             = height;
    pd_state.params.samples_per_launch = samples_per_launch;
    pd_state.params.device_idx         = pd_state.device_idx;

    pd_state.params.light.emission     = make_float3(   15.0f,  15.0f,   5.0f );
    pd_state.params.light.corner       = make_float3(  343.0f, 548.5f, 227.0f );
    pd_state.params.light.v1           = make_float3(    0.0f,   0.0f, 105.0f );
    pd_state.params.light.v2           = make_float3( -130.0f,   0.0f,   0.0f );
    pd_state.params.light.normal       = normalize  ( cross( pd_state.params.light.v1,  pd_state.params.light.v2) );
    pd_state.params.handle             = pd_state.gas_handle;

    pd_state.params.device_color_scale = g_device_color_scale;

    // IO buffers are assigned in allocIOBuffers

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &pd_state.d_params), sizeof( Params ) ) );
}


void allocIOBuffers( PerDeviceSampleState& pd_state, int num_gpus )
{
    StaticWorkDistribution wd;
    wd.setRasterSize( width, height );
    wd.setNumGPUs( num_gpus );

    pd_state.num_samples = wd.numSamples( pd_state.device_idx );

    CUDA_CHECK( cudaSetDevice( pd_state.device_idx ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.d_sample_indices ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.d_sample_accum   ) ) );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &pd_state.d_sample_indices ), pd_state.num_samples*sizeof( int2   ) ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &pd_state.d_sample_accum   ), pd_state.num_samples*sizeof( float4 ) ) );

    pd_state.params.sample_index_buffer  = pd_state.d_sample_indices;
    pd_state.params.sample_accum_buffer  = pd_state.d_sample_accum;
    pd_state.params.result_buffer        = 0; // Managed by CUDAOutputBuffer

    fillSamplesCUDA(
            pd_state.num_samples,
            pd_state.stream,
            pd_state.device_idx,
            num_gpus,
            width,
            height,
            pd_state.d_sample_indices
            );
}


void handleCameraUpdate( std::vector<PerDeviceSampleState>& pd_states )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( width ) / static_cast<float>( height ) );
    float3 u, v, w;
    camera.UVWFrame( u, v, w );

    for( PerDeviceSampleState& pd_state : pd_states )
    {
        pd_state.params.eye = camera.eye();
        pd_state.params.U   = u;
        pd_state.params.V   = v;
        pd_state.params.W   = w;
    }
}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, std::vector<PerDeviceSampleState>& pd_states )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    CUDA_CHECK( cudaSetDevice( pd_states.front().device_idx ) );
    output_buffer.resize( width, height );

    // Realloc accumulation buffer
    for( PerDeviceSampleState& pd_state : pd_states )
    {
        pd_state.params.width  = width;
        pd_state.params.height = height;
        allocIOBuffers( pd_state, static_cast<int>( pd_states.size() ) );
    }
}


void updateDeviceStates( sutil::CUDAOutputBuffer<uchar4>& output_buffer, std::vector<PerDeviceSampleState>& pd_states )
{
    // Update params on devices
    if( camera_changed || resize_dirty )
        for( PerDeviceSampleState& pd_state : pd_states )
            pd_state.params.subframe_index = 0;

    handleCameraUpdate( pd_states );
    handleResize( output_buffer, pd_states );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, std::vector<PerDeviceSampleState>& pd_states )
{
    uchar4* result_buffer_data = output_buffer.map();
    for( PerDeviceSampleState& pd_state : pd_states )
    {
        // Launch
        pd_state.params.result_buffer = result_buffer_data;
        CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( pd_state.d_params ),
                    &pd_state.params,
                    sizeof( Params ),
                    cudaMemcpyHostToDevice,
                    pd_state.stream
                    ) );

        OPTIX_CHECK( optixLaunch(
                    pd_state.pipeline,
                    pd_state.stream,
                    reinterpret_cast<CUdeviceptr>( pd_state.d_params ),
                    sizeof( Params ),
                    &pd_state.sbt,
                    pd_state.num_samples,  // launch width
                    1,                  // launch height
                    1                   // launch depth
                    ) );
    }
    output_buffer.unmap();
    for( PerDeviceSampleState& pd_state : pd_states )
    {
        CUDA_CHECK( cudaSetDevice( pd_state.device_idx ) );
        CUDA_SYNC_CHECK();
    }
}


void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );

    GLuint pbo = output_buffer.getPBO();
    gl_display.display( width, height, framebuf_res_x, framebuf_res_y, pbo );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}


void createContext( PerDeviceSampleState& pd_state )
{
    // Initialize CUDA on this device
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    pd_state.context = context;

    CUDA_CHECK( cudaStreamCreate( &pd_state.stream ) );
}


void createContexts( std::vector<PerDeviceSampleState>& pd_state )
{
    OPTIX_CHECK( optixInit() );

    int32_t device_count = 0;
    CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
    pd_state.resize( device_count );
    std::cout << "TOTAL VISIBLE GPUs: " << device_count << std::endl;

    cudaDeviceProp prop;
    for( int i = 0; i < device_count; ++i )
    {
        // note: the device index must be the same as the position in the state vector
        pd_state[i].device_idx = i;
        CUDA_CHECK( cudaGetDeviceProperties ( &prop, i ) );
        CUDA_CHECK( cudaSetDevice( i ) );
        std::cout << "GPU [" << i << "]: " << prop.name << std::endl;

        createContext( pd_state[i] );
    }
}

void uploadAdditionalShadingData( PerDeviceSampleState& pd_state )
{
    // texture coordinates
    const size_t tex_coords_size_in_bytes = g_tex_coords.size() * sizeof( TexCoord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &pd_state.d_tex_coords ), tex_coords_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( pd_state.d_tex_coords ),
                g_tex_coords.data(),
                tex_coords_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
}


void buildMeshAccel( PerDeviceSampleState& pd_state )
{
    //
    // copy mesh data to device
    //
    const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &pd_state.d_vertices ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( pd_state.d_vertices ),
                g_vertices.data(),
                vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr d_mat_indices = 0;
    const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_mat_indices ),
                mat_indices_size_in_bytes
                ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_mat_indices),
                g_mat_indices.data(),
                mat_indices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    //
    // Build triangle GAS
    //
    uint32_t triangle_input_flags[MAT_COUNT] =
    {
        // One per SBT record for this build input
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    };

    OptixBuildInput triangle_input = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
    triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( g_vertices.size() );
    triangle_input.triangleArray.vertexBuffers               = &pd_state.d_vertices;
    triangle_input.triangleArray.flags                       = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords               = MAT_COUNT;
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags            = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation             = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                pd_state.context,
                &accel_options,
                &triangle_input,
                1,  // num_build_inputs
                &gas_buffer_sizes
                ) );

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

    OPTIX_CHECK( optixAccelBuild( pd_state.context,
                                  0,                  // CUDA stream
                                  &accel_options,
                                  &triangle_input,
                                  1,                  // num build inputs
                                  d_temp_buffer,
                                  gas_buffer_sizes.tempSizeInBytes,
                                  d_buffer_temp_output_gas_and_compacted_size,
                                  gas_buffer_sizes.outputSizeInBytes,
                                  &pd_state.gas_handle,
                                  &emitProperty,      // emitted property list
                                  1                   // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_mat_indices ) ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &pd_state.d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( pd_state.context, 0, pd_state.gas_handle, pd_state.d_gas_output_buffer,
                                        compacted_gas_size, &pd_state.gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        pd_state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}


void createModule( PerDeviceSampleState& pd_state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    pd_state.pipeline_compile_options.usesMotionBlur            = false;
    pd_state.pipeline_compile_options.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pd_state.pipeline_compile_options.numPayloadValues          = 2;
    pd_state.pipeline_compile_options.numAttributeValues        = 2;
    pd_state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pd_state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixNVLink.cu" );
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                pd_state.context,
                &module_compile_options,
                &pd_state.pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &pd_state.ptx_module
                ) );
}


void createProgramGroups( PerDeviceSampleState& pd_state )
{
    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = pd_state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                pd_state.context,
                &raygen_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &pd_state.raygen_prog_group
                ) );

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = pd_state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                pd_state.context,
                &miss_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &pd_state.radiance_miss_group
                ) );

    memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = nullptr;  // NULL miss program for occlusion rays
    miss_prog_group_desc.miss.entryFunctionName = nullptr;
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                pd_state.context,
                &miss_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &pd_state.occlusion_miss_group
                ) );

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = pd_state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                pd_state.context,
                &hit_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &pd_state.radiance_hit_group
                ) );

    memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = pd_state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
    OPTIX_CHECK( optixProgramGroupCreate(
                pd_state.context,
                &hit_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &pd_state.occlusion_hit_group
                ) );

}


void createPipeline( PerDeviceSampleState& pd_state )
{
    const uint32_t max_trace_depth = 2;

    OptixProgramGroup program_groups[] =
    {
        pd_state.raygen_prog_group,
        pd_state.radiance_miss_group,
        pd_state.occlusion_miss_group,
        pd_state.radiance_hit_group,
        pd_state.occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = max_trace_depth;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                pd_state.context,
                &pd_state.pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                log,
                &sizeof_log,
                &pd_state.pipeline
                ) );

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
    OPTIX_CHECK( optixPipelineSetStackSize( pd_state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}


void createSBT( PerDeviceSampleState& pd_state )
{
    CUdeviceptr   d_raygen_record;
    const size_t  raygen_record_size = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen_record ), raygen_record_size ) );

    RayGenRecord rg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( pd_state.raygen_prog_group, &rg_sbt ) );
    rg_sbt.data = {1.0f, 0.f, 0.f};

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );


    CUdeviceptr   d_miss_records;
    const size_t  miss_record_size = sizeof( MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss_records ), miss_record_size*RAY_TYPE_COUNT ) );

    MissRecord ms_sbt[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( pd_state.radiance_miss_group, &ms_sbt[0] ) );
    ms_sbt[0].data = {0.0f, 0.0f, 0.0f};
    OPTIX_CHECK( optixSbtRecordPackHeader( pd_state.occlusion_miss_group, &ms_sbt[1] ) );
    ms_sbt[1].data = {0.0f, 0.0f, 0.0f};

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_miss_records ),
                ms_sbt,
                miss_record_size*RAY_TYPE_COUNT,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr   d_hitgroup_records;
    const size_t  hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_hitgroup_records ),
                hitgroup_record_size*RAY_TYPE_COUNT*MAT_COUNT
                ) );
    HitGroupRecord hitgroup_records[ RAY_TYPE_COUNT*MAT_COUNT ];

    for( int i = 0; i < MAT_COUNT; ++i )
    {
        {
            const int sbt_idx = i*RAY_TYPE_COUNT+0; // SBT for radiance ray-type for ith material

            OPTIX_CHECK( optixSbtRecordPackHeader( pd_state.radiance_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[ sbt_idx ].data.emission_color  = g_emission_colors[i];
            hitgroup_records[ sbt_idx ].data.diffuse_color   = g_diffuse_colors[i];
            hitgroup_records[ sbt_idx ].data.vertices        = reinterpret_cast<float4*>(pd_state.d_vertices);
            hitgroup_records[ sbt_idx ].data.tex_coords      = reinterpret_cast<float2*>(pd_state.d_tex_coords);
            hitgroup_records[ sbt_idx ].data.diffuse_texture = getDiffuseTextureObject( i, pd_state );
        }

        {
            const int sbt_idx = i*RAY_TYPE_COUNT+1; // SBT for occlusion ray-type for ith material
            memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

            OPTIX_CHECK( optixSbtRecordPackHeader( pd_state.occlusion_hit_group, &hitgroup_records[sbt_idx] ) );
        }
    }

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_hitgroup_records ),
                hitgroup_records,
                hitgroup_record_size*RAY_TYPE_COUNT*MAT_COUNT,
                cudaMemcpyHostToDevice
                ) );

    pd_state.sbt.raygenRecord                = d_raygen_record;
    pd_state.sbt.missRecordBase              = d_miss_records;
    pd_state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    pd_state.sbt.missRecordCount             = RAY_TYPE_COUNT;
    pd_state.sbt.hitgroupRecordBase          = d_hitgroup_records;
    pd_state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    pd_state.sbt.hitgroupRecordCount         = RAY_TYPE_COUNT*MAT_COUNT;
}


void cleanupState( PerDeviceSampleState& pd_state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( pd_state.pipeline            ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( pd_state.raygen_prog_group   ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( pd_state.radiance_miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( pd_state.radiance_hit_group  ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( pd_state.occlusion_hit_group ) );
    OPTIX_CHECK( optixModuleDestroy       ( pd_state.ptx_module          ) );
    OPTIX_CHECK( optixDeviceContextDestroy( pd_state.context             ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.d_vertices             ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.d_gas_output_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.d_sample_indices       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.d_sample_accum         ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pd_state.d_params               ) ) );
}


//------------------------------------------------------------------------------
//
// Texture management functions
//
//------------------------------------------------------------------------------

void destroyTextures( std::vector<PerDeviceSampleState>& pd_states )
{
    for( int i = 0; i < static_cast<int>( pd_states.size() ); ++i )
    {
        int device_idx = pd_states[i].device_idx;
        CUDA_CHECK( cudaSetDevice( device_idx ) );
        for( int material_id = 0; material_id < MAT_COUNT; ++material_id )
        {
            if( g_diffuse_textures[material_id][device_idx] != 0 )
                CUDA_CHECK( cudaDestroyTextureObject( g_diffuse_textures[material_id][device_idx] ) );

            if( g_diffuse_texture_data[material_id][device_idx] != 0 )
                CUDA_CHECK( cudaFreeArray( g_diffuse_texture_data[material_id][device_idx] ) );
        }
    }
}


cudaTextureObject_t getDiffuseTextureObject( int material_id, PerDeviceSampleState& pd_state )
{
    // If the device has a texture on it, use that one
    int device_idx = pd_state.device_idx;
    if( g_diffuse_textures[material_id][device_idx] != 0 )
        return g_diffuse_textures[material_id][device_idx];

    // Otherwise, try to find a texture on the same island as the device

    int island         = pd_state.peers | ( 1 << device_idx );
    size_t num_devices = g_diffuse_textures[material_id].size();
    for( int peer_id = 0; peer_id < static_cast<int>( num_devices ); ++peer_id )
    {
        bool peer_in_island         = ( island & ( 1 << peer_id ) ) != 0;
        bool texture_exists_on_peer = ( g_diffuse_textures[material_id][peer_id] != 0 );

        if( peer_in_island && texture_exists_on_peer )
            return g_diffuse_textures[material_id][peer_id];
    }

    return 0;
}


void createTextureImageOnHost( float4* image_data, int width, int height, int material_id )
{
    int tiles_per_side = 8;

    for( int j = 0; j < height; j++ )
    {
        for( int i = 0; i < width; i++ )
        {
            // texture coordinates of pixel
            float s = i / (float)width;
            float t = j / (float)height;

            // texture coordinates within the current tile
            float ss = ( s * tiles_per_side ) - static_cast<int>( s * tiles_per_side );
            float tt = ( t * tiles_per_side ) - static_cast<int>( t * tiles_per_side );

            // use L-norm distance from center of tile to vary shape
            float n = material_id + 0.1f;  // L-norm
            float d = powf( powf( fabs( ss - 0.5f ), n ) + powf( fabs( tt - 0.5f ), n ), 1.0f / n ) * 2.03f;
            d       = ( d < 1.0f ) ? 1.0f - powf( d, 80.0f ) : 0.0f;

            image_data[j * width + i] = {d * s, d * t, 0.3f * ( 1.0f - d ), 0.0f};
        }
    }
}


cudaTextureObject_t defineTextureOnDevice( int device_idx, cudaArray_t tex_array, int tex_width, int tex_height )
{
    CUDA_CHECK( cudaSetDevice( device_idx ) );

    cudaResourceDesc res_desc;
    std::memset( &res_desc, 0, sizeof( cudaResourceDesc ) );
    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = tex_array;

    cudaTextureDesc tex_desc;
    std::memset( &tex_desc, 0, sizeof( cudaTextureDesc ) );
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    cudaResourceViewDesc* res_view_desc = nullptr;

    cudaTextureObject_t tex;
    CUDA_CHECK( cudaCreateTextureObject( &tex, &res_desc, &tex_desc, res_view_desc ) );

    return tex;
}


float loadTextureOnDevice( int mat_index, int device_idx )
{
    std::cout << "LOADING TEXTURE: material " << mat_index << " on device " << device_idx << ".\n";
    CUDA_CHECK( cudaSetDevice( device_idx ) );

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindFloat );

    const int tex_width  = TEXTURE_WIDTH;
    int       tex_height = tex_width;
    CUDA_CHECK( cudaMallocArray( &g_diffuse_texture_data[mat_index][device_idx], &channel_desc, tex_width, tex_height ) );

    std::vector<float4> h_texture_data( tex_width * tex_height );
    createTextureImageOnHost( h_texture_data.data(), tex_width, tex_height, mat_index );
    int width_in_bytes = tex_width * sizeof( float4 );
    int pitch        = width_in_bytes;
    CUDA_CHECK( cudaMemcpy2DToArray( g_diffuse_texture_data[mat_index][device_idx], 0, 0, h_texture_data.data(), pitch,
                                     width_in_bytes, tex_height, cudaMemcpyHostToDevice ) );

    g_diffuse_textures[mat_index][device_idx] =
        defineTextureOnDevice( device_idx, g_diffuse_texture_data[mat_index][device_idx], tex_width, tex_height );

    float tex_mem_usage = static_cast<float>( tex_width * tex_height * sizeof( float4 ) );
    return tex_mem_usage;
}


int getIslandDeviceWithLowestTextureUsage( int island )
{
    int   min_device_id    = 0;
    float min_device_usage = FLT_MAX;

    int device_idx = 0;
    while( ( 1 << device_idx ) <= island )
    {
        bool device_in_island = ( island & ( 1 << device_idx ) ) != 0;
        bool mem_usage_lower  = g_device_tex_usage[device_idx] < min_device_usage;

        if( device_in_island && mem_usage_lower )
        {
            min_device_usage = g_device_tex_usage[device_idx];
            min_device_id    = device_idx;
        }
        device_idx++;
    }

    return min_device_id;
}


float loadTexture( std::vector<PerDeviceSampleState>& pd_states, std::vector<int>& p2p_islands, int mat_index )
{
    bool share_per_island = ( g_peer_usage != PEERS_NONE && g_share_textures );
    float tex_mem = 0.0f;

    if( share_per_island == true )
    {
        // Load the texture on one of the devices
        for( int i = 0; i < static_cast<int>( p2p_islands.size() ); ++i )
        {
            int   device_idx = getIslandDeviceWithLowestTextureUsage( p2p_islands[i] );
            float tex_size   = loadTextureOnDevice( mat_index, device_idx );
            g_device_tex_usage[device_idx] += tex_size;
            tex_mem += tex_size;

            // Make texture samplers for each device in the island, but reuse the data array
            int island = p2p_islands[i];
            int peer_idx = 0;
            cudaArray_t tex_array = g_diffuse_texture_data[mat_index][device_idx];
            while( ( 1 << peer_idx ) <= island )
            {
                // If peer_idx is a peer of device_idx
                if ( (peer_idx != device_idx) && (island & (1 << peer_idx)) )
                {
                    g_diffuse_textures[mat_index][peer_idx] =
                        defineTextureOnDevice( peer_idx, tex_array, TEXTURE_WIDTH, TEXTURE_WIDTH );

                }
                peer_idx++;
            }
        }
    }
    else
    {
        for( int i = 0; i < static_cast<int>( pd_states.size() ); ++i )
        {
            int   device_idx = pd_states[i].device_idx;
            float tex_size   = loadTextureOnDevice( mat_index, device_idx );
            g_device_tex_usage[device_idx] += tex_size;
            tex_mem += tex_size;
        }
    }

    return tex_mem;
}


void loadTextures( std::vector<PerDeviceSampleState>& pd_states, std::vector<int>& p2p_islands )
{
    size_t num_devices = pd_states.size();
    g_device_tex_usage.resize( num_devices, 0.0f );
    float total_tex_mem = 0.0f;

    for( int mat_index = 0; mat_index < MAT_COUNT; ++mat_index )
    {
        g_diffuse_texture_data[mat_index].resize( num_devices, 0 );
        g_diffuse_textures[mat_index].resize( num_devices, 0 );

        // If a texture is required for this material, make it
        if( g_make_diffuse_textures[mat_index] )
        {
            total_tex_mem += loadTexture( pd_states, p2p_islands, mat_index );
        }
    }

    std::cout << "TEXTURE MEMORY USAGE: " << (total_tex_mem / (1<<20)) << " MB\n";
}


//------------------------------------------------------------------------------
//
// P2P / NVLINK functions
//
//------------------------------------------------------------------------------

int getGlInteropDeviceId( int num_devices )
{
    for ( int device_idx=0; device_idx < num_devices; ++device_idx )
    {
        int is_display_device = 0;
        CUDA_CHECK( cudaDeviceGetAttribute( &is_display_device, cudaDevAttrKernelExecTimeout, device_idx ) );
        if (is_display_device)
        {
            std::cout << "DISPLAY DEVICE: " << device_idx << "\n";
            return device_idx;
        }
    }

    std::cerr << "ERROR: Could not determine GL interop device\n";
    return -1;
}


void enablePeerAccess( std::vector<PerDeviceSampleState>& pd_states )
{
    size_t num_devices = pd_states.size();
    for( int device_idx = 0; device_idx < static_cast<int>( num_devices ); ++device_idx )
    {
        CUDA_CHECK( cudaSetDevice( device_idx ) );
        for( int peer_idx = 0; peer_idx < static_cast<int>( num_devices ); ++peer_idx )
        {
            if (peer_idx == device_idx)
                continue;

            int access = 0;
            cudaDeviceCanAccessPeer(&access, device_idx, peer_idx);
            if (access)
                cudaDeviceEnablePeerAccess( peer_idx, 0 );
        }
    }
}


void shutdownPeerAccess( std::vector<PerDeviceSampleState>& pd_states )
{
    size_t num_devices = pd_states.size();
    for( int device_idx = 0; device_idx < static_cast<int>( num_devices ); ++device_idx )
    {
        CUDA_CHECK( cudaSetDevice( device_idx ) );
        for( int peer_idx = 0; peer_idx < static_cast<int>( num_devices ); ++peer_idx )
        {
            if ( (1<<peer_idx) != 0 )
                cudaDeviceDisablePeerAccess( peer_idx );
        }
    }
}


#if OPTIX_USE_NVML
nvmlDevice_t getNvmlDeviceHandle( PerDeviceSampleState& pd_state )
{
    nvmlDevice_t device = nullptr;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex_p( pd_state.device_idx, &device );
    if( result != NVML_SUCCESS )
        std::cerr << "Could not get device handle for index " << pd_state.device_idx << "\n";
    return device;
}
#endif

#if OPTIX_USE_NVML
std::string getPciBusId( PerDeviceSampleState& pd_state )
{
    nvmlDevice_t device = getNvmlDeviceHandle( pd_state );
    if( device == nullptr )
        return "";

    nvmlPciInfo_t pci_info;
    memset( &pci_info, 0, sizeof( pci_info ) );
    nvmlReturn_t  result = nvmlDeviceGetPciInfo_p( device, &pci_info );
    if( NVML_SUCCESS != result )
        return "";

    return std::string( pci_info.busId );
}
#endif

void printIsland( int island )
{
    std::cout << "{";
    int device_idx = 0;
    while ( (1<<device_idx) <= island )
    {
        if ( (1<<device_idx) & island )
        {
            std::cout << device_idx;
            if ( (1<<(device_idx+1)) <= island )
                std::cout << ",";
        }
        device_idx++;
    }
    std::cout << "} ";
}


void computeP2PIslands( std::vector<PerDeviceSampleState>& pd_states, std::vector<int>& islands )
{
    std::cout << "P2P ISLANDS: ";
    islands.clear();
    for( int i = 0; i < static_cast<int>( pd_states.size() ); ++i )
    {
        int island = pd_states[i].peers | ( 1 << pd_states[i].device_idx );
        if( std::find( islands.begin(), islands.end(), island ) == islands.end() )
        {
            islands.push_back( island );
            printIsland( island );
        }
    }
    std::cout << "\n";
}

void findPeersForDevice( std::vector<PerDeviceSampleState>& pd_states, int device_idx, bool require_nvlink )
{

#if OPTIX_USE_NVML
    // Clear the set of peers for the current device
    pd_states[device_idx].peers = 0;

    nvmlReturn_t result;
    nvmlDevice_t device     = getNvmlDeviceHandle( pd_states[device_idx] );
    std::string  pci_bus_id = getPciBusId( pd_states[device_idx] );

    // Check each link
    for( unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link )
    {
        // Check if P2P is supported on this link
        unsigned int capResult = 0;
        result = nvmlDeviceGetNvLinkCapability_p( device, link, NVML_NVLINK_CAP_P2P_SUPPORTED, &capResult );
        if( result != NVML_SUCCESS || capResult == 0 )
            continue;

        // Check if NVLINK is active on this link (if required)
        if( require_nvlink )
        {
            nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;
            result                     = nvmlDeviceGetNvLinkState_p( device, link, &isActive );
            if( result != NVML_SUCCESS || isActive != NVML_FEATURE_ENABLED )
                continue;
        }

        // Check if we're connected to another device on this link
        nvmlPciInfo_t pci = {{0}};
        result            = nvmlDeviceGetNvLinkRemotePciInfo_p( device, link, &pci );
        if( result != NVML_SUCCESS )
            continue;


        // Find neighbors with the same id as the device we are connected to
        // and add them as peers
        std::string pci_id( pci.busId );
        bool        found = false;
        for( int i = 0; i < static_cast<int>( pd_states.size() ); ++i )
        {
            std::string peerPciId = getPciBusId( pd_states[i] );
            if( std::string( pci.busId ) == pci_id )
            {
                int peer_idx = pd_states[i].device_idx;
                pd_states[device_idx].peers |= ( 1 << peer_idx );
                found = true;
                //break;
            }
        }
        if( !found )
            std::cerr << "Unable to locate device with id " << pci_id << " in active devices.\n";
    }

#else
    if ( require_nvlink == true && device_idx == 0 )
        std::cout << "NVML NOT SUPPORTED. Cannot query nvlink. Treating all P2P connections as nvlink.\n";

    size_t num_devices = pd_states.size();
    CUDA_CHECK( cudaSetDevice( device_idx ) );
    for( int peer_idx = 0; peer_idx < static_cast<int>( num_devices ); ++peer_idx )
    {
        if( peer_idx == device_idx )
            continue;

        int access = 0;
        cudaDeviceCanAccessPeer( &access, device_idx, peer_idx );
        if ( access )
            pd_states[device_idx].peers |= ( 1 << peer_idx );
    }
#endif

}


void findPeers( std::vector<PerDeviceSampleState>& pd_states, bool require_nvlink )
{
    for( int i = 0; i < static_cast<int>( pd_states.size() ); ++i )
        findPeersForDevice( pd_states, i, require_nvlink );
}


#if OPTIX_USE_NVML
void initializeNvml()
{
    CUDA_CHECK( cudaFree( 0 ) );  // Make sure cuda is initialized first
    nvmlReturn_t result = nvmlInit_p();

    if( result != NVML_SUCCESS )
        std::cerr << "ERROR: nvmlInit() failed (code " << result << ")\n";

    char buff[1024];
    result = nvmlSystemGetDriverVersion_p( buff, 1024 );

    if( result == NVML_SUCCESS )
        std::cout << "DRIVER VERSION: " << buff << "\n";
    else
        std::cerr << "ERROR: Unable to get driver version (code " << result << ")\n";
}
#endif


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void parseCommandLine( int argc, char* argv[] )
{
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
            g_outfile = argv[++i];
        }
        else if( arg == "--launch-samples" || arg == "-s" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            samples_per_launch = atoi( argv[++i] );
        }
        else if( arg == "--device-color-scale" || arg == "-d" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            g_device_color_scale = static_cast<float>( atof( argv[++i] ) );
        }
        else if ( arg == "--peers" || arg == "-p" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            const std::string arg1 = argv[++i];
            if (arg1 == "none")
                g_peer_usage = PEERS_NONE;
            else if (arg1 == "nvlink")
                g_peer_usage = PEERS_NVLINK;
            else if (arg1 == "all")
                g_peer_usage = PEERS_ALL;
            else
                printUsageAndExit( argv[0] );
        }
        else if ( arg == "--optimize-framebuffer" || arg == "-o" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            const std::string arg1 = argv[++i];
            if (arg1 == "false")
                g_optimize_framebuffer = false;
            else if (arg1 == "true")
                g_optimize_framebuffer = true;
            else
                printUsageAndExit( argv[0] );
        }
        else if ( arg == "--share-textures" || arg == "-t" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            const std::string arg1 = argv[++i];
            if (arg1 == "false")
                g_share_textures = false;
            else if (arg1 == "true")
                g_share_textures = true;
            else
                printUsageAndExit( argv[0] );
        }
        else
        {
            std::cerr << "ERROR: Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }
}


int main( int argc, char* argv[] )
{
    parseCommandLine( argc, argv );

    try
    {
#if OPTIX_USE_NVML
        g_nvmlLoaded = loadNvmlFunctions();
        if ( g_nvmlLoaded )
            initializeNvml();
#endif

        //
        // Set up per-device render states
        //
        std::vector<PerDeviceSampleState> pd_states;
        createContexts( pd_states );

        //
        // Determine P2P topology, and load textures accordingly
        //
        if (g_peer_usage != PEERS_NONE)
        {
            enablePeerAccess( pd_states );
            bool require_nvlink = (g_peer_usage == PEERS_NVLINK);
            findPeers( pd_states, require_nvlink );
        }
        std::vector<int> p2p_islands;
        computeP2PIslands( pd_states, p2p_islands );
        loadTextures( pd_states, p2p_islands );

        //
        // Set up OptiX state
        //
        for( PerDeviceSampleState& pd_state : pd_states )
        {
            CUDA_CHECK( cudaSetDevice( pd_state.device_idx ) );
            uploadAdditionalShadingData( pd_state );
            buildMeshAccel     ( pd_state );
            createModule       ( pd_state );
            createProgramGroups( pd_state );
            createPipeline     ( pd_state );
            createSBT          ( pd_state );
            allocIOBuffers     ( pd_state, static_cast<int>( pd_states.size() ) );
        }

        for( PerDeviceSampleState& pd_state : pd_states )
        {
            initLaunchParams( pd_state );
        }

        initCameraState();
        GLFWwindow* window = nullptr;

        //
        // If the output file is empty, go into interactive mode
        //
        if( g_outfile == "" )
        {
            // Set up GUI and callbacks
            window = sutil::initUI( "optixNVLink", width, height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );

            int gl_interop_device = getGlInteropDeviceId( static_cast<int>( pd_states.size() ) );

            // Decide on the frame buffer type. Use ZERO_COPY memory as a default,
            // which copies the frame buffer data through pinned host memory.
            sutil::CUDAOutputBufferType buff_type = sutil::CUDAOutputBufferType::ZERO_COPY;

            // When using a single GPU that is also the gl interop device, render directly
            // into a gl interop buffer, avoiding copies.
            if ( g_optimize_framebuffer && pd_states.size() == 1 && gl_interop_device == 0 )
            {
                buff_type = sutil::CUDAOutputBufferType::GL_INTEROP;
            }

            // If using multiple GPUs are fully connected (and one of them is the
            // gl interop device) use a device-side buffer to avoid copying to host and back.
            // Note that it can't render directly into a gl interop buffer in the multi-GPU case.
            else if ( g_optimize_framebuffer && p2p_islands.size() == 1 && ((1<<gl_interop_device) & p2p_islands[0]) )
            {
                buff_type = sutil::CUDAOutputBufferType::CUDA_P2P;
            }

            // Make the frame buffer
            sutil::CUDAOutputBuffer<uchar4> output_buffer( buff_type, width, height );
            int output_device = (gl_interop_device >= 0) ? gl_interop_device : 0;
            output_buffer.setDevice( output_device );
            sutil::GLDisplay gl_display;

            // Timing variables
            std::chrono::duration<double> state_update_time( 0.0 );
            std::chrono::duration<double> render_time( 0.0 );
            std::chrono::duration<double> display_time( 0.0 );

            // Render loop
            do
            {
                auto t0 = std::chrono::steady_clock::now();
                glfwPollEvents();

                updateDeviceStates( output_buffer, pd_states );
                auto t1 = std::chrono::steady_clock::now();
                state_update_time += t1 - t0;
                t0 = t1;

                launchSubframe( output_buffer, pd_states );
                t1 = std::chrono::steady_clock::now();
                render_time += t1 - t0;
                t0 = t1;

                displaySubframe( output_buffer, gl_display, window );
                t1 = std::chrono::steady_clock::now();
                display_time += t1-t0;

                sutil::displayStats( state_update_time, render_time, display_time );

                glfwSwapBuffers(window);

                for( PerDeviceSampleState& pd_state : pd_states )
                    ++pd_state.params.subframe_index;

            } while( !glfwWindowShouldClose( window ) );

            // Make sure all of the CUDA streams finish
            for( PerDeviceSampleState& pd_state : pd_states )
            {
                CUDA_CHECK( cudaSetDevice( pd_state.device_idx ) );
                CUDA_SYNC_CHECK();
            }
        }

        //
        // If an output file was named, render and save
        //
        else
        {
            sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::ZERO_COPY, width, height );
            output_buffer.setDevice( 0 );

            updateDeviceStates( output_buffer, pd_states );
            launchSubframe( output_buffer, pd_states );

            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = output_buffer.width();
            buffer.height       = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            sutil::saveImage( g_outfile.c_str(), buffer, false );
        }

        //
        // Clean up resources
        //
        if ( window )
            sutil::cleanupUI( window );
        destroyTextures( pd_states );
        shutdownPeerAccess( pd_states );
        for( PerDeviceSampleState& pd_state : pd_states )
            cleanupState( pd_state );

    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
