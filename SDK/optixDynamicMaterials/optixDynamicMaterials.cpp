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

#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Camera.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>

#include "optixDynamicMaterials.h"

#include <array>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct SampleState
{
    SampleState( uint32_t width, uint32_t height )
    {
        params.image_width  = width;
        params.image_height = height;
    }

    Params      params;
    CUdeviceptr d_param;

    OptixDeviceContext             context = nullptr;
    OptixTraversableHandle         gas_handle;
    OptixTraversableHandle         ias_handle;
    CUdeviceptr                    d_gas_output_buffer      = 0;
    CUdeviceptr                    d_ias_output_buffer      = 0;
    OptixModule                    module                   = nullptr;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixProgramGroup              raygen_prog_group        = nullptr;
    OptixProgramGroup              miss_prog_group          = nullptr;
    std::vector<OptixProgramGroup> hitgroup_prog_groups;
    OptixPipeline                  pipeline                 = nullptr;
    OptixShaderBindingTable        sbt                      = {};
    CUstream                       stream                   = 0;
    size_t                         hitgroupSbtRecordStride  = 0;
};


struct Matrix
{
    float m[12];
};


//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

// Transforms for instances - one on the left (sphere 0), one in the centre and one on the right (sphere 2).
std::vector<Matrix> transforms = {
    { 1, 0, 0, -6, 0, 1, 0, 0, 0, 0, 1, -10 },
    { 1, 0, 0,  0, 0, 1, 0, 0, 0, 0, 1, -10 },
    { 1, 0, 0,  6, 0, 1, 0, 0, 0, 0, 1, -10 },
};


// Offsets into SBT for each instance. Hence this needs to be in sync with transforms!
// The middle sphere has two SBT records, the two other instances have one each.
unsigned int sbtOffsets[] = { 0, 1, 3 };


const static std::array<float3, 3> g_colors =
{ { { 1.f, 0.f, 0.f }, { 0.f, 1.f, 0.f }, { 0.f, 0.f, 1.f } } };


// A cycling index (offset), used here as an offset onto hitgroup records.
template <unsigned int MAXINDEX>
struct MaterialIndex
{
    MaterialIndex()
        : mIndex( 0 )
    {
    }
    unsigned int getVal() const { return mIndex; }
    void         nextVal()
    {
        if( ++mIndex == MAXINDEX )
            mIndex = 0;
    }

  private:
    unsigned int mIndex;
};

// Left sphere
MaterialIndex<3> g_materialIndex_0;
bool             g_hasDataChanged = false;

// Middle sphere
MaterialIndex<2> g_materialIndex_1;
bool             g_hasOffsetChanged = false;

// Right sphere
MaterialIndex<3> g_materialIndex_2;
bool             g_hasSbtChanged = false;


//------------------------------------------------------------------------------
//
// Helper Functions
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}


void initCamera( SampleState& state )
{
    sutil::Camera camera;
    camera.setEye( make_float3( 0.0f, 0.0f, 3.0f ) );
    camera.setLookat( make_float3( 0.0f, 0.0f, 0.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 60.0f );
    camera.setAspectRatio( static_cast<float>( state.params.image_width ) / static_cast<float>( state.params.image_height ) );
    camera.UVWFrame( state.params.camera_u, state.params.camera_v, state.params.camera_w );
    state.params.cam_eye = camera.eye();
}


void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}


void createContext( SampleState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &state.context ) );
}


void buildGAS( SampleState& state )
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    // AABB build input
    OptixAabb   aabb = { -1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f };
    CUdeviceptr d_aabb_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabb, sizeof( OptixAabb ), cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input = {};

    aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
    aabb_input.customPrimitiveArray.numPrimitives = 1;

    uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &aabb_input, 1, &gas_buffer_sizes ) );
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = reinterpret_cast<CUdeviceptr>(
        reinterpret_cast<char*>( d_buffer_temp_output_gas_and_compacted_size ) + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild(
                state.context,
                0,                    // CUDA stream
                &accel_options,
                &aabb_input,
                1,                    // num build inputs
                d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_compacted_size,
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                &emitProperty,        // emitted property list
                1                     // num emitted properties
                ) );
    state.params.radius = 1.5f;

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_aabb_buffer ) ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, reinterpret_cast<void*>( emitProperty.result ),
                            sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact(
                    state.context,
                    0,  // CUDA stream
                    state.gas_handle,
                    state.d_gas_output_buffer,
                    compacted_gas_size,
                    &state.gas_handle
                    ) );

        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_buffer_temp_output_gas_and_compacted_size ) ) );
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}


void buildIAS( SampleState& state )
{
    std::vector<OptixInstance> instances;
    for( size_t i = 0; i < transforms.size(); ++i )
    {
        OptixInstance inst;
        memcpy( inst.transform, &transforms[i], sizeof( float ) * 12 );
        inst.instanceId        = 0;
        inst.visibilityMask    = 1;
        inst.sbtOffset         = sbtOffsets[i];
        inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
        inst.traversableHandle = state.gas_handle;
        instances.push_back( inst );
    }

    CUdeviceptr d_inst;
    size_t      instancesSizeInBytes = instances.size() * sizeof( OptixInstance );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_inst ), instancesSizeInBytes ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_inst ), &instances[0], instancesSizeInBytes, cudaMemcpyHostToDevice ) );

    OptixBuildInput instanceInput            = {};
    instanceInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances    = d_inst;
    instanceInput.instanceArray.numInstances = static_cast<unsigned int>( instances.size() );

    OptixAccelBuildOptions iasAccelOptions = {};
    iasAccelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    iasAccelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &iasAccelOptions, &instanceInput, 1, &ias_buffer_sizes ) );
    CUdeviceptr d_temp_buffer_ias;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_ias ), ias_buffer_sizes.tempSizeInBytes ) );

    // We need to free the output buffer if we are rebuilding the IAS.
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_ias_output_buffer ) ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_ias_output_buffer ), ias_buffer_sizes.outputSizeInBytes ) );

    OPTIX_CHECK( optixAccelBuild(
                 state.context,
                 0,  // CUDA stream
                 &iasAccelOptions,
                 &instanceInput,
                 1,  // num build inputs
                 d_temp_buffer_ias,
                 ias_buffer_sizes.tempSizeInBytes,
                 state.d_ias_output_buffer,
                 ias_buffer_sizes.outputSizeInBytes,
                 &state.ias_handle,
                 nullptr,
                 0
                 ) );
    CUDA_SYNC_CHECK();

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_ias ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_inst ) ) );
}


void createModule( SampleState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    state.pipeline_compile_options.numPayloadValues      = 3;  // memory size of payload (in trace())
    state.pipeline_compile_options.numAttributeValues    = 3;  // memory size of attributes (from is())
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixDynamicMaterials.cu" );

    char   log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );

    OPTIX_CHECK_LOG( optixModuleCreateFromPTX( state.context, &module_compile_options, &state.pipeline_compile_options,
                                               ptx.c_str(), ptx.size(), log, &sizeof_log, &state.module ) );
}


void createProgramGroups( SampleState& state )
{
    OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc    = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char   log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.raygen_prog_group ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log                                  = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.miss_prog_group ) );

    // hard-coded list of different CH programs for different OptixInstances
    std::vector<const char*> chNames = {// The left sphere has a single CH program
                                        "__closesthit__ch",
                                        // The middle sphere toggles between two CH programs
                                        "__closesthit__ch", "__closesthit__normal",
                                        // The right sphere uses the g_materialIndex_2.getVal()'th of these CH programs
                                        "__closesthit__blue", "__closesthit__green", "__closesthit__red"};

    std::vector<OptixProgramGroupDesc> hitgroup_prog_group_descs;
    for( auto chName : chNames )
    {
        OptixProgramGroupDesc hitgroup_prog_group_desc        = {};
        hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH            = state.module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = chName;
        hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        hitgroup_prog_group_desc.hitgroup.moduleIS            = state.module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
        hitgroup_prog_group_descs.push_back( hitgroup_prog_group_desc );
    }
    sizeof_log = sizeof( log );
    state.hitgroup_prog_groups.resize( hitgroup_prog_group_descs.size() );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &hitgroup_prog_group_descs[0],
                                              static_cast<unsigned int>( hitgroup_prog_group_descs.size() ),
                                              &program_group_options, log, &sizeof_log, &state.hitgroup_prog_groups[0] ) );
}


void createPipeline( SampleState& state )
{
    const uint32_t max_trace_depth = 1;

    std::vector<OptixProgramGroup> program_groups;
    program_groups.push_back( state.raygen_prog_group );
    program_groups.push_back( state.miss_prog_group );
    for( auto g : state.hitgroup_prog_groups )
        program_groups.push_back( g );

    OptixPipelineLinkOptions pipeline_link_options;
    pipeline_link_options.maxTraceDepth          = max_trace_depth;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate( state.context, &state.pipeline_compile_options, &pipeline_link_options,
                                          &program_groups[0], static_cast<unsigned int>( program_groups.size() ), log,
                                          &sizeof_log, &state.pipeline ) );

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


void createSbt( SampleState& state )
{
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
    RayGenSbtRecord rg_sbt;
    rg_sbt.data = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice ) );
    state.sbt.raygenRecord = raygen_record;

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );

    MissSbtRecord ms_sbt;
    ms_sbt.data = { 0.3f, 0.1f, 0.2f }; // Background color
    OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &ms_sbt ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( miss_record ), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice ) );
    state.sbt.missRecordBase          = miss_record;
    state.sbt.missRecordStrideInBytes = sizeof( MissSbtRecord );
    state.sbt.missRecordCount         = 1;

    const size_t                   hitGroupSbtRecordCount = 4;
    std::vector<HitGroupSbtRecord> hg_sbt( hitGroupSbtRecordCount );
    size_t                         hg_sbt_size = hg_sbt.size();

    // The left sphere cycles through three colors by updating the data field of the SBT record.
    hg_sbt[0].data = { g_colors[0], 0u };
    OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_groups[0], &hg_sbt[0] ) );

    // The middle sphere toggles between two SBT records by adjusting the SBT
    // offset field of the sphere instance. The IAS needs to be rebuilt for the
    // update to take effect.
    hg_sbt[1].data = { g_colors[1], 1u };
    OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_groups[1], &hg_sbt[1] ) );

    hg_sbt[2].data = { g_colors[1], 1u };
    OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_groups[2], &hg_sbt[2] ) );

    // The right sphere cycles through colors by modifying the SBT. On update, a
    // different pre-built CH program is packed into the corresponding SBT
    // record.
    hg_sbt[3].data = { { 0.f, 0.f, 0.f }, 2u };
    OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_groups[g_materialIndex_2.getVal() + 3], &hg_sbt[3] ) );

    CUdeviceptr hitgroup_record;
    state.hitgroupSbtRecordStride = sizeof( HitGroupSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), state.hitgroupSbtRecordStride * hg_sbt_size ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( hitgroup_record ), &hg_sbt[0],
                            state.hitgroupSbtRecordStride * hg_sbt_size, cudaMemcpyHostToDevice ) );
    state.sbt.hitgroupRecordBase          = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>( state.hitgroupSbtRecordStride );
    state.sbt.hitgroupRecordCount         = static_cast<unsigned int>( hg_sbt_size );
}


void updateHitGroupData( SampleState& state )
{
    // Method 1:
    // Change the material parameters for the left sphere by directly modifying
    // the HitGroupData for the first SBT record.

    // Cycle through three base colors.
    g_materialIndex_0.nextVal();
    HitGroupData hg_data = HitGroupData { g_colors[g_materialIndex_0.getVal()], 0u };

    // Update the data field of the SBT record for the left sphere with the new base color.
    HitGroupSbtRecord* hg_sbt_0_ptr = &reinterpret_cast<HitGroupSbtRecord*>( state.sbt.hitgroupRecordBase )[0];
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( &hg_sbt_0_ptr->data ),
                            &hg_data, sizeof( HitGroupData ), cudaMemcpyHostToDevice ) );

    g_hasDataChanged = false;
}


void updateInstanceOffset( SampleState& state )
{
    // Method 2:
    // Update the SBT offset of the middle sphere. The offset is used to select
    // an SBT record during traversal, which dertermines the CH & AH programs
    // that will be invoked for shading.

    g_materialIndex_1.nextVal();
    sbtOffsets[1] = g_materialIndex_1.getVal() + 1;

    // It's necessary to rebuild the IAS for the updated offset to take effect.
    buildIAS( state );

    g_hasOffsetChanged = false;
}


void updateSbtHeader( SampleState& state )
{
    // Method 3:
    // Select a new material by re-packing the SBT header for the right sphere
    // with a different CH program.

    // The right sphere will use the next compiled program group.
    g_materialIndex_2.nextVal();

    HitGroupSbtRecord hg_sbt_2;
    hg_sbt_2.data = { { 0.f, 0.f, 0.f }, 2u };  // The color is hard-coded in the CH program.
    OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_groups[g_materialIndex_2.getVal() + 3], &hg_sbt_2 ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase + sizeof( HitGroupSbtRecord ) * sbtOffsets[2] ),
                            &hg_sbt_2, sizeof( HitGroupSbtRecord ), cudaMemcpyHostToDevice ) );

    g_hasSbtChanged = false;
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, SampleState& state )
{
    // Change the material properties using one of three different approaches.
    if( g_hasDataChanged )
        updateHitGroupData( state );
    if( g_hasOffsetChanged )
        updateInstanceOffset( state );
    if( g_hasSbtChanged )
        updateSbtHeader( state );
}


void initLaunch( SampleState& state )
{
    CUDA_CHECK( cudaStreamCreate( &state.stream ) );

    state.params.handle = state.ias_handle;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_param ), sizeof( Params ) ) );
}


void launch( SampleState& state, sutil::CUDAOutputBuffer<uchar4>& output_buffer )
{
    state.params.image = output_buffer.map();
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.d_param ), &state.params, sizeof( Params ), cudaMemcpyHostToDevice ) );

    assert( state.sbt.hitgroupRecordStrideInBytes % OPTIX_SBT_RECORD_ALIGNMENT == 0 );
    assert( state.sbt.hitgroupRecordBase % OPTIX_SBT_RECORD_ALIGNMENT == 0 );

    OPTIX_CHECK( optixLaunch( state.pipeline, state.stream, state.d_param, sizeof( Params ), &state.sbt,
                              state.params.image_width, state.params.image_height, /*depth=*/1 ) );
    CUDA_SYNC_CHECK();

    output_buffer.unmap();
}


void displayUsage()
{
    static char display_text[256];
    sutil::beginFrameImGui();
    {
        sprintf( display_text,
                 "Use the arrow keys to modify the materials:\n"
                 " [LEFT]  left sphere\n"
                 " [UP]    middle sphere\n"
                 " [RIGHT] right sphere\n" );
    }
    sutil::displayText( display_text, 20.0f, 20.0f );
    sutil::endFrameImGui();
}


void display( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display( static_cast<int>( output_buffer.width() ), static_cast<int>( output_buffer.height() ),
                        framebuf_res_x, framebuf_res_y, output_buffer.getPBO() );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
        else if( key == GLFW_KEY_LEFT )
        {
            g_hasDataChanged = true;
        }
        else if( key == GLFW_KEY_RIGHT )
        {
            g_hasSbtChanged = true;
        }
        else if( key == GLFW_KEY_UP )
        {
            g_hasOffsetChanged = true;
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


std::string getIndexedFilename( const std::string& name )
{
    static unsigned int counter = 0;
    size_t              pos     = name.find_last_of( '.' );
    if( pos == std::string::npos )
    {
        std::cerr << "Cannot find image format suffix" << std::endl;
        return name;
    }
    std::string       suffix = name.substr( pos );
    std::string       root   = name.substr( 0, pos );
    std::stringstream s;
    s << '_' << counter++ << suffix;
    return root + s.str();
}


void printBuffer( sutil::CUDAOutputBuffer<uchar4>& output_buffer, const std::string& outfile )
{
    sutil::ImageBuffer buffer;
    buffer.data         = output_buffer.getHostPointer();
    buffer.width        = static_cast<unsigned int>( output_buffer.width() );
    buffer.height       = static_cast<unsigned int>( output_buffer.height() );
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
    sutil::saveImage( outfile.c_str(), buffer, false );
}


int main( int argc, char* argv[] )
{
    SampleState                 state( 1024, 768 );
    std::string                 outfile;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
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
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int               width, height;
            sutil::parseDimensions( dims_arg.c_str(), width, height );
            state.params.image_width  = width;
            state.params.image_height = height;
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        initCamera( state );
        createContext( state );
        buildGAS( state );
        buildIAS( state );
        createModule( state );
        createProgramGroups( state );
        createPipeline( state );
        createSbt( state );

        initLaunch( state );
        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixDynamicMaterials", state.params.image_width, state.params.image_height );
            glfwSetKeyCallback( window, keyCallback );
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, state.params.image_width,
                                                               state.params.image_height );
                output_buffer.setStream( state.stream );
                sutil::GLDisplay gl_display;

                while( !glfwWindowShouldClose( window ) )
                {
                    glfwPollEvents();
                    updateState( output_buffer, state );
                    launch( state, output_buffer );
                    display( output_buffer, gl_display, window );
                    displayUsage();
                    glfwSwapBuffers( window );
                }
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

            sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, state.params.image_width, state.params.image_height );
            updateState( output_buffer, state );
            launch( state, output_buffer );

            // Original setup - R, G, B spheres from left to right.
            printBuffer( output_buffer, getIndexedFilename( outfile ) );

            // Now add "dynamism" - first cycle through three colors of sphere 0
            g_hasDataChanged = true;
            updateState( output_buffer, state );
            launch( state, output_buffer );
            printBuffer( output_buffer, getIndexedFilename( outfile ) );

            g_hasDataChanged = true;
            updateState( output_buffer, state );
            launch( state, output_buffer );
            printBuffer( output_buffer, getIndexedFilename( outfile ) );

            g_hasDataChanged = true;
            updateState( output_buffer, state );
            launch( state, output_buffer );
            printBuffer( output_buffer, getIndexedFilename( outfile ) );

            // Now cycle through three SBT entries for sphere 2
            g_hasSbtChanged = true;
            updateState( output_buffer, state );
            launch( state, output_buffer );
            printBuffer( output_buffer, getIndexedFilename( outfile ) );

            g_hasSbtChanged = true;
            updateState( output_buffer, state );
            launch( state, output_buffer );
            printBuffer( output_buffer, getIndexedFilename( outfile ) );

            // This should give us an image identical to the original one
            g_hasSbtChanged = true;
            updateState( output_buffer, state );
            launch( state, output_buffer );
            printBuffer( output_buffer, getIndexedFilename( outfile ) );

            // Toggle the material on the middle sphere
            g_hasOffsetChanged = true;
            updateState( output_buffer, state );
            launch( state, output_buffer );
            printBuffer( output_buffer, getIndexedFilename( outfile ) );

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

        //
        // Cleanup
        //
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_ias_output_buffer ) ) );

            OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
            for( auto grp : state.hitgroup_prog_groups )
                OPTIX_CHECK( optixProgramGroupDestroy( grp ) );
            OPTIX_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( state.module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
