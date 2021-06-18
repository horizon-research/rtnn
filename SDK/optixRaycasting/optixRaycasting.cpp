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

#include <sampleConfig.h>

#include "cuda/whitted.h"
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Matrix.h>
#include <sutil/Record.h>
#include <sutil/Scene.h>
#include <sutil/sutil.h>

#include "optixRaycasting.h"
#include "optixRaycastingKernels.h"

#include <iomanip>


struct RaycastingState
{
    int                         width                    = 0;
    int                         height                   = 0;

    OptixDeviceContext          context                  = 0;
    sutil::Scene                scene                    = {};

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixModule                 ptx_module               = 0;
    OptixPipeline               pipeline_1               = 0;
    OptixPipeline               pipeline_2               = 0;

    OptixProgramGroup           raygen_prog_group        = 0;
    OptixProgramGroup           miss_prog_group          = 0;
    OptixProgramGroup           hit_prog_group           = 0;

    Params                      params                   = {};
    Params                      params_translated        = {};
    OptixShaderBindingTable     sbt                      = {};

    sutil::Texture           mask                     = {};
};


typedef sutil::Record<whitted::HitGroupData> HitGroupRecord;


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n"
              << "Options:\n"
              << "  -h | --help                           Print this usage message\n"
              << "  -f | --file  <prefix>                 Prefix of output file\n"
              << "       --mask  <mask.ppm>               Mask texture\n"
              << "  -m | --model <model.gltf>             Model to be rendered\n"
              << "  -w | --width <number>                 Output image width\n"
              << std::endl;

    exit( 1 );
}


void createModule( RaycastingState& state )
{
    char   log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    state.pipeline_compile_options.numPayloadValues      = 4;
    state.pipeline_compile_options.numAttributeValues    = 2;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixRaycasting.cu" );
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX( state.context, &module_compile_options, &state.pipeline_compile_options,
                                               ptx.c_str(), ptx.size(), log, &sizeof_log, &state.ptx_module ) );
}

void createProgramGroups( RaycastingState& state, bool enableAnyHit )
{
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc    = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__from_buffer";

    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.raygen_prog_group ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__buffer_miss";
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.miss_prog_group ) );


    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if( enableAnyHit )
    {
        hit_prog_group_desc.hitgroup.moduleAH            = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__texture_mask";
    }
    hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__buffer_hit";
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &hit_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.hit_prog_group ) );
}


void createPipelines( RaycastingState& state )
{
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    const uint32_t    max_trace_depth   = 1;
    OptixProgramGroup program_groups[3] = {state.raygen_prog_group, state.miss_prog_group, state.hit_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace_depth;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_CHECK_LOG( optixPipelineCreate( state.context, &state.pipeline_compile_options, &pipeline_link_options,
                                          program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ), log,
                                          &sizeof_log, &state.pipeline_1 ) );
    OPTIX_CHECK_LOG( optixPipelineCreate( state.context, &state.pipeline_compile_options, &pipeline_link_options,
                                          program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ), log,
                                          &sizeof_log, &state.pipeline_2 ) );

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
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline_1, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            2  // maxTraversableDepth
                                            ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline_2, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            2  // maxTraversableDepth
                                            ) );
}


void createSBT( RaycastingState& state, const std::string& maskfile )
{
    // raygen
    CUdeviceptr  d_raygen_record    = 0;
    const size_t raygen_record_size = sizeof( sutil::EmptyRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen_record ), raygen_record_size ) );

    sutil::EmptyRecord rg_record;
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_record ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_raygen_record ), &rg_record, raygen_record_size, cudaMemcpyHostToDevice ) );

    // miss
    CUdeviceptr  d_miss_record    = 0;
    const size_t miss_record_size = sizeof( sutil::EmptyRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss_record ), miss_record_size ) );

    sutil::EmptyRecord ms_record;
    OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &ms_record ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_miss_record ), &ms_record, miss_record_size, cudaMemcpyHostToDevice ) );

    // load the mask (if there is one)
    if( !maskfile.empty() )
    {
        std::cerr << "Loading mask: " << maskfile << std::endl;
        state.mask = sutil::loadTexture( maskfile.c_str(), make_float3( 1.0f ) );
    }

    // hit group
    std::vector<HitGroupRecord> hitgroup_records;
    for( const auto mesh : state.scene.meshes() )
    {
        for( size_t i = 0; i < mesh->material_idx.size(); ++i )
        {
            HitGroupRecord rec = {};
            OPTIX_CHECK( optixSbtRecordPackHeader( state.hit_prog_group, &rec ) );
            rec.data.geometry_data.type                    = GeometryData::TRIANGLE_MESH;
            rec.data.geometry_data.triangle_mesh.positions = mesh->positions[i];
            rec.data.geometry_data.triangle_mesh.normals   = mesh->normals[i];
            rec.data.geometry_data.triangle_mesh.texcoords = mesh->texcoords[i];
            rec.data.geometry_data.triangle_mesh.indices   = mesh->indices[i];

            rec.data.material_data.pbr                = MaterialData::Pbr();
            rec.data.material_data.pbr.base_color_tex = state.mask.texture;

            hitgroup_records.push_back( rec );
        }
    }

    CUdeviceptr  d_hitgroup_record    = 0;
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup_record ), hitgroup_record_size * hitgroup_records.size() ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_hitgroup_record ), hitgroup_records.data(),
                            hitgroup_record_size * hitgroup_records.size(), cudaMemcpyHostToDevice ) );

    state.sbt.raygenRecord                = d_raygen_record;
    state.sbt.missRecordBase              = d_miss_record;
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    state.sbt.missRecordCount             = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase          = d_hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    state.sbt.hitgroupRecordCount         = static_cast<int>( hitgroup_records.size() );
}


void bufferRays( RaycastingState& state )
{
    // Create CUDA buffers for rays and hits
    sutil::Aabb aabb = state.scene.aabb();
    aabb.invalidate();
    for( const auto mesh : state.scene.meshes() )
        aabb.include( mesh->world_aabb );
    const float3 bbox_span = aabb.extent();
    state.height           = static_cast<int>( state.width * bbox_span.y / bbox_span.x );

    Ray*   rays_d             = 0;
    Ray*   translated_rays_d  = 0;
    size_t rays_size_in_bytes = sizeof( Ray ) * state.width * state.height;
    CUDA_CHECK( cudaMalloc( &rays_d, rays_size_in_bytes ) );
    CUDA_CHECK( cudaMalloc( &translated_rays_d, rays_size_in_bytes ) );

    createRaysOrthoOnDevice( rays_d, state.width, state.height, aabb.m_min, aabb.m_max, 0.05f );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaMemcpy( translated_rays_d, rays_d, rays_size_in_bytes, cudaMemcpyDeviceToDevice ) );

    translateRaysOnDevice( translated_rays_d, state.width * state.height, bbox_span * make_float3( 0.2f, 0, 0 ) );
    CUDA_CHECK( cudaGetLastError() );

    Hit*   hits_d             = 0;
    Hit*   translated_hits_d  = 0;
    size_t hits_size_in_bytes = sizeof( Hit ) * state.width * state.height;
    CUDA_CHECK( cudaMalloc( &hits_d, hits_size_in_bytes ) );
    CUDA_CHECK( cudaMalloc( &translated_hits_d, hits_size_in_bytes ) );

    state.params            = {state.scene.traversableHandle(), rays_d, hits_d};
    state.params_translated = {state.scene.traversableHandle(), translated_rays_d, translated_hits_d};
}


void launch( RaycastingState& state )
{
    CUstream stream_1 = 0;
    CUstream stream_2 = 0;
    CUDA_CHECK( cudaStreamCreate( &stream_1 ) );
    CUDA_CHECK( cudaStreamCreate( &stream_2 ) );

    Params* d_params            = 0;
    Params* d_params_translated = 0;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( Params ) ) );
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params ), &state.params, sizeof( Params ),
                                 cudaMemcpyHostToDevice, stream_1 ) );

    OPTIX_CHECK( optixLaunch( state.pipeline_1, stream_1, reinterpret_cast<CUdeviceptr>( d_params ), sizeof( Params ),
                              &state.sbt, state.width, state.height, 1 ) );

    // Translated
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params_translated ), sizeof( Params ) ) );
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params_translated ), &state.params_translated,
                                 sizeof( Params ), cudaMemcpyHostToDevice, stream_2 ) );

    OPTIX_CHECK( optixLaunch( state.pipeline_2, stream_2, reinterpret_cast<CUdeviceptr>( d_params_translated ),
                              sizeof( Params ), &state.sbt, state.width, state.height, 1 ) );

    CUDA_SYNC_CHECK();

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params_translated ) ) );
}


void shadeHits( RaycastingState& state, const std::string& outfile )
{
    sutil::CUDAOutputBufferType     output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
    sutil::CUDAOutputBuffer<float3> output_buffer( output_buffer_type, state.width, state.height );

    sutil::ImageBuffer buffer;
    buffer.width        = state.width;
    buffer.height       = state.height;
    buffer.pixel_format = sutil::BufferImageFormat::FLOAT3;

    // Original
    shadeHitsOnDevice( output_buffer.map(), state.width * state.height, state.params.hits );
    CUDA_CHECK( cudaGetLastError() );
    output_buffer.unmap();

    std::string ppmfile = outfile + ".ppm";
    buffer.data         = output_buffer.getHostPointer();
    sutil::saveImage( ppmfile.c_str(), buffer, false );
    std::cerr << "Wrote image to " << ppmfile << std::endl;

    // Translated
    shadeHitsOnDevice( output_buffer.map(), state.width * state.height, state.params_translated.hits );
    CUDA_CHECK( cudaGetLastError() );
    output_buffer.unmap();

    ppmfile     = outfile + "_translated.ppm";
    buffer.data = output_buffer.getHostPointer();
    sutil::saveImage( ppmfile.c_str(), buffer, false );
    std::cerr << "Wrote translated image to " << ppmfile << std::endl;
}


void cleanup( RaycastingState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline_1 ) );
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline_2 ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.hit_prog_group ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.rays ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.hits ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params_translated.rays ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params_translated.hits ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );

    CUDA_CHECK( cudaDestroyTextureObject( state.mask.texture ) );
    CUDA_CHECK( cudaFreeArray( state.mask.array ) );
}


int main( int argc, char** argv )
{
    std::string     infile, outfile, maskfile;
    RaycastingState state;
    state.width = 640;

    // parse arguments
    for( int i = 1; i < argc; ++i )
    {
        std::string arg( argv[i] );
        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( ( arg == "-f" || arg == "--file" ) && i + 1 < argc )
        {
            outfile = argv[++i];
        }
        else if( ( arg == "--mask" ) && i + 1 < argc )
        {
            maskfile = argv[++i];
        }
        else if( ( arg == "-m" || arg == "--model" ) && i + 1 < argc )
        {
            infile = argv[++i];
        }
        else if( ( arg == "-w" || arg == "--width" ) && i + 1 < argc )
        {
            state.width = atoi( argv[++i] );
        }
        else
        {
            std::cerr << "Bad option: '" << arg << "'" << std::endl;
            printUsageAndExit( argv[0] );
        }
    }

    // Set default scene if user did not specify scene
    if( infile.empty() )
    {
        std::cerr << "No model specified, using default model (Duck.gltf)" << std::endl;
        infile = sutil::sampleDataFilePath( "Duck/Duck.gltf" );

        if( maskfile.empty() )
        {
            maskfile = sutil::sampleDataFilePath( "Duck/DuckMask.ppm" );
        }
    }

    // Set default output file prefix
    if( outfile.empty() )
    {
        std::cerr << "No file prefix specified, using default file prefix (output)" << std::endl;
        outfile = "output";
    }

    try
    {
        sutil::loadScene( infile.c_str(), state.scene );
        state.scene.createContext();

        uint32_t triangle_input_flags = maskfile.empty() ? OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT : OPTIX_GEOMETRY_FLAG_NONE;
        state.scene.buildMeshAccels( triangle_input_flags );
        state.scene.buildInstanceAccel( RAY_TYPE_COUNT );
        state.context = state.scene.context();

        OPTIX_CHECK( optixInit() );  // Need to initialize function table
        createModule( state );
        createProgramGroups( state, triangle_input_flags != OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT );
        createPipelines( state );
        createSBT( state, maskfile );

        bufferRays( state );
        launch( state );
        shadeHits( state, outfile );
        cleanup( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

