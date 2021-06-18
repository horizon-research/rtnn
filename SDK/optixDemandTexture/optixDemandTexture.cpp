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

#include <optixDemandTexture.h>
#include <CheckerBoardImage.h>
#include <DemandTextureManager.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace demandLoading;

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

void configureCamera( const uint32_t width, const uint32_t height, float3& cam_eye, float3& camera_u, float3& camera_v, float3& camera_w )
{
    sutil::Camera camera;
    cam_eye = {0.0f, -3.0f, 0.0f};
    camera.setEye( cam_eye );
    camera.setLookat( make_float3( 0.0f, 0.0f, 0.0f ) );
    camera.setUp( make_float3( 0.0f, 0.0f, 1.0f ) );
    camera.setFovY( 60.0f );
    camera.setAspectRatio( (float)width / (float)height );
    camera.UVWFrame( camera_u, camera_v, camera_w );
}


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file                  | -f <filename>  Specify file for image output\n";
    std::cerr << "         --help                  | -h             Print this usage message\n";
    std::cerr << "         --cols <1-32>                            The number of columns in the sphere grid (default 1)\n";
    std::cerr << "         --rows <1-32>                            The number of rows in the sphere grid (default 1)\n";
    std::cerr << "         --num-textures <1-1024>                  The number of texture samplers to create (default 1)\n";
    std::cerr << "         --mip-levels=<max>x<min>                 Set the mip level range for the textures (default 0, finest mip level)\n";
    std::cerr << "                                                  The range is [0-9] (inclusive). Max. must be less than or equal to min.\n";
    std::cerr << "         --dim=<width>x<height>                   Set image dimensions\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}


int main( int argc, char* argv[] )
{
    std::string outfile;
    int         width       = 1024;
    int         height      = 1024;
    int         numCols     = 1;
    int         numRows     = 1;
    int         numTextures = 1;
    int         maxMip      = 0;
    int         minMip      = 0;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
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
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else if( arg == "--cols" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            numCols = std::max( 1, std::min( atoi( argv[++i] ), 32 ) );
        }
        else if( arg == "--rows" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            numRows = std::max( 1, std::min( atoi( argv[++i] ), 32 ) );
        }
        else if( arg == "--num-textures" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            numTextures = std::max( 1, std::min( atoi( argv[++i] ), 1024 ) );
        }
        else if( arg.substr( 0, 13 ) == "--mip-levels=" )
        {
            const std::string range_arg = arg.substr( 13 );
            sutil::parseDimensions( range_arg.c_str(), maxMip, minMip );
            if( maxMip < 0 || minMip < maxMip )
                printUsageAndExit( argv[0] );

            // Force maxMip to the range [0-9] and minMip to the range [maxMip-9].
            maxMip = std::max( 0,      std::min( maxMip, 9 ) );
            minMip = std::max( maxMip, std::min( minMip, 9 ) );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        char log[2048];  // For error reporting from OptiX creation functions


        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );

            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixInit() );
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }

        //
        // Generate an MxN grid of spheres
        //
        std::vector<Sphere> spheres;
        {
            const float xSpacing = 3.0f / numCols;
            const float ySpacing = 3.0f / numRows;
            const float xOffset  = -( xSpacing * numCols / 2 ) + xSpacing / 2;
            const float yOffset  = -( ySpacing * numRows / 2 ) + ySpacing / 2;
            const float radius   = std::min<float>( xSpacing, ySpacing ) / 2.0f;

            for( int yIdx = 0; yIdx < numRows; ++yIdx )
            {
                for( int xIdx = 0; xIdx < numCols; ++xIdx )
                {
                    float3 center = make_float3( xOffset + xIdx * xSpacing, 0.0f, yOffset + yIdx * ySpacing );
                    Sphere sphere = { center, radius };
                    spheres.emplace_back( sphere );
                }
            }
        }

        //
        // accel handling
        //
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

            // AABB build input
            std::vector<OptixAabb> aabbs;
            for( size_t idx = 0; idx < spheres.size(); ++idx )
            {
                aabbs.push_back( spheres[idx].bounds() );
            }

            CUdeviceptr d_aabb_buffer;
            const size_t aabbs_size = sizeof( OptixAabb ) * aabbs.size();
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), aabbs_size ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabbs[0], aabbs_size, cudaMemcpyHostToDevice ) );

            std::vector<uint32_t> sbt_index;
            std::vector<uint32_t> aabb_input_flags;
            for( size_t idx = 0; idx < spheres.size(); ++idx )
            {
                sbt_index.push_back( static_cast<uint32_t>( idx ) );
                aabb_input_flags.push_back( OPTIX_GEOMETRY_FLAG_NONE );
            }

            CUdeviceptr d_sbt_index;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), sizeof( uint32_t ) * sbt_index.size() ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_sbt_index ),
                                    sbt_index.data(),
                                    sizeof( uint32_t ) * sbt_index.size(),
                                    cudaMemcpyHostToDevice ) );

            OptixBuildInput aabb_input = {};
            aabb_input.type                                             = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            aabb_input.customPrimitiveArray.flags                       = aabb_input_flags.data();
            aabb_input.customPrimitiveArray.aabbBuffers                 = &d_aabb_buffer;
            aabb_input.customPrimitiveArray.numPrimitives               = static_cast<uint32_t>( spheres.size() );
            aabb_input.customPrimitiveArray.numSbtRecords               = static_cast<uint32_t>( spheres.size() );
            aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer        = d_sbt_index;
            aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
            aabb_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );
            aabb_input.customPrimitiveArray.primitiveIndexOffset        = 0;

            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &aabb_input, 1, &gas_buffer_sizes ) );
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

            // non-compacted output
            CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
            size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                                    compactedSizeOffset + 8 ) );

            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );


            OPTIX_CHECK( optixAccelBuild( context,
                                          0,              // CUDA stream
                                          &accel_options, &aabb_input,
                                          1,              // num build inputs
                                          d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                                          d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                          &emitProperty,  // emitted property list
                                          1               // num emitted properties
                                          ) );

            CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
            CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );
            CUDA_CHECK( cudaFree( (void*)d_sbt_index ) );

            size_t compacted_gas_size;
            CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

            if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
            {
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

                // use handle as input and output
                OPTIX_CHECK( optixAccelCompact( context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

                CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
            }
            else
            {
                d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
            }
        }

        //
        // Create module
        //
        OptixModule                 module                   = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

            pipeline_compile_options.usesMotionBlur        = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues      = 3;
            pipeline_compile_options.numAttributeValues    = 3;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            const std::string ptx        = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixDemandTexture.cu" );
            size_t            sizeof_log = sizeof( log );

            OPTIX_CHECK_LOG( optixModuleCreateFromPTX( context, &module_compile_options, &pipeline_compile_options,
                                                       ptx.c_str(), ptx.size(), log, &sizeof_log, &module ) );
        }

        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {};  //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            size_t sizeof_log                               = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate( context, &raygen_prog_group_desc,
                                                      1,  // num program groups
                                                      &program_group_options, log, &sizeof_log, &raygen_prog_group ) );

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            sizeof_log                                  = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate( context, &miss_prog_group_desc,
                                                      1,  // num program groups
                                                      &program_group_options, log, &sizeof_log, &miss_prog_group ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc        = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.moduleIS            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
            sizeof_log                                            = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate( context, &hitgroup_prog_group_desc,
                                                      1,  // num program groups
                                                      &program_group_options, log, &sizeof_log, &hitgroup_prog_group ) );
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 1;
            OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth            = max_trace_depth;
            pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            size_t sizeof_log                              = sizeof( log );
            OPTIX_CHECK_LOG( optixPipelineCreate( context, &pipeline_compile_options, &pipeline_link_options,
                                                  program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ),
                                                  log, &sizeof_log, &pipeline ) );

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
            OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state, continuation_stack_size,
                                                    1  // maxTraversableDepth
                                                    ) );
        }

        //
        // Initialize DemandTextureManager and create an array of demand-loaded textures.
        // The texture id is passed to the closest hit shader via a hit group record in the SBT.
        // The texture sampler array (indexed by texture id) is passed as a launch parameter.
        //
        DemandTextureManager textureManager;

        // We use a procedurally generated image for the textures.
        std::shared_ptr<CheckerBoardImage> textureReader( std::make_shared<CheckerBoardImage>( 1024, 1024 ) );
        const float                        textureScale = 1.f;

        std::vector<DemandTexture> textures;
        for( int idx = 0; idx < numTextures; ++idx )
        {
            textures.push_back( textureManager.createTexture( textureReader ) );
        }

        //
        // Set up shader binding table.  The demand-loaded texture is passed to the closest hit
        // program via the hitgroup record.
        //
        OptixShaderBindingTable sbt = {};
        CUdeviceptr d_hitgroup_records = 0;
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt = {};
            configureCamera( width, height, rg_sbt.data.cam_eye, rg_sbt.data.camera_u, rg_sbt.data.camera_v, rg_sbt.data.camera_w );
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice ) );

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            ms_sbt.data = {0.3f, 0.1f, 0.2f};
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( miss_record ), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice ) );

            // The demand-loaded texture id is passed to the closest hit program via the hitgroup record.
            std::vector<HitGroupSbtRecord> hitgroup_records;
            hitgroup_records.resize( spheres.size() );

            int       lod      = 0;
            const int lodRange = 1 + minMip - maxMip;
            size_t    texIdx   = 0;
            for( size_t idx = 0; idx < spheres.size(); ++idx )
            {
                OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hitgroup_records[idx] ) );
                hitgroup_records[idx].data = { spheres[idx],
                                               textures[ ++texIdx % numTextures ].getId(),
                                               textureScale,
                                               static_cast<float>( (maxMip + (lod++ % lodRange)) % 10 ) };
            }

            size_t      hitgroup_records_size = sizeof( HitGroupSbtRecord ) * hitgroup_records.size();
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup_records ), hitgroup_records_size ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_hitgroup_records ),
                                                             hitgroup_records.data(),
                                                             hitgroup_records_size,
                                                             cudaMemcpyHostToDevice ) );

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = d_hitgroup_records;
            sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
            sbt.hitgroupRecordCount         = static_cast<uint32_t>( spheres.size() );
        }

        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        //
        // launch
        //
        {
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );

            Params params;
            params.image         = output_buffer.map();
            params.image_width   = width;
            params.image_height  = height;
            params.origin_x      = width / 2;
            params.origin_y      = height / 2;
            params.handle        = gas_handle;
            params.pagingContext = textureManager.getPagingContext();

            // Sync demand-texture sampler array to the device and provide it as a launch parameter.
            textureManager.launchPrepare();
            params.demandTextures = textureManager.getSamplers();

            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_param ), &params, sizeof( params ), cudaMemcpyHostToDevice ) );

            // The initial launch might accumulate texture requests.
            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
            CUDA_SYNC_CHECK();

            // Repeatedly process any texture requests and relaunch until done.
            for( int numFilled = textureManager.processRequests(); numFilled > 0; numFilled = textureManager.processRequests() )
            {
                std::cout << "Filled " << numFilled << " requests.  Relaunching..." << std::endl;

                // Sync sampler array and update launch parameter if it grew.
                textureManager.launchPrepare();
                if( params.demandTextures != textureManager.getSamplers() )
                {
                    params.demandTextures = textureManager.getSamplers();
                    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_param ), &params, sizeof( params ), cudaMemcpyHostToDevice ) );
                }

                // Relaunch
                OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
                CUDA_SYNC_CHECK();
            }

            output_buffer.unmap();
        }

        //
        // Display results
        //
        {
            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = width;
            buffer.height       = height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            if( outfile.empty() )
                sutil::displayBufferWindow( argv[0], buffer );
            else
                sutil::saveImage( outfile.c_str(), buffer, false );
        }

        //
        // Cleanup
        //
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer ) ) );

            OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
            OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
