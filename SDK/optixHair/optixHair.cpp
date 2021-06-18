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
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iterator>

#include <cuda/whitted.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <optix_function_table_definition.h>

#include "Hair.h"
#include "Head.h"
#include "ProgramGroups.h"
#include "Renderers.h"
#include "Util.h"
#include "optixHair.h"


void makeHairGAS( HairState* pState )
{
    Hair* const pHair = pState->pHair;
    // Free any HairGAS related memory previously allocated.
    cudaFree( reinterpret_cast<void*>( pState->deviceBufferHairGAS ) );
    pState->deviceBufferHairGAS = 0;
    pState->hHairGAS            = 0;

    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accelBuildOptions = {};
    accelBuildOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accelBuildOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    CUdeviceptr devicePoints                 = 0;
    CUdeviceptr deviceWidths                 = 0;
    CUdeviceptr deviceStrands                = 0;

    auto tempPoints = pState->pHair->points();
    createOnDevice( tempPoints, &devicePoints );
    createOnDevice( pHair->widths(), &deviceWidths );
    auto segments = pHair->segments();
    createOnDevice( segments, &deviceStrands );
    unsigned int numberOfHairSegments = static_cast<unsigned int>( segments.size() );

    // Curve build input.
    OptixBuildInput buildInput = {};

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
    switch( pHair->splineMode() )
    {
        case Hair::LINEAR_BSPLINE:
            buildInput.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
            break;
        case Hair::QUADRATIC_BSPLINE:
            buildInput.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
            break;
        case Hair::CUBIC_BSPLINE:
            buildInput.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
            break;
        default:
            SUTIL_ASSERT_MSG( false, "Invalid spline mode" );
    }
    buildInput.curveArray.numPrimitives        = numberOfHairSegments;
    buildInput.curveArray.vertexBuffers        = &devicePoints;
    buildInput.curveArray.numVertices          = static_cast<unsigned int>( tempPoints.size() );
    buildInput.curveArray.vertexStrideInBytes  = sizeof( float3 );
    buildInput.curveArray.widthBuffers         = &deviceWidths;
    buildInput.curveArray.widthStrideInBytes   = sizeof( float );
    buildInput.curveArray.normalBuffers        = 0;
    buildInput.curveArray.normalStrideInBytes  = 0;
    buildInput.curveArray.indexBuffer          = deviceStrands;
    buildInput.curveArray.indexStrideInBytes   = sizeof( int );
    buildInput.curveArray.flag                 = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.curveArray.primitiveIndexOffset = 0;

    OptixAccelBufferSizes bufferSizesGAS;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( pState->context,
                                               &accelBuildOptions,
                                               &buildInput,
                                               1,  // Number of build inputs
                                               &bufferSizesGAS ) );

    CUdeviceptr deviceTempBufferGAS;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &deviceTempBufferGAS ),
                            bufferSizesGAS.tempSizeInBytes ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &pState->deviceBufferHairGAS ),
                            bufferSizesGAS.outputSizeInBytes ) );

    OPTIX_CHECK( optixAccelBuild( pState->context,
                                  0,  // CUDA stream
                                  &accelBuildOptions,
                                  &buildInput,
                                  1,  // num build inputs
                                  deviceTempBufferGAS,
                                  bufferSizesGAS.tempSizeInBytes,
                                  pState->deviceBufferHairGAS,
                                  bufferSizesGAS.outputSizeInBytes,
                                  &pState->hHairGAS,
                                  nullptr,  // emitted property list
                                  0 ) );    // num emitted properties

    // We can now free the scratch space buffers used during build.
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( deviceTempBufferGAS ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( devicePoints ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( deviceWidths ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( deviceStrands ) ) );
}

void makeInstanceAccelerationStructure( HairState* pState )
{
    // Free any memory that has been previously allocated.
    cudaFree( reinterpret_cast<void*>( pState->deviceBufferIAS ) );
    pState->deviceBufferIAS = 0;
    pState->hIAS            = 0;

    std::vector<OptixInstance> instances;
    unsigned int               sbtOffset = 0;

    OptixInstance instance = {};
    // Common instance settings
    instance.instanceId           = 0;
    instance.visibilityMask       = 0xFF;
    instance.flags                = OPTIX_INSTANCE_FLAG_NONE;
    sutil::Matrix3x4 yUpTransform = {
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
    };
    // Head first
    if( pState->pHead )
    {
        memcpy( instance.transform, yUpTransform.getData(), sizeof( float ) * 12 );
        instance.sbtOffset         = sbtOffset;
        instance.traversableHandle = pState->pHead->traversable();
        sbtOffset += whitted::RAY_TYPE_COUNT;
        instances.push_back( instance );
        sutil::Aabb bb = pState->pHead->aabb();
        bb.transform( yUpTransform );
        pState->aabb.include( bb );
    }
    // Hair second
    if( pState->pHair )
    {
        memcpy( instance.transform, yUpTransform.getData(), sizeof( float ) * 12 );
        instance.sbtOffset         = sbtOffset;
        instance.traversableHandle = pState->hHairGAS;
        sbtOffset += whitted::RAY_TYPE_COUNT;
        instances.push_back( instance );
        sutil::Aabb bb = pState->pHair->aabb();
        bb.transform( yUpTransform );
        pState->aabb.include( bb );
    }

    CUdeviceptr deviceInstances = 0;
    createOnDevice( instances, &deviceInstances );

    // Instance build input.
    OptixBuildInput buildInput = {};

    buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances    = deviceInstances;
    buildInput.instanceArray.numInstances = static_cast<unsigned int>( instances.size() );

    OptixAccelBuildOptions accelBuildOptions = {};
    accelBuildOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accelBuildOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizesIAS;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( pState->context, &accelBuildOptions, &buildInput,
                                               1,  // Number of build inputs
                                               &bufferSizesIAS ) );

    CUdeviceptr deviceTempBufferIAS;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &deviceTempBufferIAS ),
                            bufferSizesIAS.tempSizeInBytes ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &pState->deviceBufferIAS ),
                            bufferSizesIAS.outputSizeInBytes ) );

    OPTIX_CHECK( optixAccelBuild( pState->context,
                                  0,  // CUDA stream
                                  &accelBuildOptions,
                                  &buildInput,
                                  1,  // num build inputs
                                  deviceTempBufferIAS,
                                  bufferSizesIAS.tempSizeInBytes,
                                  pState->deviceBufferIAS,
                                  bufferSizesIAS.outputSizeInBytes,
                                  &pState->hIAS,
                                  nullptr,  // emitted property list
                                  0 ) );    // num emitted properties

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( deviceInstances ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( deviceTempBufferIAS ) ) );
}

void initializeCamera( HairState* pState )
{
    const float aspectRatio = pState->width / static_cast<float>( pState->height );
    const float fovAngle    = 30.0f;
    pState->camera.setFovY( fovAngle );
    const float r        = length( pState->aabb.m_max - pState->aabb.center() );
    float       distance = r / sin( (float)M_PI / 180.0f * fovAngle );
    if( aspectRatio > 1.0f )
        distance *= aspectRatio;
    pState->camera.setLookat( pState->aabb.center() );
    pState->camera.setEye( pState->aabb.center() + make_float3( 0.0f, 0.0f, distance ) );
    pState->camera.setUp( {0.0f, 1.0f, 0.0f} );
    pState->camera.setAspectRatio( aspectRatio );
}

void updateSize( HairState* pState, int width, int height )
{
    pState->width           = width;
    pState->height          = height;
    const float aspectRatio = pState->width / static_cast<float>( pState->height );
    pState->camera.setAspectRatio( aspectRatio );
    pState->outputBuffer.resize( pState->width, pState->height );
    pState->accumBuffer.resize( pState->width, pState->height );
}

OptixPipelineCompileOptions defaultPipelineCompileOptions( HairState* pState )
{
    OptixPipelineCompileOptions pipeOptions = {};
    pipeOptions.usesMotionBlur              = false;
    pipeOptions.traversableGraphFlags       = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeOptions.numPayloadValues            = 4;
    pipeOptions.numAttributeValues          = 1;
#ifdef DEBUG  // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    pipeOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    pipeOptions.pipelineLaunchParamsVariableName = "params";

    unsigned int primitiveTypes = 0;
    if( pState->pHead )
        primitiveTypes |= pState->pHead->usesPrimitiveTypes();
    if( pState->pHair )
        primitiveTypes |= pState->pHair->usesPrimitiveTypes();
    pipeOptions.usesPrimitiveTypeFlags = primitiveTypes;

    return pipeOptions;
}

void makeProgramGroups( HairState* pState )
{
    delete( pState->pProgramGroups );
    pState->pProgramGroups = new HairProgramGroups( pState->context, defaultPipelineCompileOptions( pState ) );
    // Miss program groups
    OptixProgramGroupDesc programGroupDesc  = {};
    programGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    programGroupDesc.miss.module            = pState->pProgramGroups->m_whittedModule;
    programGroupDesc.miss.entryFunctionName = "__miss__constant_radiance";
    pState->pProgramGroups->add( programGroupDesc, "miss" );

    memset( &programGroupDesc, 0, sizeof( OptixProgramGroupDesc ) );
    programGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    programGroupDesc.miss.module            = nullptr;  // NULL program for occlusion rays
    programGroupDesc.miss.entryFunctionName = nullptr;
    pState->pProgramGroups->add( programGroupDesc, "missOcclude" );

    // add raygen group
    {
        OptixProgramGroupDesc programGroupDesc    = {};
        programGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        programGroupDesc.raygen.module            = pState->pProgramGroups->m_whittedModule;
        programGroupDesc.raygen.entryFunctionName = "__raygen__pinhole";
        pState->pProgramGroups->add( programGroupDesc, "raygen" );
    }
    if( pState->pHair )
        pState->pHair->gatherProgramGroups( pState->pProgramGroups );
    if( pState->pHead )
        pState->pHead->gatherProgramGroups( pState->pProgramGroups );
}

std::vector<HitRecord> hairSbtHitRecords( HairState* pState, const ProgramGroups& programs )
{
    // clear curves_ data
    cudaFree( reinterpret_cast<void*>( pState->curves.strand_u.data ) );
    pState->curves.strand_u.data = 0;
    cudaFree( reinterpret_cast<void*>( pState->curves.strand_i.data ) );
    pState->curves.strand_i.data = 0;
    cudaFree( reinterpret_cast<void*>( pState->curves.strand_info.data ) );
    pState->curves.strand_info.data = 0;

    std::vector<HitRecord> records;
    HitRecord              hitGroupRecord = {};

    switch( pState->pHair->splineMode() )
    {
        case Hair::LINEAR_BSPLINE:
            hitGroupRecord.data.geometry_data.type = GeometryData::LINEAR_CURVE_ARRAY;
            break;
        case Hair::QUADRATIC_BSPLINE:
            hitGroupRecord.data.geometry_data.type = GeometryData::QUADRATIC_CURVE_ARRAY;
            break;
        case Hair::CUBIC_BSPLINE:
            hitGroupRecord.data.geometry_data.type = GeometryData::CUBIC_CURVE_ARRAY;
            break;
        default:
            SUTIL_ASSERT_MSG( false, "Invalid spline mode." );
    }

    CUdeviceptr strandUs = 0;
    createOnDevice( pState->pHair->strandU(), &strandUs );
    pState->curves.strand_u.data        = strandUs;
    pState->curves.strand_u.byte_stride = static_cast<uint16_t>( sizeof( float2 ) );
    SUTIL_ASSERT( pState->pHair->numberOfSegments() == static_cast<int>( pState->pHair->strandU().size() ) );
    pState->curves.strand_u.count          = static_cast<uint16_t>( pState->pHair->numberOfSegments() );
    pState->curves.strand_u.elmt_byte_size = static_cast<uint16_t>( sizeof( float2 ) );
    CUdeviceptr strandIs                   = 0;
    createOnDevice( pState->pHair->strandIndices(), &strandIs );
    pState->curves.strand_i.data           = strandIs;
    pState->curves.strand_i.byte_stride    = static_cast<uint16_t>( sizeof( unsigned int ) );
    pState->curves.strand_i.count          = static_cast<uint16_t>( pState->pHair->numberOfSegments() );
    pState->curves.strand_i.elmt_byte_size = static_cast<uint16_t>( sizeof( unsigned int ) );
    CUdeviceptr strandInfos                = 0;
    createOnDevice( pState->pHair->strandInfo(), &strandInfos );
    pState->curves.strand_info.data           = strandInfos;
    pState->curves.strand_info.byte_stride    = static_cast<uint16_t>( sizeof( uint2 ) );
    pState->curves.strand_info.count          = static_cast<uint16_t>( pState->pHair->numberOfStrands() );
    pState->curves.strand_info.elmt_byte_size = static_cast<uint16_t>( sizeof( uint2 ) );

    hitGroupRecord.data.geometry_data.curves         = pState->curves;
    hitGroupRecord.data.material_data.pbr.base_color = {0.8f, 0.1f, 0.1f};
    hitGroupRecord.data.material_data.pbr.metallic   = 0.0f;
    hitGroupRecord.data.material_data.pbr.roughness  = 0.6f;

    std::string name = pState->pHair->programName() + pState->pHair->programSuffix();
    OPTIX_CHECK( optixSbtRecordPackHeader( programs[name], &hitGroupRecord ) );
    records.push_back( hitGroupRecord );

    OPTIX_CHECK( optixSbtRecordPackHeader( programs["occludeCurve"], &hitGroupRecord ) );
    records.push_back( hitGroupRecord );

    return records;
}


void makeSBT( HairState* pState )
{
    std::vector<MissRecord> missRecords;
    MissRecord              missRecord;
    OPTIX_CHECK( optixSbtRecordPackHeader( ( *pState->pProgramGroups )["miss"], &missRecord ) );
    missRecords.push_back( missRecord );
    OPTIX_CHECK( optixSbtRecordPackHeader( ( *pState->pProgramGroups )["missOcclude"], &missRecord ) );
    missRecords.push_back( missRecord );

    std::vector<HitRecord> hitRecords;
    // Head first
    if( pState->pHead )
    {
        std::vector<HitRecord> headRecords = pState->pHead->sbtHitRecords( *pState->pProgramGroups, whitted::RAY_TYPE_COUNT );
        std::copy( headRecords.begin(), headRecords.end(), std::back_inserter( hitRecords ) );
    }
    // Hair second
    if( pState->pHair )
    {
        std::vector<HitRecord> hairRecords = hairSbtHitRecords( pState, *pState->pProgramGroups );
        std::copy( hairRecords.begin(), hairRecords.end(), std::back_inserter( hitRecords ) );
    }

    // raygen record
    RayGenRecord raygenRecord;
    OPTIX_CHECK( optixSbtRecordPackHeader( ( *pState->pProgramGroups )["raygen"], &raygenRecord ) );

    cudaFree( reinterpret_cast<void*>( pState->SBT.raygenRecord ) );
    cudaFree( reinterpret_cast<void*>( pState->SBT.missRecordBase ) );
    cudaFree( reinterpret_cast<void*>( pState->SBT.hitgroupRecordBase ) );

    CUdeviceptr deviceRayGenRecord;
    createOnDevice( raygenRecord, &deviceRayGenRecord );
    CUdeviceptr deviceMissRecords;
    createOnDevice( missRecords, &deviceMissRecords );
    CUdeviceptr deviceHitGroupRecords;
    createOnDevice( hitRecords, &deviceHitGroupRecords );

    pState->SBT.raygenRecord                = deviceRayGenRecord;
    pState->SBT.missRecordBase              = deviceMissRecords;
    pState->SBT.missRecordStrideInBytes     = sizeof( MissRecord );
    pState->SBT.missRecordCount             = static_cast<unsigned int>( missRecords.size() );
    pState->SBT.hitgroupRecordBase          = deviceHitGroupRecords;
    pState->SBT.hitgroupRecordStrideInBytes = sizeof( HitRecord );
    pState->SBT.hitgroupRecordCount         = static_cast<unsigned int>( hitRecords.size() );
}

void makePipeline( HairState* pState )
{
    const uint32_t max_trace_depth = 2;

    if( pState->pipeline )
        OPTIX_CHECK( optixPipelineDestroy( pState->pipeline ) );
    OptixPipelineCompileOptions pipelineCompileOptions = defaultPipelineCompileOptions( pState );
    OptixPipelineLinkOptions    pipelineLinkOptions    = {};
    pipelineLinkOptions.maxTraceDepth                  = max_trace_depth;
    pipelineLinkOptions.debugLevel                     = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    OPTIX_CHECK_LOG2( optixPipelineCreate( pState->context,
                                           &pipelineCompileOptions,
                                           &pipelineLinkOptions,
                                           pState->pProgramGroups->data(),
                                           pState->pProgramGroups->size(),
                                           LOG,
                                           &LOG_SIZE,
                                           &pState->pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( unsigned int i = 0; i < pState->pProgramGroups->size(); ++i )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( pState->pProgramGroups->data()[i], &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                             0,  // maxCCDepth
                                             0,  // maxDCDEpth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( pState->pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}

void printLogMessage( unsigned int level, const char* tag, const char* message, void* /* cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << std::endl;
}

void initializeOptix( HairState* pState )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( nullptr ) );
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &printLogMessage;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( 0 /* default cuda context */, &options, &pState->context ) );
}

void initializeParams( HairState* pState )
{
    pState->params.accum_buffer = nullptr;  // Unused for the moment
    pState->params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    pState->params.subframe_index = 0u;

    const float loffset = 2.0f * pState->aabb.maxExtent();

    pState->params.miss_color = make_float3( 0.1f, 0.1f, 0.4f );
    pState->params.eye        = pState->camera.eye();
    pState->camera.UVWFrame( pState->params.U, pState->params.V, pState->params.W );

    pState->lights[0].type            = Light::Type::POINT;
    pState->lights[0].point.color     = {1.0f, 1.0f, 1.0f};
    pState->lights[0].point.intensity = 2.0f;
    pState->lights[0].point.position  = pState->aabb.center() + make_float3( loffset );
    pState->lights[0].point.falloff   = Light::Falloff::QUADRATIC;

    pState->lights[1].type            = Light::Type::POINT;
    pState->lights[1].point.color     = {1.0f, 1.0f, 1.0f};
    pState->lights[1].point.intensity = 2.0f;
    // headlight...slightly offset to the side of eye.
    pState->lights[1].point.position  = pState->camera.eye() + pState->params.U;
    pState->lights[1].point.falloff   = Light::Falloff::QUADRATIC;

    pState->params.lights.count = 2;
    createOnDevice( pState->lights, &pState->params.lights.data );

    pState->params.handle = pState->hIAS;
    createOnDevice( pState->params, reinterpret_cast<CUdeviceptr*>( &pState->deviceParams ) );
}

void updateParams( HairState* pState )
{
    pState->params.eye = pState->camera.eye();
    pState->camera.UVWFrame( pState->params.U, pState->params.V, pState->params.W );
    pState->lights[1].point.position = pState->camera.eye() + pState->params.U;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( pState->params.lights.data ),
                                 &pState->lights,
                                 sizeof( pState->lights ),
                                 cudaMemcpyHostToDevice,
                                 0  // stream
                                 ) );
}

void renderFrame( HairState* pState )
{
    // Launch
    pState->params.frame_buffer = pState->outputBuffer.map();
    pState->params.accum_buffer = pState->accumBuffer.map();
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( pState->deviceParams ),
                                 &pState->params,
                                 sizeof( whitted::LaunchParams ),
                                 cudaMemcpyHostToDevice,
                                 0  // stream
                                 ) );

    OPTIX_CHECK( optixLaunch( pState->pipeline,
                              0,  // stream
                              reinterpret_cast<CUdeviceptr>( pState->deviceParams ),
                              sizeof( whitted::LaunchParams ),
                              &( pState->SBT ),
                              pState->width,   // launch width
                              pState->height,  // launch height
                              1                // launch depth
                              ) );
    pState->outputBuffer.unmap();
    pState->accumBuffer.unmap();

    pState->params.subframe_index++;

    CUDA_SYNC_CHECK();
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 1024x768\n";
    std::cerr << "         --hair <model.hair>         Specify the hair model; defaults to \"Hair/wStraight.hair\"\n";
    std::cerr << "         --deg=<1|2|3>               Specify the curve degree; defaults to 3\n";
    std::cerr << "         --help | -h                 Print this usage message\n\n\n";
    std::cerr << "\n\nKeyboard commands:\n\n"
                 "  'q' (or 'ESC'): Quit the application.\n"
                 "  '1' linear b-spline interpretation of the geometry.\n"
                 "  '2' quadratic b-spline interpretation of the geometry.\n"
                 "  '3' cubic b-spline interpretation of the geometry.\n"
                 "  's' \"segment u\": lerp from red to green via  segment u,\n"
                 "      i.e. each segment starts green and ends red.\n"
                 "  'r' \"root-to-tip u\": lerp red to green with root-to-tip u,\n"
                 "      i.e. hair roots are red and tips are green.\n"
                 "  'i' \"index color\": assign one of five solid colors (green,\n"
                 "      blue, magenta, cyan, and yellow) based on a hair's index;\n"
                 "      tips lerp to red. The shader in this mode demonstrates\n"
                 "      how to compute a hair index from the primitive index.\n"
                 "      It also does root to tip shading but uses index based math\n"
                 "      to compute a contiguous u along the hair.\n"
                 "  'c' \"constant radius\" hair geometry.\n"
                 "  't' \"tapered radius\" hair geometry.\n";
    exit( 0 );
}

//
// Main program
//
int main( int argc, char* argv[] )
{
    //
    // Command-line parameter parsing
    //
    std::string      hairFile( "Hair/wStraight.hair" );
    std::vector<int> image_size( 2 );
    image_size[0]           = 1024;
    image_size[1]           = 786;
    int         curveDegree = 3;
    std::string outputFile;

    //
    // Parse command line options
    //
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--hair" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            hairFile = argv[++i];
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outputFile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), image_size[0], image_size[1] );
        }
        else if( arg.substr( 0, 6 ) == "--deg=" )
        {
            const std::string deg_arg = arg.substr( 6 );
            curveDegree               = atoi( deg_arg.c_str() );
            std::cerr << "curveDegree = " << curveDegree << std::endl;
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        HairState state = {};
        initializeOptix( &state );

        state.outputBuffer.setStream( 0 );  // CUDA stream
        state.accumBuffer.setStream( 0 );   // CUDA stream

        std::string hairFileName = sutil::sampleDataFilePath( hairFile.c_str() );
        Hair        hair( state.context, hairFileName );

        state.width  = image_size[0];
        state.height = image_size[1];
        state.outputBuffer.resize( state.width, state.height );
        state.accumBuffer.resize( state.width, state.height );

        if( 1 == curveDegree )
            hair.setSplineMode( Hair::LINEAR_BSPLINE );
        else if( 2 == curveDegree )
            hair.setSplineMode( Hair::QUADRATIC_BSPLINE );
        else if( 3 == curveDegree )
            hair.setSplineMode( Hair::CUBIC_BSPLINE );
        else
            SUTIL_ASSERT_MSG( false, "Invalid curve degree" );
        std::cout << hair << std::endl;
        state.pHair = &hair;

        std::string headFileName = sutil::sampleDataFilePath( "Hair/woman.gltf" );
        const Head  head( state.context, headFileName );
        std::cout << head << std::endl;
        state.pHead = &head;

        // with head and hair set put them into an IAS...
        makeHairGAS( &state );
        makeInstanceAccelerationStructure( &state );
        initializeCamera( &state );
        makeProgramGroups( &state );
        makePipeline( &state );
        makeSBT( &state );

        initializeParams( &state );

        if( !outputFile.empty() )  // render single frame to file
        {
            const FileRenderer renderer( &state );
            renderer.renderFile( outputFile.c_str() );
        }
        else
        {
            WindowRenderer renderer( &state );
            renderer.run();
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
