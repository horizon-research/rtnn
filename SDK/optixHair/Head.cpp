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
#include "Head.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#define TINYGLTF_IMPLEMENTATION
#if defined( WIN32 )
#pragma warning( push )
#pragma warning( disable : 4267 )
#endif
#include <support/tinygltf/tiny_gltf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <tinygltf/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/stb_image_write.h>
#if defined( WIN32 )
#pragma warning( pop )
#endif

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "ProgramGroups.h"
#include "Util.h"
#include "optixHair.h"


Head::Head( const OptixDeviceContext context, const std::string& fileName )

{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;

    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile( &model, &err, &warn, fileName );
    if( !warn.empty() )
        std::cout << "glTF WARNING: " << warn << std::endl;
    if( !ret )
    {
        std::cout << "Failed to load GLTF scene '" << fileName << "': " << err << std::endl;
        throw sutil::Exception( err.c_str() );
    }
    //
    // Process buffer data first -- buffer views will reference this list
    //
    SUTIL_ASSERT( 1 == model.buffers.size() );
    createOnDevice( model.buffers[0].data, &m_buffer );

    SUTIL_ASSERT( model.nodes.size() == 1 );
    SUTIL_ASSERT( model.nodes[0].mesh != -1 );
    const auto& gltfMesh = model.meshes[model.nodes[0].mesh];
    std::cout << "Processing glTF mesh: '" << gltfMesh.name << "'\n";
    std::cout << "\tNum mesh primitive groups: " << gltfMesh.primitives.size() << std::endl;
    SUTIL_ASSERT( gltfMesh.primitives.size() == 1 );

    auto primitive = gltfMesh.primitives[0];
    SUTIL_ASSERT( primitive.mode == TINYGLTF_MODE_TRIANGLES );

    // Indices

    std::cout << "Processing index buffer" << std::endl;
    SUTIL_ASSERT( primitive.indices != -1 );
    auto& accessor   = model.accessors[primitive.indices];
    auto& bufferView = model.bufferViews[accessor.bufferView];
    OptixBuildInput buildInput = {};

    m_triangleMesh.indices.data = m_buffer + bufferView.byteOffset + accessor.byteOffset;
    SUTIL_ASSERT( accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT );
    m_triangleMesh.indices.elmt_byte_size = accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ?
                                                2 :
                                                accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT ?
                                                4 :
                                                accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT ? 4 : 0;
    SUTIL_ASSERT_MSG( m_triangleMesh.indices.elmt_byte_size != 0, "gltf accessor component type not supported" );
    m_triangleMesh.indices.byte_stride = static_cast<unsigned int>( bufferView.byteStride ? bufferView.byteStride : m_triangleMesh.indices.elmt_byte_size );
    SUTIL_ASSERT( accessor.count % 3 == 0 );
    m_triangleMesh.indices.count = static_cast<unsigned int>( accessor.count );

    // index buffer build input
    buildInput.triangleArray.indexFormat =
        m_triangleMesh.indices.elmt_byte_size == 2 ? OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 : OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = m_triangleMesh.indices.byte_stride * 3;
    buildInput.triangleArray.numIndexTriplets   = m_triangleMesh.indices.count / 3;
    buildInput.triangleArray.indexBuffer        = m_triangleMesh.indices.data;
    const unsigned int triangleFlags            = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.triangleArray.flags              = &triangleFlags;
    buildInput.triangleArray.numSbtRecords      = 1;
    m_triangles                                 = m_triangleMesh.indices.count;

    // Vertex array
    SUTIL_ASSERT( primitive.attributes.find( "POSITION" ) != primitive.attributes.end() );
    const int32_t positionIndex = primitive.attributes.at( "POSITION" );
    std::cout << "Processing position array" << positionIndex << std::endl;
    SUTIL_ASSERT( positionIndex != -1 );
    accessor   = model.accessors[positionIndex];
    bufferView = model.bufferViews[accessor.bufferView];
    SUTIL_ASSERT( accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT );
    m_triangleMesh.positions.data           = m_buffer + bufferView.byteOffset + accessor.byteOffset;
    m_triangleMesh.positions.elmt_byte_size = accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ?
                                                  2 :
                                                  accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT ?
                                                  4 :
                                                  accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT ? 4 : 0;
    m_triangleMesh.positions.elmt_byte_size *= 3;
    SUTIL_ASSERT_MSG( m_triangleMesh.indices.elmt_byte_size != 0, "gltf accessor component type not supported" );
    m_triangleMesh.positions.byte_stride = static_cast<unsigned int>( bufferView.byteStride ? bufferView.byteStride : m_triangleMesh.positions.elmt_byte_size );
    m_triangleMesh.positions.count = static_cast<unsigned int>( accessor.count );
    // bounding box
    sutil::Aabb bb = sutil::Aabb( make_float3( (float) accessor.minValues[0], (float) accessor.minValues[1], (float) accessor.minValues[2] ),
                                  make_float3( (float) accessor.maxValues[0], (float) accessor.maxValues[1], (float) accessor.maxValues[2] ) );
    m_aabb.include( bb );
    m_vertices = m_triangleMesh.positions.count;

    // vertex buffer build input
    buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    SUTIL_ASSERT( m_triangleMesh.positions.byte_stride == sizeof( float3 ) );
    buildInput.triangleArray.vertexStrideInBytes = m_triangleMesh.positions.byte_stride;
    buildInput.triangleArray.numVertices         = m_triangleMesh.positions.count;
    buildInput.triangleArray.vertexBuffers       = &m_triangleMesh.positions.data;

    // Normal array
    auto normalAccessorIter = primitive.attributes.find( "NORMAL" );
    SUTIL_ASSERT( normalAccessorIter != primitive.attributes.end() );
    const int32_t normalIndex = normalAccessorIter->second;
    std::cout << "Processing normal array" << std::endl;
    accessor   = model.accessors[normalIndex];
    bufferView = model.bufferViews[accessor.bufferView];
    m_triangleMesh.normals.data             = m_buffer + bufferView.byteOffset + accessor.byteOffset;
    m_triangleMesh.normals.byte_stride      = static_cast<unsigned int>( bufferView.byteStride ? bufferView.byteStride : sizeof( float3 ) );
    m_triangleMesh.normals.count            = static_cast<unsigned int>( accessor.count );
    m_triangleMesh.positions.elmt_byte_size = accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ?
                                                  2 :
                                                  accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT ?
                                                  4 :
                                                  accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT ? 4 : 0;
    m_triangleMesh.positions.elmt_byte_size *= 3;
    SUTIL_ASSERT_MSG( m_triangleMesh.indices.elmt_byte_size != 0, "gltf accessor component type not supported" );
    std::cout << "Build input type: " << buildInput.type << std::endl;

    OptixAccelBufferSizes  bufferSizes;
    OptixAccelBuildOptions accelBuildOptions = {};
    accelBuildOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accelBuildOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accelBuildOptions, &buildInput, 1, &bufferSizes ) );

    CUdeviceptr deviceTempBuffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &deviceTempBuffer ), bufferSizes.tempSizeInBytes ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_deviceBufferGAS ), bufferSizes.outputSizeInBytes ) );

    OPTIX_CHECK( optixAccelBuild( context, 0,  // CUDA stream
                                  &accelBuildOptions, &buildInput, 1, deviceTempBuffer, bufferSizes.tempSizeInBytes,
                                  m_deviceBufferGAS, bufferSizes.outputSizeInBytes, &m_hGAS,
                                  nullptr,  // emitted property list
                                  0 ) );    // num emitted properties

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( deviceTempBuffer ) ) );
}

Head::~Head()
{

    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( m_buffer ) ) );
}

void Head::gatherProgramGroups( HairProgramGroups* pProgramGroups ) const
{
    OptixProgramGroupDesc programGroupDesc        = {};
    programGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programGroupDesc.hitgroup.moduleCH            = pProgramGroups->m_whittedModule;
    programGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pProgramGroups->add( programGroupDesc, "hitTriangle" );

    memset( &programGroupDesc, 0, sizeof( OptixProgramGroupDesc ) );
    programGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programGroupDesc.hitgroup.moduleCH            = pProgramGroups->m_whittedModule;
    programGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
    pProgramGroups->add( programGroupDesc, "occludeTriangle" );
}

std::vector<HitRecord> Head::sbtHitRecords( const ProgramGroups& programs, size_t rayTypes ) const
{
    SUTIL_ASSERT_MSG( 2 == rayTypes, "Head requires two ray types." );
    std::vector<HitRecord> records;

    HitRecord hitGroupRecord = {};

    hitGroupRecord.data.geometry_data.type           = GeometryData::TRIANGLE_MESH;
    hitGroupRecord.data.geometry_data.triangle_mesh  = m_triangleMesh;
    hitGroupRecord.data.material_data.pbr.base_color = {0.5f, 0.5f, 0.5f};
    hitGroupRecord.data.material_data.pbr.metallic   = 0.2f;
    hitGroupRecord.data.material_data.pbr.roughness  = 1.0f;
    OPTIX_CHECK( optixSbtRecordPackHeader( programs["hitTriangle"], &hitGroupRecord ) );
    records.push_back( hitGroupRecord );

    OPTIX_CHECK( optixSbtRecordPackHeader( programs["occludeTriangle"], &hitGroupRecord ) );
    records.push_back( hitGroupRecord );

    return records;
}

OptixTraversableHandle Head::traversable() const
{
    return m_hGAS;
}

std::ostream& operator<<( std::ostream& o, const Head& head )
{
    o << "Head: " << std::endl;
    o << "Number of vertices:         " << head.numberOfVertices() << std::endl;
    o << "Number of triangles:        " << head.numberOfTriangles() << std::endl;
    o << "Bounding box: [" << head.m_aabb.m_min << ", " << head.m_aabb.m_max << "]" << std::endl;

    return o;
}
