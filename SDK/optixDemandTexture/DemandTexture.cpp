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

#include <DemandTexture.h>
#include <sutil/Exception.h>

#include <cassert>
#include <cstring>

namespace demandLoading {

// DemandTexture is constructed by DemandTextureManager::createTexture
DemandTexture::DemandTexture( unsigned int id, std::shared_ptr<ImageReader> image )
    : m_id( id )
    , m_image( image )
{
}

DemandTexture::~DemandTexture()
{
    destroyMipmapAndTextureObject( m_mipLevelData );
}

// Initialize the texture, e.g. reading image info from file header.  Returns false on error.
bool DemandTexture::init()
{
    if( !m_isInitialized )
    {
        m_isInitialized = true;
        return m_image->open( &m_info );
    }
    return true;
}

// Reallocate backing storage to span the specified miplevels.
void DemandTexture::reallocate( unsigned int minMipLevel, unsigned int maxMipLevel )
{
    unsigned int oldMinMipLevel = m_minMipLevel;
    unsigned int oldMaxMipLevel = m_maxMipLevel;
    if( minMipLevel >= oldMinMipLevel && maxMipLevel <= oldMaxMipLevel )
        return;

    unsigned newMinMipLevel = std::min( oldMinMipLevel, minMipLevel );
    unsigned newMaxMipLevel = std::max( oldMaxMipLevel, maxMipLevel );
    m_minMipLevel           = newMinMipLevel;
    m_maxMipLevel           = newMaxMipLevel;

    unsigned int newWidth  = getLevelWidth( newMinMipLevel );
    unsigned int newHeight = getLevelHeight( newMinMipLevel );
    unsigned int numLevels = newMaxMipLevel - newMinMipLevel + 1;

    // Allocate new array.
    cudaMipmappedArray_t         newMipLevelData;
    const cudaChannelFormatDesc& channelDesc = getInfo().channelDesc;
    cudaExtent                   extent      = make_cudaExtent( newWidth, newHeight, 0 );
    CUDA_CHECK( cudaMallocMipmappedArray( &newMipLevelData, &channelDesc, extent, numLevels ) );

    // Copy any existing levels from the old array.
    cudaMipmappedArray_t oldMipLevelData = m_mipLevelData;
    m_mipLevelData                       = newMipLevelData;
    for( unsigned int nominalLevel = oldMinMipLevel; nominalLevel <= oldMaxMipLevel; ++nominalLevel )
    {
        unsigned int sourceLevel  = nominalLevel - oldMinMipLevel;
        unsigned int destLevel    = nominalLevel - newMinMipLevel;
        unsigned int width        = getLevelWidth( nominalLevel );
        unsigned int height       = getLevelHeight( nominalLevel );
        unsigned int widthInBytes = width * getBitsPerPixel( getInfo().channelDesc ) / 8;

        // Get the CUDA arrays for the source and destination miplevels.
        cudaArray_t sourceArray, destArray;
        CUDA_CHECK( cudaGetMipmappedArrayLevel( &sourceArray, oldMipLevelData, sourceLevel ) );
        CUDA_CHECK( cudaGetMipmappedArrayLevel( &destArray, newMipLevelData, destLevel ) );

        // Copy the miplevel.
        CUDA_CHECK( cudaMemcpy2DArrayToArray( destArray, 0, 0, sourceArray, 0, 0, widthInBytes, height, cudaMemcpyDeviceToDevice ) );
    }

    // Destroy the old mipmapped array and the old texture.

    destroyMipmapAndTextureObject( oldMipLevelData );

    // Create new texture object.
    m_texture = createTextureObject();
}

// Create CUDA texture object (called internally after reallocation).
cudaTextureObject_t DemandTexture::createTextureObject() const
{
    // Create resource description
    cudaResourceDesc resDesc  = {};
    resDesc.resType           = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = m_mipLevelData;

    // Construct texture description with various options.
    cudaTextureDesc texDesc     = {};
    texDesc.addressMode[0]      = cudaAddressModeWrap;
    texDesc.addressMode[1]      = cudaAddressModeWrap;
    texDesc.filterMode          = cudaFilterModeLinear;
    texDesc.maxMipmapLevelClamp = static_cast<float>( m_maxMipLevel );
    texDesc.minMipmapLevelClamp = 0.f;
    texDesc.mipmapFilterMode    = cudaFilterModeLinear;
    texDesc.normalizedCoords    = 1;
    texDesc.readMode            = cudaReadModeElementType;
    texDesc.maxAnisotropy       = 16;

    // Bias miplevel access in demand loaded texture based on the current minimum miplevel loaded.
    texDesc.mipmapLevelBias = -static_cast<float>( m_minMipLevel );

    // Create texture object
    cudaTextureObject_t texture;
    CUDA_CHECK( cudaCreateTextureObject( &texture, &resDesc, &texDesc, nullptr /*cudaResourceViewDesc*/ ) );
    return texture;
}

void DemandTexture::destroyMipmapAndTextureObject( cudaMipmappedArray_t mipMap )
{
    try
    {
        CUDA_CHECK( cudaFreeMipmappedArray( mipMap ) );
        CUDA_CHECK( cudaDestroyTextureObject( m_texture ) );
    }
    catch( ... )
    {
        cudaGetLastError();
    }
}

// Fill the specified miplevel.
void DemandTexture::fillMipLevel( unsigned int nominalMipLevel )
{
    // We retain the host-side fill buffer to amortize allocation overhead.
    unsigned int width         = getLevelWidth( nominalMipLevel );
    unsigned int height        = getLevelHeight( nominalMipLevel );
    int          bytesPerPixel = getBitsPerPixel( getInfo().channelDesc ) / 8;
    m_hostMipLevel.resize( width * height * bytesPerPixel );

    // Read the requested miplevel into the host buffer.
    bool ok = m_image->readMipLevel( m_hostMipLevel.data(), nominalMipLevel, width, height );
    ok, assert( ok );  // TODO: handle image read failure.

    // Get the backing storage for the specified miplevel.
    assert( m_minMipLevel <= nominalMipLevel && nominalMipLevel <= m_maxMipLevel );
    unsigned int actualMipLevel = nominalMipLevel - m_minMipLevel;
    cudaArray_t  array;
    CUDA_CHECK( cudaGetMipmappedArrayLevel( &array, m_mipLevelData, actualMipLevel ) );

    // Copy data into the miplevel on the device.
    size_t widthInBytes = width * getBitsPerPixel( getInfo().channelDesc ) / 8;
    size_t pitch        = widthInBytes;
    CUDA_CHECK( cudaMemcpy2DToArray( array, 0, 0, m_hostMipLevel.data(), pitch, widthInBytes, height, cudaMemcpyHostToDevice ) );
}

}  // namespace demandLoading
