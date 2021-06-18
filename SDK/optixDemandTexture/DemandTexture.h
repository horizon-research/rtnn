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
#pragma once

#include "optixDemandTexture.h"
#include <ImageReader.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace demandLoading {

/// Demand-loaded textures are created by the DemandTextureManager.  This base class describes the
/// public interface, while the derived class below is employed by the DemandTextureManager.
class DemandTexture
{
  public:
    /// DemandTexture is constructed by DemandTextureManager::createTexture
    DemandTexture( unsigned int id, std::shared_ptr<ImageReader> image );

    /// DemandTexture instances are owned by the DemandTextureManager
    ~DemandTexture();

    /// Get the texture id, which is used as an index into the device-side DemandTextureSampler array.
    unsigned int getId() const { return m_id; }

    /// Get the image info.  Valid only after the image has been initialized (e.g. opened).
    const ImageReader::Info& getInfo() const
    {
        assert( m_isInitialized );
        return m_info;
    }

    /// Get the texture sampler, which bundles the CUDA texture object with additional info.
    DemandTextureSampler getSampler() const { return DemandTextureSampler{ m_texture }; }

    /// Initialize the texture, e.g. reading image info from file header.  Returns false on error.
    bool init();

    /// Reallocate backing storage to span the specified miplevels.
    void reallocate( unsigned int minMipLevel, unsigned int maxMipLevel );

    /// Fill the specified miplevel.
    void fillMipLevel( unsigned int mipLevel );

    /// Get the min and max miplevels that are currently allocated.
    unsigned int getMinMipLevel() const { return m_minMipLevel; }
    unsigned int getMaxMipLevel() const { return m_maxMipLevel; }

  private:
    // The texture identifier is used as an index into the device-side DemandTextureSampler array.
    unsigned int m_id = 0;

    // The image provides a read() method that fills requested miplevels.
    std::shared_ptr<ImageReader> m_image;

    // The image is lazily intitialized (e.g. opened).
    bool m_isInitialized = false;

    // Image info, including dimensions and format.  Not valid until the image is initialized.
    ImageReader::Info m_info = {};

    // The backing storage is a CUDA mipmapped array.
    cudaMipmappedArray_t m_mipLevelData = {};

    // CUDA texture object.
    cudaTextureObject_t m_texture = {};

    // Host buffer for filling miplevels.
    std::vector<char> m_hostMipLevel;

    // Current min and max miplevels
    unsigned int m_minMipLevel = std::numeric_limits<unsigned int>::max();
    unsigned int m_maxMipLevel = 0;

    // Create CUDA texture object (called internally after reallocation).
    cudaTextureObject_t createTextureObject() const;

    void destroyMipmapAndTextureObject( cudaMipmappedArray_t mipMap );

    // Get the width of the specified miplevel.
    unsigned int getLevelWidth( unsigned int mipLevel ) const { return getInfo().width >> mipLevel; }

    // Get the height of the specified miplevel.
    unsigned int getLevelHeight( unsigned int mipLevel ) const { return getInfo().height >> mipLevel; }
};

}  // namespace demandLoading
