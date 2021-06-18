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

#include <DemandTextureManager.h>
#include <sutil/Exception.h>

#include <cassert>
#include <cstring>

const unsigned int NUM_PAGES            = 1024 * 1024;  // 1M 64KB pages => 64 GB (virtual)
const unsigned int MAX_REQUESTED_PAGES  = 1024;
const unsigned int MAX_NUM_FILLED_PAGES = 1024;

namespace demandLoading {

// Construct demand texture manager, initializing the OptiX paging library.
DemandTextureManager::DemandTextureManager()
{
    // Configure the paging library.
    OptixPagingOptions options{NUM_PAGES, NUM_PAGES};
    optixPagingCreate( &options, &m_pagingContext );
    OptixPagingSizes sizes{};
    optixPagingCalculateSizes( options.initialVaSizeInPages, sizes );

    // Allocate device memory required by the paging library.
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_pagingContext->pageTable ), sizes.pageTableSizeInBytes ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_pagingContext->usageBits ), sizes.usageBitsSizeInBytes ) );
    optixPagingSetup( m_pagingContext, sizes, 1 );

    // Allocate device memory that is used to call paging library routines.
    // These allocations are retained to reduce allocation overhead.
    CUDA_CHECK( cudaMalloc( &m_devRequestedPages, MAX_REQUESTED_PAGES * sizeof( unsigned int ) ) );
    CUDA_CHECK( cudaMalloc( &m_devNumPagesReturned, 3 * sizeof( unsigned int ) ) );
    CUDA_CHECK( cudaMalloc( &m_devFilledPages, MAX_NUM_FILLED_PAGES * sizeof( PageMapping ) ) );
}

DemandTextureManager::~DemandTextureManager()
{
    try
    {
        // Free device memory and destroy the paging system.
        CUDA_CHECK( cudaFree( m_pagingContext->pageTable ) );
        CUDA_CHECK( cudaFree( m_pagingContext->usageBits ) );
        optixPagingDestroy( m_pagingContext );

        CUDA_CHECK( cudaFree( m_devRequestedPages ) );
        CUDA_CHECK( cudaFree( m_devNumPagesReturned ) );
        CUDA_CHECK( cudaFree( m_devFilledPages ) );
    }
    catch( ... )
    {
    }
}

// Extract texture id from page id.
static unsigned int getTextureId( unsigned int pageId )
{
    return pageId >> 4;
}

// Extract miplevel from page id.
static unsigned int getMipLevel( unsigned int pageId )
{
    return pageId & 0x0F;
}

// Create a demand-loaded texture with the specified dimensions and format.  The texture initially has no
// backing storage.
const DemandTexture& DemandTextureManager::createTexture( std::shared_ptr<ImageReader> imageReader )
{
    // Add new texture to the end of the list of textures.  The texture identifier is simply its
    // index in the DemandTexture array, which also serves as an index into the device-side
    // DemandTextureSampler array.  The texture holds a pointer to the image, from which miplevel
    // data is obtained on demand.
    unsigned int textureId = static_cast<unsigned int>( m_textures.size() );
    m_textures.emplace_back( DemandTexture( textureId, imageReader ) );
    DemandTexture& texture = m_textures.back();

    // Create texture sampler, which will be synched to the device in launchPrepare().  Note that we
    // don't set m_hostSamplersDirty when adding new samplers.
    m_hostSamplers.emplace_back( texture.getSampler() );

    return texture;
}

// Prepare for launch, updating device-side demand texture samplers.
void DemandTextureManager::launchPrepare()
{
    // Are there new samplers?
    size_t numOldSamplers = m_numDevSamplers;
    size_t numNewSamplers = m_textures.size() - numOldSamplers;
    if( numNewSamplers == 0 )
    {
        // No new samplers.  Sync existing texture samplers to device if they're dirty.
        if( m_hostSamplersDirty )
        {
            CUDA_CHECK( cudaMemcpy( m_devSamplers, m_hostSamplers.data(),
                                    m_hostSamplers.size() * sizeof( DemandTextureSampler ), cudaMemcpyHostToDevice ) );
            m_hostSamplersDirty = false;
        }
    }
    else
    {
        // Reallocate device sampler array.
        DemandTextureSampler* oldSamplers = m_devSamplers;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_devSamplers ), m_textures.size() * sizeof( DemandTextureSampler ) ) );

        // If any samplers are dirty (e.g. textures were reallocated), copy them all from the host.
        if( m_hostSamplersDirty )
        {
            CUDA_CHECK( cudaMemcpy( m_devSamplers, m_hostSamplers.data(),
                                    m_hostSamplers.size() * sizeof( DemandTextureSampler ), cudaMemcpyHostToDevice ) );
            m_hostSamplersDirty = false;
        }
        else
        {
            // Otherwise copy the old samplers from device memory and the new samplers from host memory.
            if( numOldSamplers > 0 )
            {
                CUDA_CHECK( cudaMemcpy( m_devSamplers, oldSamplers, numOldSamplers * sizeof( DemandTextureSampler ),
                                        cudaMemcpyDeviceToDevice ) );
            }
            CUDA_CHECK( cudaMemcpy( m_devSamplers + numOldSamplers, &m_hostSamplers[numOldSamplers],
                                    numNewSamplers * sizeof( DemandTextureSampler ), cudaMemcpyHostToDevice ) );
        }
        CUDA_CHECK( cudaFree( oldSamplers ) );
        m_numDevSamplers = m_textures.size();
    }
}

// Process requests for missing miplevels (from optixPagingMapOrRequest), reallocating textures
// and invoking callbacks to fill the new miplevels.
int DemandTextureManager::processRequests()
{
    std::vector<unsigned int> requestedPages;
    pullRequests( requestedPages );
    return processRequestsImpl( requestedPages );
}

// Get page requests from the device (via optixPagingPullRequests).
void DemandTextureManager::pullRequests( std::vector<unsigned int>& requestedPages )
{
    // Get a list of requested page ids, along with lists of stale and evictable pages (which are
    // currently unused).
    optixPagingPullRequests( m_pagingContext, m_devRequestedPages, MAX_REQUESTED_PAGES, nullptr /*stalePages*/, 0,
                             nullptr /*evictablePages*/, 0, m_devNumPagesReturned );

    // Get the sizes of the requsted, stale, and evictable page lists.
    unsigned int numReturned[3] = {0};
    CUDA_CHECK( cudaMemcpy( &numReturned[0], m_devNumPagesReturned, 3 * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );

    // Return early if no pages requested.
    unsigned int numRequests = numReturned[0];
    if( numRequests == 0 )
        return;

    // Copy the requested page list from this device.
    requestedPages.resize( numRequests );
    CUDA_CHECK( cudaMemcpy( requestedPages.data(), m_devRequestedPages, numRequests * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
}

// Process requests.  Implemented as a separate method to permit testing.
int DemandTextureManager::processRequestsImpl( std::vector<unsigned int>& requestedPages )
{
    if( requestedPages.empty() )
        return 0;

    // Sort the requests by page number.  This ensures that all the requests for a particular
    // texture are adjacent in the request list, allowing us to perform a single reallocation that
    // spans all the requested miplevels.
    std::sort( requestedPages.begin(), requestedPages.end() );

    // Reallocate textures to accommodate newly requested miplevels.
    size_t numRequests = requestedPages.size();
    for( size_t i = 0; i < numRequests; /* nop */ )
    {
        unsigned int     pageId    = requestedPages[i];
        unsigned int textureId = getTextureId( pageId );

        // Initialize the texture if necessary, e.g. reading image info from file header.
        DemandTexture* texture = &m_textures[textureId];
        bool           ok      = texture->init();
        ok, assert( ok );  // TODO: handle image reader errors.

        unsigned int requestedMipLevel = getMipLevel( pageId );
        requestedMipLevel              = std::min( requestedMipLevel, texture->getInfo().numMipLevels );

        unsigned int minMipLevel = requestedMipLevel;
        unsigned int maxMipLevel = requestedMipLevel;

        // Accumulate requests for other miplevels from the same texture.
        for( ++i; i < numRequests && getTextureId( requestedPages[i] ) == textureId; ++i )
        {
            unsigned int     pageId            = requestedPages[i];
            unsigned int requestedMipLevel = getMipLevel( pageId );

            minMipLevel = std::min( minMipLevel, requestedMipLevel );
            maxMipLevel = std::max( maxMipLevel, requestedMipLevel );
        }

        // Reallocate the texture's backing storage to accomodate the new miplevels.
        // Existing miplevels are copied (using a device-to-device memcpy).
        texture->reallocate( minMipLevel, maxMipLevel );

        // Update the host-side texture sampler.
        m_hostSamplers[textureId] = texture->getSampler();
        m_hostSamplersDirty       = true;
    }

    // Fill each requested miplevel.
    std::vector<PageMapping> filledPages;
    for( size_t i = 0; i < numRequests; ++i )
    {
        unsigned int       pageId   = requestedPages[i];
        DemandTexture& texture  = m_textures[getTextureId( pageId )];
        unsigned int   mipLevel = getMipLevel( pageId );
        texture.fillMipLevel( mipLevel );

        // Keep track of which pages were filled.  (The value of the page table entry is not used.)
        filledPages.push_back( PageMapping{pageId, 1} );
    }

    // Push the new page mappings to the device.
    unsigned int numFilledPages = static_cast<unsigned int>( filledPages.size() );
    CUDA_CHECK( cudaMemcpy( m_devFilledPages, filledPages.data(), numFilledPages * sizeof( PageMapping ), cudaMemcpyHostToDevice ) );
    optixPagingPushMappings( m_pagingContext, m_devFilledPages, numFilledPages, nullptr /*invalidatedPages*/, 0 );

    return static_cast<int>( filledPages.size() );
}

} // namespace demandLoading
