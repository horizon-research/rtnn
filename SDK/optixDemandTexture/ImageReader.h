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

#include <cuda_runtime.h>

namespace demandLoading {

/// Abstract base class for a mipmapped image.
class ImageReader
{
  public:
    /// Image info, including dimensions and format.
    struct Info
    {
        unsigned int          width;
        unsigned int          height;
        unsigned int          tileWidth;
        unsigned int          tileHeight;
        cudaChannelFormatDesc channelDesc;
        unsigned int          numMipLevels;
    };

    /// The destructor is virtual to ensure that instances of derived classes are properly destroyed.
    virtual ~ImageReader() {}

    /// Open the image and read header info, including dimensions and format.  Returns false on error.
    virtual bool open( Info* info ) = 0;

    /// Close the image.
    virtual void close() = 0;

    /// Get the image info.  Valid only after calling open().
    virtual const Info& getInfo() = 0;

    /// Read the specified miplevel into the given buffer.  Returns true for success.
    virtual bool readMipLevel( char* dest, unsigned int miplevel, unsigned int width, unsigned int height ) = 0;
};

/// Get the number of bits per pixel
inline int getBitsPerPixel( const cudaChannelFormatDesc& channelDesc ) 
{ 
    return channelDesc.x + channelDesc.y + channelDesc.z + channelDesc.w; 
}

} // namespace demandLoading
