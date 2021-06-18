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

#include <ImageReader.h>
#include <vector>

namespace demandLoading {

/// If OpenEXR is not available, this test image is used.  It generates a
/// procedural pattern, rather than loading image data from disk.
class CheckerBoardImage : public ImageReader
{
  public:
    /// Create a test image with the specified dimensions.
    CheckerBoardImage( unsigned int width, unsigned int height );

    /// The open method simply initializes the given image info struct.
    bool open( Info* info ) override;

    /// The close operation is a no-op.
    void close() override {}

    /// Get the image info.  Valid only after calling open().
    const Info& getInfo() override { return m_info; }

    /// Fill the given buffer with the test image data for the specified miplevel.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height ) override;

  private:
    Info                m_info;
    std::vector<float4> m_mipLevelColors;
};

} // namespace demandLoading
