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
#include <CheckerBoardImage.h>
#include <algorithm>
#include <cmath>

#include <cstring>

namespace demandLoading {
    
CheckerBoardImage::CheckerBoardImage( unsigned int width, unsigned int height )
    : m_info( {width, height, /*tileWidth=*/32, /*tileHeight=*/32, /*channelDesc=*/0, /*numMipLevels=*/0} )
{
    unsigned int dim    = std::max( width, height );
    m_info.numMipLevels = 1 + static_cast<unsigned int>( std::ceil( std::log2f( static_cast<float>( dim ) ) ) );

    // We fill in the cudaChannelFormatDesc struct ourselves rather than
    // calling cudaCreateChannelDesc to avoid a CUDA API dependency here.
    const int floatBits = sizeof( float ) * 8;
    m_info.channelDesc  = cudaChannelFormatDesc{floatBits, floatBits, floatBits, floatBits, cudaChannelFormatKindFloat};

    // Use a different color per miplevel.
    std::vector<float4> colors{
        {255, 0, 0, 0},    // red
        {255, 127, 0, 0},  // orange
        {255, 255, 0, 0},  // yellow
        {0, 255, 0, 0},    // green
        {0, 0, 255, 0},    // blue
        {127, 0, 0, 0},    // dark red
        {127, 63, 0, 0},   // dark orange
        {127, 127, 0, 0},  // dark yellow
        {0, 127, 0, 0},    // dark green
        {0, 0, 127, 0},    // dark blue
    };
    // Normalize the miplevel colors to [0,1]
    for( float4& color : colors )
    {
        color.x /= 255.f;
        color.y /= 255.f;
        color.z /= 255.f;
    }
    m_mipLevelColors.swap( colors );
}

bool CheckerBoardImage::open( Info* info )
{
    *info = m_info;
    return true;
}

bool CheckerBoardImage::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height )
{
    // Create a checkerboard pattern with a color based on the miplevel.
    float4 black = make_float4( 0.f, 0.f, 0.f, 0.f );
    float4 color = m_mipLevelColors.at( mipLevel );

    std::vector<float4> pixels( width * height );

    unsigned int gridSize = std::max( 1, 16 >> mipLevel );
    for( unsigned int y = 0; y < height; ++y )
    {
        float4* row = pixels.data() + y * width;
        for( unsigned int x = 0; x < width; ++x )
        {
            bool a = x / gridSize % 2 != 0;
            bool b = y / gridSize % 2 != 0;
            row[x] = ( a && b ) || ( !a && !b ) ? color : black;
        }
    }

    // Copy host buffer into the CUDA array for this miplevel.
    memcpy( dest, pixels.data(), pixels.size() * sizeof( float4 ) );
    return true;
}

} // namespace demandLoading
