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

#include <optix.h>
#include <optixPaging/optixPaging.h>
#include <sutil/vec_math.h>

/// DemandTextureSampler contains the device-side info required for a demand texture fetch.
/// The index of a DemandTextureSampler is passed to the closest hit shader via a hit group
/// record in the SBT, and a table of samplers is available as a launch parameter.
struct DemandTextureSampler
{
    /// The CUDA texture object.
    cudaTextureObject_t texture;
};

struct Sphere
{
    float3 center;
    float  radius;

    OptixAabb bounds() const
    {
        float3 m_min = center - radius;
        float3 m_max = center + radius;

        OptixAabb aabb = {m_min.x, m_min.y, m_min.z, m_max.x, m_max.y, m_max.z};
        return aabb;
    }
};


struct Params
{
    uchar4*                     image;
    unsigned int                image_width;
    unsigned int                image_height;
    int                         origin_x;
    int                         origin_y;
    OptixTraversableHandle      handle;
    OptixPagingContext          pagingContext;
    const DemandTextureSampler* demandTextures;
};


struct RayGenData
{
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};


struct MissData
{
    float r, g, b;
};


struct HitGroupData
{
    Sphere       sphere;
    unsigned int demand_texture_id;
    float        texture_scale;
    float        texture_lod;
};
