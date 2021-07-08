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

#include <vector_types.h>
#include <optix_device.h>
#include "optixRangeSearch.h"
#include "random.h"
#include "helpers.h"
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__knn()
{
    const uint3 idx = optixGetLaunchIndex();
    unsigned int rayIdx = idx.x;

    // if d_r2q_map is null, it could be 1) an unsorted run, 2) an initial run
    // for sorting, or 3) a sorted run with queries pre-gathered. Either case,
    // we directly map rays to queries.

    unsigned int queryIdx;
    if (params.d_r2q_map == nullptr)
      queryIdx = rayIdx;
    else
      queryIdx = params.d_r2q_map[rayIdx];

    float3 ray_origin = params.queries[queryIdx];
    float3 ray_direction = normalize(make_float3(1, 0, 0));

    unsigned int id = 0;
    const float tmin = 0.f;
    const float tmax = 1.e-16f;

    // TODO: change this to max float. when this get re-casted to float, it will be a big float.
    float a0_key = 10e6;
    unsigned int a0_id = 0xFFFFFFFF;
    float a1_key = 0.f;
    unsigned int a1_id = 0xFFFFFFFF;
    float a2_key = 0.f;
    unsigned int a2_id = 0xFFFFFFFF;

    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT |
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        RAY_TYPE_RADIANCE,
        1,
        RAY_TYPE_RADIANCE,
        reinterpret_cast<unsigned int&>(queryIdx),
        reinterpret_cast<unsigned int&>(a0_key),
        reinterpret_cast<unsigned int&>(a0_id),
        reinterpret_cast<unsigned int&>(a1_key),
        reinterpret_cast<unsigned int&>(a1_id),
        reinterpret_cast<unsigned int&>(a2_key),
        reinterpret_cast<unsigned int&>(a2_id)
    );

    params.frame_buffer[queryIdx * params.knn] = a0_id;
    params.frame_buffer[queryIdx * params.knn + 1] = a1_id;
    params.frame_buffer[queryIdx * params.knn + 2] = a2_id;
}

extern "C" __global__ void __raygen__pinhole_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    unsigned int rayIdx = idx.x;

    // if d_r2q_map is null, it could be 1) an unsorted run, 2) an initial run
    // for sorting, or 3) a sorted run with queries pre-gathered. Either case,
    // we directly map rays to queries.

    unsigned int queryIdx;
    if (params.d_r2q_map == nullptr)
      queryIdx = rayIdx;
    else
      queryIdx = params.d_r2q_map[rayIdx];

    float3 ray_origin = params.queries[queryIdx];
    float3 ray_direction = normalize(make_float3(1, 0, 0));

    unsigned int id = 0;
    const float tmin = 0.f;
    const float tmax = 1.e-16f;

    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        //OPTIX_RAY_FLAG_DISABLE_ANYHIT |
        //OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        RAY_TYPE_RADIANCE,
        1,
        RAY_TYPE_RADIANCE,
        reinterpret_cast<unsigned int&>(queryIdx),
        reinterpret_cast<unsigned int&>(id)
    );
    //params.frame_buffer[queryIdx * params.knn] = queryIdx;
    //params.frame_buffer[queryIdx * params.knn+1] = ray_origin.x;
    //params.frame_buffer[queryIdx * params.knn+2] = ray_origin.y;
    //params.frame_buffer[queryIdx * params.knn+3] = ray_origin.z;
}
