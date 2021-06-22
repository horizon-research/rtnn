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
#include "optixWhitted.h"
#include "random.h"
#include "helpers.h"
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__pinhole_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    //const uint3 dim = optixGetLaunchDimensions();

    const GeomData* geom = (GeomData*) optixGetSbtDataPointer();
    //const CameraData* camera = (CameraData*) optixGetSbtDataPointer();

    //const unsigned int image_index = params.width * idx.y + idx.x;
    //unsigned int image_index = params.width * params.numPrims * idx.y + idx.x * params.numPrims;
    unsigned int rayIdx = idx.x;

    // calculate d by transforming <0, 0> from the top-left corner to the center of the image
    //float2 d = make_float2(idx.x, idx.y) / make_float2(params.width, params.height) * 2.f - 1.f;
    //float3 ray_origin = camera->eye;
    //float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);

    float3 ray_origin = geom->spheres[rayIdx];
    float3 ray_direction = normalize(make_float3(1, 1, 1));

    unsigned int id = 0;

    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        params.scene_epsilon,
        //1e16f,
        1e-2f,
        0.0f,
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
        reinterpret_cast<unsigned int&>(rayIdx),
        reinterpret_cast<unsigned int&>(id)
    );
}
