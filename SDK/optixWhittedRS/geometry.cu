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

#include <optix.h>

#include "optixWhitted.h"
#include "random.h"
#include "helpers.h"

extern "C" {
__constant__ Params params;
}

extern "C" __device__ void intersect_sphere()
{
    // This is called when a ray-bbox intersection is found. We still need to
    // perform the ray-sphere intersection test ourselves, basically using the
    // classic algorithm here:
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection

    const HitGroupData &sbt_data = *(const HitGroupData*) optixGetSbtDataPointer();
    const Sphere sphere = sbt_data.geometry.sphere;

    const float3  ray_orig = optixGetWorldRayOrigin();
    const float3  ray_dir  = optixGetWorldRayDirection();
    const float   ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    float3 O = ray_orig - sphere.center;
    float  l = 1 / length(ray_dir);
    float3 D = ray_dir * l;
    float radius = sphere.radius;

    // this is O projected onto D, which will be the tangential line length if
    // the ray just touches the sphere.
    float b = dot(O, D);
    // dot(O, O) gets us the square of the ray-center distance
    float c = dot(O, O)-radius*radius;
    float disc = b*b-c;
    //params.frame_buffer[optixGetPayload_0() * params.numPrims + optixGetPrimitiveIndex()] = b;

    // disc > 0 means b^2 + radius^2 > dot(O, O)^2, which mean we have an intersection
    if(disc > 0.0f)
    {
	// record the intersected primitive (might not be the closest and can
	// be overwritten later). we won't report the intersection so the miss
	// program will be called (although that's empty; if not, then the
	// value set here will be overwritten there) and the closest hit
	// program won't be called.

        bool isApprox = true;

        unsigned int rayIdx = optixGetPayload_0();
        //unsigned int id = optixGetPayload_1();
        unsigned int primIdx = optixGetPrimitiveIndex();
        if (isApprox)
        {
          // approximate search here.
          //params.frame_buffer[rayIdx * params.numPrims + id] = optixGetPrimitiveIndex() + 1;
          params.frame_buffer[rayIdx * params.numPrims + primIdx] = 1;
	  // each ray's traversal is sequential; payload set here will be used in
	  // the next intersection.
        }
        else
        {
          float sdisc = sqrtf(disc);
          float root1 = (-b - sdisc);
          float root2 = (-b + sdisc);

          float t0, t1;
          t0 = root1 * l;
          t1 = root2 * l;

          if ((t0 > 0 && t1 < 0) || (t0 < 0 && t1 > 0)) {
            //params.frame_buffer[rayIdx * params.numPrims + id] = optixGetPrimitiveIndex() + 1;
            params.frame_buffer[rayIdx * params.numPrims + primIdx] = 1;
          }
        }
        //optixSetPayload_1( id+1 );
    }
}

extern "C" __global__ void __intersection__sphere()
{
    unsigned int rayIdx = optixGetPayload_0();
    unsigned int primIdx = optixGetPrimitiveIndex();
    params.frame_buffer[rayIdx * params.numPrims + primIdx] = rayIdx;
    //intersect_sphere();
}

