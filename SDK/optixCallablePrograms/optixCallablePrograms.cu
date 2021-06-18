//
// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda/whitted_cuda.h>

#include "optixCallablePrograms.h"

// Direct callables for shading
extern "C" __device__ float3 __direct_callable__phong_shade( float3 hit_point, float3 ray_dir, float3 normal )
{
    float3 Ka        = {0.2f, 0.5f, 0.5f};
    float3 Kd        = {0.2f, 0.7f, 0.8f};
    float3 Ks        = {0.9f, 0.9f, 0.9f};
    float  phong_exp = 64.0f;

    float3 result = make_float3( 0.0f );

    for( int i = 0; i < whitted::params.lights.count; ++i )
    {
        Light light = whitted::params.lights[i];
        if( light.type == Light::Type::POINT )
        {
            // compute direct lighting
            float  Ldist = length( light.point.position - hit_point );
            float3 L     = normalize( light.point.position - hit_point );
            float  nDl   = dot( normal, L );

            result += Kd * nDl * light.point.color;

            float3 H   = normalize( L - ray_dir );
            float  nDh = dot( normal, H );
            if( nDh > 0 )
            {
                float power = pow( nDh, phong_exp );
                result += Ks * power * light.point.color;
            }
        }
        else if( light.type == Light::Type::AMBIENT )
        {
            // ambient contribution
            result += Ka * light.ambient.color;
        }
    }

    return result;
}

extern "C" __device__ float3 __direct_callable__checkered_shade( float3 hit_point, float3 ray_dir, float3 normal )
{
    float3 result;

    float value = dot( normal, ray_dir );
    if( value < 0 )
    {
        value *= -1;
    }

    float3         sphere_normal = normalize( hit_point );
    float          a             = acos( sphere_normal.y );
    float          b             = atan2( sphere_normal.x, sphere_normal.z ) + M_PIf;
    Light::Ambient light         = whitted::params.lights[0].ambient;
    if( ( fmod( a, M_PIf / 8 ) < M_PIf / 16 ) ^ ( fmod( b, M_PIf / 4 ) < M_PIf / 8 ) )
    {
        result = light.color + ( value * make_float3( 0.0f ) );
    }
    else
    {
        result = light.color + ( value * make_float3( 1.0f ) );
    }

    return clamp( result, 0.0f, 1.0f );
}

extern "C" __device__ float3 __direct_callable__normal_shade( float3 hit_point, float3 ray_dir, float3 normal )
{
    return normalize( normal ) * 0.5f + 0.5f;
}

// Closest hit
extern "C" __global__ void __closesthit__radiance()
{
    const HitGroupData* hitgroup_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    const float3 ray_orig  = optixGetWorldRayOrigin();
    const float3 ray_dir   = optixGetWorldRayDirection();
    const float  ray_t     = optixGetRayTmax();
    float3       hit_point = ray_orig + ray_t * ray_dir;

    float3 object_normal = make_float3( int_as_float( optixGetAttribute_0() ), int_as_float( optixGetAttribute_1() ),
                                        int_as_float( optixGetAttribute_2() ) );
    float3 world_normal  = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
    float3 ffnormal      = faceforward( world_normal, -ray_dir, world_normal );

    // Use a direct callable to set the result
    float3 result = optixDirectCall<float3, float3, float3, float3>( hitgroup_data->dc_index, hit_point, ray_dir, ffnormal );
    whitted::setPayloadResult( result );
}

// Continuation callable for background
extern "C" __device__ float3 __continuation_callable__raydir_shade( float3 ray_dir )
{
    return normalize( ray_dir ) * 0.5f + 0.5f;
}

// Miss
extern "C" __global__ void __miss__raydir_shade()
{
    const float3 ray_dir = optixGetWorldRayDirection();

    float3 result = optixContinuationCall<float3, float3>( 0, ray_dir );
    whitted::setPayloadResult( result );
}
