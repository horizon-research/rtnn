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

#include "optixDynamicMaterials.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

extern "C" __constant__ Params params;


static __forceinline__ __device__ void trace( OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction, float tmin, float tmax, float3* prd )
{
    unsigned int p0, p1, p2;
    p0 = float_as_int( prd->x );
    p1 = float_as_int( prd->y );
    p2 = float_as_int( prd->z );
    optixTrace( handle, ray_origin, ray_direction, tmin, tmax,
                0.0f,  // rayTime
                OptixVisibilityMask( 1 ), OPTIX_RAY_FLAG_NONE,
                0,     // SBT offset
                0,     // SBT stride
                0,     // missSBTIndex
                p0, p1, p2 );
    prd->x = int_as_float( p0 );
    prd->y = int_as_float( p1 );
    prd->z = int_as_float( p2 );
}


static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3( int_as_float( optixGetPayload_0() ),
                        int_as_float( optixGetPayload_1() ),
                        int_as_float( optixGetPayload_2() ) );
}


static __forceinline__ __device__ float3 getShadingNormal()
{
    return make_float3( int_as_float( optixGetAttribute_0() ),
                        int_as_float( optixGetAttribute_1() ),
                        int_as_float( optixGetAttribute_2() ) );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const float3 U = params.camera_u;
    const float3 V = params.camera_v;
    const float3 W = params.camera_w;
    const float2 d = 2.0f * make_float2( static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
                                         static_cast<float>( idx.y ) / static_cast<float>( dim.y ) ) - 1.0f;

    const float3 origin      = params.cam_eye;
    const float3 direction   = normalize( d.x * U + d.y * V + W );
    float3       payload_rgb = make_float3( 0.5f, 0.5f, 0.5f );
    trace( params.handle, origin, direction,
           0.00f,  // tmin
           1e16f,  // tmax
           &payload_rgb );

    params.image[idx.y * params.image_width + idx.x] = make_color( payload_rgb );
}


extern "C" __global__ void __miss__ms()
{
    MissData* missData = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3    payload  = getPayload();
    setPayload( missData->color );
}


extern "C" __global__ void __intersection__is()
{
    HitGroupData* hgData = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const float3  orig   = optixGetObjectRayOrigin();
    const float3  dir    = optixGetObjectRayDirection();

    const float3 center = {0.f, 0.f, 0.f};
    const float3 O      = orig - center;
    const float  l      = 1 / length( dir );
    const float3 D      = dir * l;

    const float b    = dot( O, D );
    const float c    = dot( O, O ) - params.radius * params.radius;
    const float disc = b * b - c;
    if( disc > 0.0f )
    {
        const float sdisc = sqrtf( disc );
        const float root1 = ( -b - sdisc );

        const float  root11         = 0.0f;
        const float3 shading_normal = ( O + ( root1 + root11 ) * D ) / params.radius;
        unsigned int p0, p1, p2;
        p0 = float_as_int( shading_normal.x );
        p1 = float_as_int( shading_normal.y );
        p2 = float_as_int( shading_normal.z );

        optixReportIntersection( root1,  // t hit
                                 0,      // user hit kind
                                 p0, p1, p2 );
    }
}


__forceinline__ __device__ void closesthit_impl( float3 baseColor )
{
    float3 normals = normalize( optixTransformNormalFromObjectToWorldSpace( getShadingNormal() ) ) * 0.5f + 0.5f;

    // add material effects
    normals *= baseColor;
    setPayload( normals );
}

extern "C" __global__ void __closesthit__ch()
{
    HitGroupData* hgData = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    closesthit_impl( hgData->color );
}

extern "C" __global__ void __closesthit__normal()
{
    float3 normals = normalize( optixTransformNormalFromObjectToWorldSpace( getShadingNormal() ) ) * 0.5f + 0.5f;
    setPayload( normals );
}

extern "C" __global__ void __closesthit__red()
{
    closesthit_impl( make_float3( 1.f, 0.f, 0.f ) );
}

extern "C" __global__ void __closesthit__green()
{
    closesthit_impl( make_float3( 0.f, 1.f, 0.f ) );
}

extern "C" __global__ void __closesthit__blue()
{
    closesthit_impl( make_float3( 0.f, 0.f, 1.f ) );
}
