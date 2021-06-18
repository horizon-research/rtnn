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

#include <cuda_runtime.h>
#include <optixDemandTexture.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void trace( OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction, float tmin, float tmax, float3* prd )
{
    unsigned int p0, p1, p2;
    p0 = float_as_int( prd->x );
    p1 = float_as_int( prd->y );
    p2 = float_as_int( prd->z );
    optixTrace( handle, ray_origin, ray_direction, tmin, tmax,
                0.0f,  // rayTime
                OptixVisibilityMask( 1 ), OPTIX_RAY_FLAG_NONE,
                0,  // SBT offset
                1,  // SBT stride
                0,  // missSBTIndex
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
    return make_float3( int_as_float( optixGetPayload_0() ), int_as_float( optixGetPayload_1() ),
                        int_as_float( optixGetPayload_2() ) );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    const float3      U      = rtData->camera_u;
    const float3      V      = rtData->camera_v;
    const float3      W      = rtData->camera_w;
    const float2      d      = 2.0f * make_float2( static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
                                         static_cast<float>( idx.y ) / static_cast<float>( dim.y ) )
                     - 1.0f;

    const float3 origin      = rtData->cam_eye;
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
    MissData* rt_data = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3    payload = getPayload();
    setPayload( make_float3( rt_data->r, rt_data->g, rt_data->b ) );
}

// Convert Cartesian coordinates to polar coordinates
__forceinline__ __device__ float3 cartesian_to_polar( const float3& v )
{
    float azimuth;
    float elevation;
    float radius = length( v );

    float r = sqrtf( v.x * v.x + v.y * v.y );
    if( r > 0.0f )
    {
        azimuth   = atanf( v.y / v.x );
        elevation = atanf( v.z / r );

        if( v.x < 0.0f )
            azimuth += M_PIf;
        else if( v.y < 0.0f )
            azimuth += M_PIf * 2.0f;
    }
    else
    {
        azimuth = 0.0f;

        if( v.z > 0.0f )
            elevation = +M_PI_2f;
        else
            elevation = -M_PI_2f;
    }

    return make_float3( azimuth, elevation, radius );
}

extern "C" __global__ void __intersection__is()
{
    HitGroupData* hg_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const Sphere  sphere  = hg_data->sphere;
    const float3  orig    = optixGetObjectRayOrigin();
    const float3  dir     = optixGetObjectRayDirection();

    const float3 O = orig - sphere.center;
    const float  l = 1 / length( dir );
    const float3 D = dir * l;

    const float b    = dot( O, D );
    const float c    = dot( O, O ) - sphere.radius * sphere.radius;
    const float disc = b * b - c;
    if( disc > 0.0f )
    {
        const float sdisc = sqrtf( disc );
        const float root1 = ( -b - sdisc );

        const float  root11         = 0.0f;
        const float3 shading_normal = ( O + ( root1 + root11 ) * D ) / sphere.radius;

        float3 polar    = cartesian_to_polar( shading_normal );
        float3 texcoord = make_float3( polar.x * 0.5f * M_1_PIf, ( polar.y + M_PI_2f ) * M_1_PIf, polar.z / sphere.radius );

        unsigned int p0, p1, p2;
        p0 = float_as_int( texcoord.x );
        p1 = float_as_int( texcoord.y );
        p2 = float_as_int( texcoord.z );

        optixReportIntersection( root1,  // t hit
                                 0,      // user hit kind
                                 p0, p1, p2 );
    }
}

// Check whether a specified miplevel of a demand-loaded texture is resident, recording a request if not.
inline __device__ void requestMipLevel( unsigned int textureId, const DemandTextureSampler& sampler, unsigned int mipLevel, bool& isResident )
{
    // A page id consists of the texture id (upper 28 bits) and the miplevel number (lower 4 bits).
    const unsigned int requestedPage = textureId << 4 | mipLevel;

    // The paging context was provided as a launch parameter.
    const OptixPagingContext& context = params.pagingContext;

    // Check whether the requested page is resident, recording a request if not.
    optixPagingMapOrRequest( context.usageBits, context.residenceBits, context.pageTable, requestedPage, &isResident );
}

// Fetch from a demand-loaded texture with a specified LOD.  The necessary miplevels are requested
// if they are not resident, which is indicated by the boolean result parameter.
inline __device__ float4
tex2DLodLoadOrRequest( unsigned int textureId, const DemandTextureSampler& sampler, float x, float y, float lod, bool& isResident )
{
    // Request the closest miplevels.  We conservatively clamp the miplevel to [0,15].  It is
    // subsequently clamped to the actual number of miplevels when processing the request.
    const unsigned int maxMipLevel = 15;  // limited to 4 bits.
    const unsigned int lowerLevel  = clamp( static_cast<unsigned int>( lod ), 0U, maxMipLevel );
    const unsigned int upperLevel  = clamp( static_cast<unsigned int>( ceilf( lod ) ), 0U, maxMipLevel );
    requestMipLevel( textureId, sampler, lowerLevel, isResident );
    if( upperLevel != lowerLevel )
    {
        bool isResident2;
        requestMipLevel( textureId, sampler, upperLevel, isResident2 );
        isResident = isResident && isResident2;
    }
    if( isResident )
        return tex2DLod<float4>( sampler.texture, x, y, lod );
    else
        return make_float4( 1.f, 0.f, 1.f, 0.f );
}

extern "C" __global__ void __closesthit__ch()
{
    // The demand-loaded texture id is provided in the hit group data.  It's used as an index into
    // the sampler array, which is a launch parameter.
    HitGroupData*               hg_data   = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    unsigned int                textureId = hg_data->demand_texture_id;
    const DemandTextureSampler& sampler   = params.demandTextures[textureId];

    // The texture coordinates calculated by the intersection shader are provided as attributes.
    const float3 texcoord = make_float3( int_as_float( optixGetAttribute_0() ), int_as_float( optixGetAttribute_1() ),
                                         int_as_float( optixGetAttribute_2() ) );

    // Fetch from the demand-loaded texture, requesting any non-resident miplevels as neeeded.
    float scale = hg_data->texture_scale;
    float lod   = hg_data->texture_lod;
    bool  isResident;
    float4 pixel = tex2DLodLoadOrRequest( textureId, sampler, texcoord.x * scale, 1.f - texcoord.y * scale, lod, isResident );
    setPayload( make_float3( pixel ) );
}
