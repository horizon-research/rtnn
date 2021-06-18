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

#define OPTIX_COMPATIBILITY 7
#include <optix.h>

#include "optixNVLink.h"
#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include <cuda/random.h>

extern "C" {
__constant__ Params params;
}


//------------------------------------------------------------------------------
//
// Per ray data, and getting at it
//
//------------------------------------------------------------------------------

// Per-ray data for radiance rays
struct RadiancePRD
{
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    float3       origin;
    float3       direction;
    unsigned int seed;
    int          countEmitted;
    int          done;
    int          pad;
};

static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
}


// Per-ray data for occlusion rays
static __forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<unsigned int>( occluded ) );
}


//------------------------------------------------------------------------------
//
// Sampling and color
//
//------------------------------------------------------------------------------

struct Onb
{
  __forceinline__ __device__ Onb(const float3& normal)
  {
    m_normal = normal;

    if( fabs(m_normal.x) > fabs(m_normal.z) )
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y =  m_normal.x;
      m_binormal.z =  0;
    }
    else
    {
      m_binormal.x =  0;
      m_binormal.y = -m_normal.z;
      m_binormal.z =  m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross( m_binormal, m_normal );
  }

  __forceinline__ __device__ void inverse_transform(float3& p) const
  {
    p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  p.x = r * cosf( phi );
  p.y = r * sinf( phi );

  // Project up to hemisphere.
  p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}


__forceinline__ __device__ float3 deviceColor( unsigned int idx )
{
    return make_float3(
            idx == 0 ? 0.05f : 0.0f,
            idx == 1 ? 0.05f : 0.0f,
            idx == 2 ? 0.05f : 0.0f
            );
}


//------------------------------------------------------------------------------
//
// Tracing rays
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        )
{
    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1 );
}


static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    unsigned int occluded = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION,      // missSBTIndex
            occluded );
    return occluded;
}


//------------------------------------------------------------------------------
//
// Optix Programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w          = params.width;
    const int    h          = params.height;
    const uint3  launch_idx = optixGetLaunchIndex();
    const int2   pixel_idx  = params.sample_index_buffer[ launch_idx.x ];

    // Work distribution might assign tiles that cross over image boundary
    if( pixel_idx.x > w-1 || pixel_idx.y > h-1 )
        return;

    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;

    const int    subframe_index = params.subframe_index;

    unsigned int seed = tea<4>( pixel_idx.y*w + pixel_idx.x, subframe_index );


    float3 result = make_float3( 0.0f );
    int i = params.samples_per_launch;
    do
    {
        const float2 subpixel_jitter = make_float2( rnd( seed )-0.5f, rnd( seed )-0.5f );

        const float2 d = 2.0f * make_float2(
                ( static_cast<float>( pixel_idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
                ( static_cast<float>( pixel_idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
                ) - 1.0f;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);
        float3 ray_origin    = eye;

        RadiancePRD prd;
        prd.emitted      = make_float3(0.f);
        prd.radiance     = make_float3(0.f);
        prd.attenuation  = make_float3(1.f);
        prd.countEmitted = true;
        prd.done         = false;
        prd.seed         = seed;

        int depth = 0;
        for( ;; )
        {
            traceRadiance(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    0.01f,  // tmin  
                    1e16f,  // tmax
                    &prd );

            result += prd.emitted;
            result += prd.radiance * prd.attenuation;

            if( prd.done  || depth >= 3 ) 
                break;

            ray_origin    = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }
    }
    while( --i );

    float3 accum_color = result / static_cast<float>( params.samples_per_launch );
    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.sample_accum_buffer[ launch_idx.x ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }
    params.sample_accum_buffer [ launch_idx.x ] = make_float4( accum_color, 1.0f);

    const unsigned int image_index  = pixel_idx.y * params.width + pixel_idx.x;

    float3 device_color = deviceColor( params.device_idx ) * params.device_color_scale;
    params.result_buffer[ image_index ] = make_color ( accum_color + device_color );
}


extern "C" __global__ void __miss__radiance()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD* prd = getPRD();

    prd->radiance = make_float3( rt_data->r, rt_data->g, rt_data->b );
    prd->done     = true;
}


extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}


extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    RadiancePRD* prd = getPRD();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    // Compute normal and hit point
    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const float3 N    = faceforward( N_0, -ray_dir, N_0 );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    // Account for emission
    if( prd->countEmitted )
        prd->emitted = rt_data->emission_color;
    else
        prd->emitted = make_float3( 0.0f );

    // Compute attenuation (diffuse color) using texture if available
    cudaTextureObject_t texture = rt_data->diffuse_texture;
    if (texture != 0)
    {
        // get barycentric coordinates
        const float2 barycentrics = optixGetTriangleBarycentrics();
        const float b1 = barycentrics.x;
        const float b2 = barycentrics.y;
        const float b0 = 1.0f - (b1 + b2);
        
        // compute texture coordinates
        const int vindex = optixGetPrimitiveIndex() * 3;

        const float2 t0 = rt_data->tex_coords[ vindex+0 ];
        const float2 t1 = rt_data->tex_coords[ vindex+1 ];
        const float2 t2 = rt_data->tex_coords[ vindex+2 ];

        float2 tex_coord = b0*t0 + b1*t1 + b2*t2;
        float s = tex_coord.x;
        float t = tex_coord.y;

        // sample texture
        float4 tex_val = tex2D<float4>( rt_data->diffuse_texture, s, t );
        prd->attenuation *= make_float3( tex_val );
    }
    else
    {
        prd->attenuation *= rt_data->diffuse_color;
    }
    
    unsigned int seed = prd->seed; 

    // Sample a hemisphere direction and place in per-ray data
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_sample_hemisphere( z1, z2, w_in );
        Onb onb( N );
        onb.inverse_transform( w_in );
        prd->direction = w_in;
        prd->origin    = P;

        prd->countEmitted = false;
    }

    // Sample a position on the light source
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd->seed = seed;

    ParallelogramLight light = params.light;
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P );
    const float3 L     = normalize(light_pos - P );
    const float  nDl   = dot( N, L );
    const float  LnDl  = -dot( light.normal, L );

    // Cast the shadow ray
    float weight = 0.0f;
    if( nDl > 0.0f && LnDl > 0.0f )
    {
        const bool occluded = traceOcclusion(
            params.handle,
            P,
            L,
            0.01f,         // tmin
            Ldist - 0.01f  // tmax
            );

        if( !occluded )
        {
            const float A = length(cross(light.v1, light.v2));
            weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }

    prd->radiance += light.emission * weight;
}


