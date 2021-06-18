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

#include <cuda/GeometryData.h>
#include <cuda/LocalGeometry.h>
#include <cuda/curve.h>
#include <cuda/helpers.h>
#include <cuda/whitted_cuda.h>
#include <sutil/vec_math.h>


// Get curve hit-point in world coordinates.
static __forceinline__ __device__ float3 getHitPoint()
{
    const float  t            = optixGetRayTmax();
    const float3 rayOrigin    = optixGetWorldRayOrigin();
    const float3 rayDirection = optixGetWorldRayDirection();

    return rayOrigin + t * rayDirection;
}

// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ float3 normalLinear( const int primitiveIndex )
{
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[2];

    optixGetLinearCurveVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    LinearBSplineSegment interpolator( controlPoints );
    float3               hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
    const float3 normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
    return optixTransformNormalFromObjectToWorldSpace( normal );
}

// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ float3 normalQuadratic( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[3];

    optixGetQuadraticBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    QuadraticBSplineSegment interpolator( controlPoints );
    float3                  hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
    const float3 normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
    return optixTransformNormalFromObjectToWorldSpace( normal );
}

// Compute surface normal of cubic pimitive in world space.
static __forceinline__ __device__ float3 normalCubic( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCubicBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicBSplineSegment interpolator( controlPoints );
    float3              hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
    const float3 normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
    return optixTransformNormalFromObjectToWorldSpace( normal );
}

static __forceinline__ __device__ float3 shade( const whitted::HitGroupData* hitGroupData, const float3 hitPoint, const float3 normal, const float3 base_color )
{
    //
    // Retrieve material data
    //
    float metallic  = hitGroupData->material_data.pbr.metallic;
    float roughness = hitGroupData->material_data.pbr.roughness;

    //
    // Convert to material params
    //
    const float F0         = 0.04f;
    const float3      diff_color = base_color * ( 1.0f - F0 ) * ( 1.0f - metallic );
    const float3      spec_color = lerp( make_float3( F0 ), base_color, metallic );
    const float       alpha      = roughness * roughness;

    float3 result = make_float3( 0.0f );

    for( int i = 0; i < whitted::params.lights.count; ++i )
    {
        Light light = whitted::params.lights[i];

        if( light.type == Light::Type::POINT )
        {
            const float  L_dist  = length( light.point.position - hitPoint );
            const float3 L       = ( light.point.position - hitPoint ) / L_dist;
            const float3 V       = -normalize( optixGetWorldRayDirection() );
            const float3 H       = normalize( L + V );
            const float  N_dot_L = dot( normal, L );
            const float  N_dot_V = dot( normal, V );
            const float  N_dot_H = dot( normal, H );
            const float  V_dot_H = dot( V, H );

            if( N_dot_L > 0.0f && N_dot_V > 0.0f )
            {
                const float tmin     = 0.001f;           // TODO
                const float tmax     = L_dist - 0.001f;  // TODO
                const bool  occluded = whitted::traceOcclusion( whitted::params.handle, hitPoint, L, tmin, tmax );
                if( !occluded )
                {
                    const float3 F     = whitted::schlick( spec_color, V_dot_H );
                    const float  G_vis = whitted::vis( N_dot_L, N_dot_V, alpha );
                    const float  D     = whitted::ggxNormal( N_dot_H, alpha );

                    const float3 diff = ( 1.0f - F ) * diff_color / M_PIf;
                    const float3 spec = F * G_vis * D;

                    result += light.point.color * light.point.intensity * N_dot_L * ( diff + spec );
                }
            }
        }
    }

    return result;
}

// Get u-parameter for full strand.
//
// Parameters:
//    geo - the GeometricData from the SBT.
//    primitiveIndex - the primitive index
//
static __forceinline__ __device__ float getStrandU( const GeometryData& geo, const int primitiveIndex )
{
    float  segmentU   = optixGetCurveParameter();
    float2 strandInfo = geo.curves.strand_u[primitiveIndex];
    // strandInfo.x ~ strand u at segment start
    // strandInfo.y ~ scale factor (i.e. 1/numberOfSegments)
    return strandInfo.x + segmentU * strandInfo.y;
}

// Compute normal
//
static __forceinline__ __device__ float3 computeNormal( OptixPrimitiveType type, const int primitiveIndex )
{
  switch( type ) {
  case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
        return  normalLinear( primitiveIndex );
    break;
  case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
        return normalQuadratic( primitiveIndex );
    break;
  case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
        return  normalCubic( primitiveIndex );
    break;
  }
  return make_float3(0.0f);
}

extern "C" __global__ void __closesthit__curve_strand_u()
{
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();

    const whitted::HitGroupData* hitGroupData = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const GeometryData&          geometryData = reinterpret_cast<const GeometryData&>( hitGroupData->geometry_data );

    const float3 normal     = computeNormal( optixGetPrimitiveType(), primitiveIndex );
    const float3 colors[2]  = {make_float3( 1, 0, 0 ), make_float3( 0, 1, 0 )};
    const float  u          = getStrandU( geometryData, primitiveIndex );
    const float3 base_color = colors[0] * u + colors[1] * ( 1 - u );

    const float3 hitPoint = getHitPoint();
    const float3 result   = shade( hitGroupData, hitPoint, normal, base_color );

    whitted::setPayloadResult( result );
}

extern "C" __global__ void __closesthit__curve_segment_u()
{
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();

    const whitted::HitGroupData* hitGroupData = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );

    const float3 normal     = computeNormal( optixGetPrimitiveType(), primitiveIndex );
    const float3 colors[3]  = {make_float3( 1, 0, 0 ), make_float3( 0, 1, 0 ),
                               make_float3( 0, 0, 1 ) };
    const float  u          = optixGetCurveParameter();
    float3 base_color;
    if( u == 0.0f || u == 1.0f )  // on end-cap
        base_color = colors[2];
    else
        base_color = colors[0] * u + colors[1] * ( 1 - u );

    const float3 hitPoint = getHitPoint();
    const float3 result   = shade( hitGroupData, hitPoint, normal, base_color );

    whitted::setPayloadResult( result );
}

extern "C" __global__ void __closesthit__curve_strand_idx()
{
    unsigned int primitiveIndex = optixGetPrimitiveIndex();

    const whitted::HitGroupData* hitGroupData = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const GeometryData&          geometryData = reinterpret_cast<const GeometryData&>( hitGroupData->geometry_data );

    float3       normal      = computeNormal( optixGetPrimitiveType(), primitiveIndex );
    float3       colors[6]   = {make_float3( 1, 0, 0 ), make_float3( 0, 1, 0 ), make_float3( 0, 0, 1 ),
				make_float3( 1, 1, 0 ), make_float3( 1, 0, 1 ), make_float3( 0, 1, 1 )};
    unsigned int strandIndex = geometryData.curves.strand_i[primitiveIndex];
    uint2        strandInfo  = geometryData.curves.strand_info[strandIndex];
    float        u           = ( primitiveIndex - strandInfo.x ) / (float)strandInfo.y;
    float3       base_color  = colors[0] * u + colors[strandIndex % 5 + 1] * ( 1.0f - u );

    float3 hitPoint = getHitPoint();
    float3 result   = shade( hitGroupData, hitPoint, normal, base_color );

    whitted::setPayloadResult( result );
}
