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
#include "helpers.h"

extern "C" {
__constant__ Params params;
}

//static __device__ __inline__ RadiancePRD getRadiancePRD()
//{
//    RadiancePRD prd;
//    prd.result.x = int_as_float( optixGetPayload_0() );
//    prd.result.y = int_as_float( optixGetPayload_1() );
//    prd.result.z = int_as_float( optixGetPayload_2() );
//    prd.importance = int_as_float( optixGetPayload_3() );
//    prd.depth = optixGetPayload_4();
//    return prd;
//}
//
//static __device__ __inline__ void setRadiancePRD( const RadiancePRD &prd )
//{
//    optixSetPayload_0( float_as_int(prd.result.x) );
//    optixSetPayload_1( float_as_int(prd.result.y) );
//    optixSetPayload_2( float_as_int(prd.result.z) );
//    optixSetPayload_3( float_as_int(prd.importance) );
//    optixSetPayload_4( prd.depth );
//}
//
//static __device__ __inline__ OcclusionPRD getOcclusionPRD()
//{
//    OcclusionPRD prd;
//    prd.attenuation.x = int_as_float( optixGetPayload_0() );
//    prd.attenuation.y = int_as_float( optixGetPayload_1() );
//    prd.attenuation.z = int_as_float( optixGetPayload_2() );
//    return prd;
//}
//
//static __device__ __inline__ void setOcclusionPRD( const OcclusionPRD &prd )
//{
//    optixSetPayload_0( float_as_int(prd.attenuation.x) );
//    optixSetPayload_1( float_as_int(prd.attenuation.y) );
//    optixSetPayload_2( float_as_int(prd.attenuation.z) );
//}
//
//static __device__ __inline__ float3
//traceRadianceRay(
//    float3 origin,
//    float3 direction,
//    int depth,
//    float importance)
//{
//    RadiancePRD prd;
//    prd.depth = depth;
//    prd.importance = importance;
//
//    optixTrace(
//        params.handle,
//        origin,
//        direction,
//        params.scene_epsilon,
//        1e16f,
//        0.0f,
//        OptixVisibilityMask( 1 ),
//        OPTIX_RAY_FLAG_NONE,
//        RAY_TYPE_RADIANCE,
//        RAY_TYPE_COUNT,
//        RAY_TYPE_RADIANCE,
//        float3_as_args(prd.result),
//        /* Can't use float_as_int() because it returns rvalue but payload requires a lvalue */
//        reinterpret_cast<unsigned int&>(prd.importance),
//        reinterpret_cast<unsigned int&>(prd.depth) );
//
//    return prd.result;
//}
//
//static
//__device__ void phongShadowed()
//{
//    // this material is opaque, so it fully attenuates all shadow rays
//    OcclusionPRD prd;
//    prd.attenuation = make_float3(0.f);
//    setOcclusionPRD(prd);
//}
//
//static
//__device__ void phongShade( float3 p_Kd,
//                            float3 p_Ka,
//                            float3 p_Ks,
//                            float3 p_Kr,
//                            float  p_phong_exp,
//                            float3 p_normal )
//{
//    const float3 ray_orig = optixGetWorldRayOrigin();
//    const float3 ray_dir  = optixGetWorldRayDirection();
//    const float  ray_t    = optixGetRayTmax();
//
//    RadiancePRD prd = getRadiancePRD();
//
//    float3 hit_point = ray_orig + ray_t * ray_dir;
//
//    // ambient contribution
//    float3 result = p_Ka * params.ambient_light_color;
//
//    // compute direct lighting
//    BasicLight light = params.light;
//    float Ldist = length(light.pos - hit_point);
//    float3 L = normalize(light.pos - hit_point);
//    float nDl = dot( p_normal, L);
//
//    // cast shadow ray
//    float3 light_attenuation = make_float3(static_cast<float>( nDl > 0.0f ));
//    if ( nDl > 0.0f )
//    {
//        OcclusionPRD shadow_prd;
//        shadow_prd.attenuation = make_float3(1.0f);
//
//        optixTrace(
//            params.handle,
//            hit_point,
//            L,
//            0.01f,
//            Ldist,
//            0.0f,
//            OptixVisibilityMask( 1 ),
//            OPTIX_RAY_FLAG_NONE,
//            RAY_TYPE_OCCLUSION,
//            RAY_TYPE_COUNT,
//            RAY_TYPE_OCCLUSION,
//            float3_as_args(shadow_prd.attenuation) );
//
//        light_attenuation = shadow_prd.attenuation;
//    }
//
//    // If not completely shadowed, light the hit point
//    if( fmaxf(light_attenuation) > 0.0f )
//    {
//        float3 Lc = light.color * light_attenuation;
//
//        result += p_Kd * nDl * Lc;
//
//        float3 H = normalize(L - ray_dir);
//        float nDh = dot( p_normal, H );
//        if(nDh > 0)
//        {
//            float power = pow(nDh, p_phong_exp);
//            result += p_Ks * power * Lc;
//        }
//    }
//
//    if( fmaxf( p_Kr ) > 0 )
//    {
//
//        // ray tree attenuation
//        float new_importance = prd.importance * luminance( p_Kr );
//        int new_depth = prd.depth + 1;
//
//        // reflection ray
//        // compare new_depth to max_depth - 1 to leave room for a potential shadow ray trace
//        if( new_importance >= 0.01f && new_depth <= params.max_depth - 1)
//        {
//            float3 R = reflect( ray_dir, p_normal );
//
//            result += p_Kr * traceRadianceRay(
//                hit_point,
//                R,
//                new_depth,
//                new_importance);
//        }
//    }
//
//    // pass the color back
//    prd.result = result;
//    setRadiancePRD(prd);
//}
//
//extern "C" __global__ void __closesthit__checker_radiance()
//{
//    const HitGroupData* sbt_data = (HitGroupData*) optixGetSbtDataPointer();
//    const CheckerPhong &checker = sbt_data->shading.checker;
//
//    float3 Kd, Ka, Ks, Kr;
//    float  phong_exp;
//
//    float2 texcoord = make_float2(
//        int_as_float( optixGetAttribute_3() ),
//        int_as_float( optixGetAttribute_4() ) );
//    float2 t  = texcoord * checker.inv_checker_size;
//    t.x = floorf(t.x);
//    t.y = floorf(t.y);
//
//    int which_check = ( static_cast<int>( t.x ) +
//                        static_cast<int>( t.y ) ) & 1;
//
//    if ( which_check )
//    {
//        Kd = checker.Kd1;
//        Ka = checker.Ka1;
//        Ks = checker.Ks1;
//        Kr = checker.Kr1;
//        phong_exp = checker.phong_exp1;
//    } else
//    {
//        Kd = checker.Kd2;
//        Ka = checker.Ka2;
//        Ks = checker.Ks2;
//        Kr = checker.Kr2;
//        phong_exp = checker.phong_exp2;
//    }
//
//    float3 object_normal = make_float3(
//        int_as_float( optixGetAttribute_0() ),
//        int_as_float( optixGetAttribute_1() ),
//        int_as_float( optixGetAttribute_2() ));
//    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace(object_normal) );
//    float3 ffnormal  = faceforward( world_normal, -optixGetWorldRayDirection(), world_normal );
//    phongShade( Kd, Ka, Ks, Kr, phong_exp, ffnormal );
//}

extern "C" __global__ void __closesthit__metal_radiance()
{
    //const HitGroupData* sbt_data = (HitGroupData*) optixGetSbtDataPointer();
    //const Phong &phong = sbt_data->shading.metal;

    //float3 object_normal = make_float3(
    //    int_as_float( optixGetAttribute_0() ),
    //    int_as_float( optixGetAttribute_1() ),
    //    int_as_float( optixGetAttribute_2() ));

    //float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
    //float3 ffnormal = faceforward( world_normal, -optixGetWorldRayDirection(), world_normal );
    //phongShade( phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, ffnormal );

    //const float3 shading_normal =
    //    make_float3(
    //            int_as_float( optixGetAttribute_0() ),
    //            int_as_float( optixGetAttribute_1() ),
    //            int_as_float( optixGetAttribute_2() )
    //            );
    //const float3 color = normalize( optixTransformNormalFromObjectToWorldSpace( shading_normal ) ) * 0.5f + 0.5f;
    //optixSetPayload_0( float_as_int(color.x) );
    //optixSetPayload_1( float_as_int(color.y) );
    //optixSetPayload_2( float_as_int(color.z) );

    //optixSetPayload_0( optixGetPrimitiveIndex() );
}

extern "C" __global__ void __closesthit__full_occlusion()
{
    //phongShadowed();
}

extern "C" __global__ void __closesthit__glass_radiance()
{
    //const HitGroupData* sbt_data = (HitGroupData*) optixGetSbtDataPointer();
    //const Glass &glass = sbt_data->shading.glass;

    //RadiancePRD prd_radiance = getRadiancePRD();

    //float3 object_normal = make_float3(
    //    int_as_float( optixGetAttribute_0() ),
    //    int_as_float( optixGetAttribute_1() ),
    //    int_as_float( optixGetAttribute_2() ));
    //object_normal = normalize( object_normal );

    //// intersection vectors
    //const float3 n = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal) ); // normal
    //const float3 ray_orig = optixGetWorldRayOrigin();
    //const float3 ray_dir  = optixGetWorldRayDirection();                 // incident direction
    //const float  ray_t    = optixGetRayTmax();
    //float3 t;                                                            // transmission direction
    //float3 r;                                                            // reflection direction

    //float3 hit_point = ray_orig + ray_t * ray_dir;
    //SphereShellHitType hit_type = (SphereShellHitType) optixGetHitKind();
    //float3 front_hit_point = hit_point, back_hit_point = hit_point;

    //if (hit_type & HIT_OUTSIDE_FROM_OUTSIDE || hit_type & HIT_INSIDE_FROM_INSIDE)
    //{
    //    front_hit_point += params.scene_epsilon * object_normal;
    //    back_hit_point  -= params.scene_epsilon * object_normal;
    //}
    //else
    //{
    //    front_hit_point -= params.scene_epsilon * object_normal;
    //    back_hit_point  += params.scene_epsilon * object_normal;
    //}

    //const float3 fhp = optixTransformPointFromObjectToWorldSpace( front_hit_point );
    //const float3 bhp = optixTransformPointFromObjectToWorldSpace( back_hit_point );

    //float reflection = 1.0f;
    //float3 result = make_float3(0.0f);

    //const int depth = prd_radiance.depth;

    //float3 beer_attenuation;
    //if(dot(n, ray_dir) > 0)
    //{
    //    // Beer's law attenuation
    //    beer_attenuation = exp(glass.extinction_constant * ray_t);
    //} else
    //{
    //    beer_attenuation = make_float3(1);
    //}

    //// refraction
    //// compare depth to max_depth - 1 to leave room for a potential shadow ray trace
    //if (depth < min(glass.refraction_maxdepth, params.max_depth - 1))
    //{
    //    if ( refract(t, ray_dir, n, glass.refraction_index) )
    //    {
    //        // check for external or internal reflection
    //        float cos_theta = dot(ray_dir, n);
    //        if (cos_theta < 0.0f)
    //            cos_theta = -cos_theta;
    //        else
    //            cos_theta = dot(t, n);

    //        reflection = fresnel_schlick(
    //            cos_theta,
    //            glass.fresnel_exponent,
    //            glass.fresnel_minimum,
    //            glass.fresnel_maximum);

    //        float importance =
    //            prd_radiance.importance
    //            * (1.0f-reflection)
    //            * luminance( glass.refraction_color * beer_attenuation );
    //        float3 color = glass.cutoff_color;
    //        if ( importance > glass.importance_cutoff )
    //        {
    //            color = traceRadianceRay(bhp, t, depth+1, importance);
    //        }
    //        result += (1.0f - reflection) * glass.refraction_color * color;
    //    }
    //    // else TIR
    //} // else reflection==1 so refraction has 0 weight

    //// reflection
    //// compare depth to max_depth - 1 to leave room for a potential shadow ray trace
    //float3 color = glass.cutoff_color;
    //if (depth < min(glass.reflection_maxdepth, params.max_depth - 1))
    //{
    //    r = reflect(ray_dir, n);

    //    float importance =
    //        prd_radiance.importance
    //        * reflection
    //        * luminance( glass.reflection_color * beer_attenuation );
    //    if ( importance > glass.importance_cutoff )
    //    {
    //        color = traceRadianceRay( fhp, r, depth+1, importance );
    //    }
    //}
    //result += reflection * glass.reflection_color * color;

    //result = result * beer_attenuation;

    //prd_radiance.result = result;
    //setRadiancePRD(prd_radiance);
}
//
//extern "C" __global__ void __anyhit__glass_occlusion()
//{
//    const HitGroupData* sbt_data = (HitGroupData*) optixGetSbtDataPointer();
//    const Glass &glass = sbt_data->shading.glass;
//
//    float3 object_normal = make_float3(
//        int_as_float( optixGetAttribute_0() ),
//        int_as_float( optixGetAttribute_1() ),
//        int_as_float( optixGetAttribute_2() ));
//
//    OcclusionPRD shadow_prd = getOcclusionPRD();
//
//    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
//    float nDi = fabs(dot(world_normal, optixGetWorldRayDirection()));
//
//    shadow_prd.attenuation *= 1-fresnel_schlick(nDi, 5, 1-glass.shadow_attenuation, make_float3(1));
//    setOcclusionPRD(shadow_prd);
//
//    // Test the attenuation of the light from the glass shell
//    if(luminance(shadow_prd.attenuation) < glass.importance_cutoff)
//        // The attenuation is so high, > 99% blocked, that we can consider testing to be done.
//        optixTerminateRay();
//    else
//        // There is still some light coming through the glass shell that we should test other occluders.
//        // We "ignore" the intersection with the glass shell, meaning that shadow testing will continue.
//        // If the ray does not hit another occluder, the light's attenuation from this glass shell
//        // (along with other glass shells) is then used.
//        optixIgnoreIntersection();
//}

extern "C" __global__ void __miss__constant_bg()
{
    //optixSetPayload_0( 5 );
    //const MissData* sbt_data = (MissData*) optixGetSbtDataPointer();
    //RadiancePRD prd = getRadiancePRD();
    //prd.result = sbt_data->bg_color;
    //setRadiancePRD(prd);
}
