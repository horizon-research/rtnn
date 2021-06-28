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
#include <optix_types.h>
#include <sutil/vec_math.h>

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

struct BasicLight
{
    float3  pos;
    float3  color;
};


struct Params
{
    unsigned int     subframe_index;
    unsigned int*    frame_buffer;
    float3*          points;
    float3*          queries;
    float            radius;
    unsigned int     numPrims;
    unsigned int     knn;
    unsigned int*    d_vec_val;
    unsigned int*    d_vec_key;
    unsigned int     limit; // 1 for the initial run to sort indices; knn for future runs.

    int          max_depth;
    float        scene_epsilon;

    OptixTraversableHandle handle;
};


struct CameraData
{
    float3       eye;
    float3       U;
    float3       V;
    float3       W;
};


struct MissData
{
    float3 bg_color;
};


struct Sphere
{
    float3	center;
    float 	radius;
};


struct GeomData
{
  float3 *spheres;
};


enum SphereShellHitType {
    HIT_OUTSIDE_FROM_OUTSIDE = 1u << 0,
    HIT_OUTSIDE_FROM_INSIDE  = 1u << 1,
    HIT_INSIDE_FROM_OUTSIDE  = 1u << 2,
    HIT_INSIDE_FROM_INSIDE   = 1u << 3
};


struct SphereShell
{
	float3 	center;
	float 	radius1;
	float 	radius2;
};


struct Parallelogram
{
    Parallelogram() = default;
    Parallelogram( float3 v1, float3 v2, float3 anchor ):
    v1( v1 ), v2( v2 ), anchor( anchor )
    {
        float3 normal = normalize(cross( v1, v2 ));
        float d = dot( normal, anchor );
        this->v1 *= 1.0f / dot( v1, v1 );
        this->v2 *= 1.0f / dot( v2, v2 );
        plane = make_float4(normal, d);
    }
    float4	plane;
    float3 	v1;
    float3 	v2;
    float3 	anchor;
};


struct Phong
{
    float3 Ka;
    float3 Kd;
    float3 Ks;
    float3 Kr;
    float  phong_exp;
};


struct Glass
{
    float  importance_cutoff;
    float3 cutoff_color;
    float  fresnel_exponent;
    float  fresnel_minimum;
    float  fresnel_maximum;
    float  refraction_index;
    float3 refraction_color;
    float3 reflection_color;
    float3 extinction_constant;
    float3 shadow_attenuation;
    int    refraction_maxdepth;
    int    reflection_maxdepth;
};


struct CheckerPhong
{
    float3 Kd1, Kd2;
    float3 Ka1, Ka2;
    float3 Ks1, Ks2;
    float3 Kr1, Kr2;
    float  phong_exp1, phong_exp2;
    float2 inv_checker_size;
};


struct HitGroupData
{
    union
    {
        Sphere          sphere;
        SphereShell     sphere_shell;
        Parallelogram   parallelogram;
    } geometry;

    union
    {
        Phong           metal;
        Glass           glass;
        CheckerPhong    checker;
    } shading;
};


struct RadiancePRD
{
    float3 result;
    float  importance;
    int    depth;
};


struct OcclusionPRD
{
    float3 attenuation;
};
