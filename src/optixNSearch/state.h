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

#pragma once

#include <float.h>
#include <vector_types.h>
#include <optix_types.h>
#include <unordered_set>
#include "optixNSearch.h"

// the SDK cmake defines NDEBUG in the Release build, but we still want to use assert
// TODO: fix it in cmake files?
#undef NDEBUG
#include <assert.h>

#define OMIT_ON_E2EMSR(x) \
  if (state.msr == 0) x   \

struct RTNNState
{
    OptixDeviceContext          context                   = 0;
    OptixTraversableHandle*     gas_handle                = nullptr;
    CUdeviceptr*                d_gas_output_buffer       = nullptr;

    OptixModule                 geometry_module           = 0;
    OptixModule                 camera_module             = 0;

    OptixProgramGroup           raygen_prog_group         = 0;
    OptixProgramGroup           radiance_miss_prog_group  = 0;
    OptixProgramGroup           radiance_metal_sphere_prog_group  = 0;

    OptixPipeline*              pipeline                  = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options  = {};

    cudaStream_t*               stream                    = nullptr;
    Params                      params;
    Params*                     d_params                  = nullptr;

    float3*                     h_points                  = nullptr;
    float3*                     h_queries                 = nullptr;
    float3**                    h_ndpoints                = nullptr;
    float3**                    h_ndqueries               = nullptr;
    float*                      h_radii                   = nullptr;
    int                         dim;
    bool                        msr                       = true;
    bool                        sanCheck                  = false;

    int32_t                     device_id                 = 0;
    std::string                 searchMode                = "radius";
    std::string                 pfile;
    std::string                 qfile;
    unsigned int                knn                       = 50;
    float                       gRadius                   = 2.0;
    float                       radius                    = 2.0;
    float*                      radii                     = nullptr;
    int                         qGasSortMode              = 2; // no GAS-based sort vs. 1D vs. ID
    int                         pointSortMode             = 1; // no sort vs. morton order vs. raster order vs. 1D order
    int                         querySortMode             = 1; // no sort vs. morton order vs. raster order vs. 1D order
    float                       crRatio                   = 8; // cellSize = radius / crRatio
    float                       gsrRatio                  = 1;
    bool                        toGather                  = false;
    bool                        samepq                    = false;
    bool                        sameData                  = false;
    bool                        interleave                = true;
    bool                        partition                 = true;
    bool                        autoNB                    = true;
    bool                        autoCR                    = true;
    int                         approxMode                = 2;
    int                         mcScale                   = 4;
    float                       crStep                    = 1.01;
    bool                        deferFree                 = true;
    bool                        filterQueries             = false;

    unsigned int                numPoints                 = 0;
    unsigned int                numQueries                = 0;
    unsigned int**              d_r2q_map                 = nullptr;
    unsigned int*               numActQueries             = nullptr;
    float*                      launchRadius              = nullptr;
    void**                      h_res                     = nullptr;
    float3**                    d_actQs                   = nullptr;
    float3**                    h_actQs                   = nullptr;
    void**                      d_aabb                    = nullptr;
    void**                      d_temp_buffer_gas         = nullptr;
    void**                      d_buffer_temp_output_gas_and_compacted_size = nullptr;
    void*                       d_CellParticleCounts_ptr_p = nullptr;
    void*                       d_CellOffsets_ptr_p       = nullptr;
    float3*                     h_fltQs                   = nullptr;
    unsigned int                numFltQs                  = 0;

    std::unordered_set<void*>   d_pointers;
    std::unordered_set<void*>   d_gridPointers;

    int                         numOfBatches              = -1;
    int                         maxBatchCount             = 1;
    float                       totDRAMSize               = 0; // GB
    float                       gpuMemUsed                = 0; // MB
    float                       estGasSize                = -1; // MB

    float3                      pMin;
    float3                      pMax;
    float3                      qMin;
    float3                      qMax;
    float3                      Min;
    float3                      Max;

    OptixShaderBindingTable     sbt                       = {};
};
