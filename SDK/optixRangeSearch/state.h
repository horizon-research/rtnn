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

#include <vector_types.h>
#include <optix_types.h>
#include "optixRangeSearch.h"

// TODO remove this but use cudaStream_t for stream
//#include <sutil/CUDAOutputBuffer.h>

struct WhittedState
{
    OptixDeviceContext          context                   = 0;
    OptixTraversableHandle      gas_handle                = {};
    CUdeviceptr                 d_gas_output_buffer       = {};

    OptixModule                 geometry_module           = 0;
    OptixModule                 camera_module             = 0;

    OptixProgramGroup           raygen_prog_group         = 0;
    OptixProgramGroup           radiance_miss_prog_group  = 0;
    OptixProgramGroup           radiance_metal_sphere_prog_group  = 0;

    OptixPipeline               pipeline                  = 0;
    OptixPipelineCompileOptions pipeline_compile_options  = {};

    cudaStream_t                stream                    = 0;
    Params                      params;
    Params*                     d_params                  = nullptr;

    float*                      d_key                     = nullptr;
    float3*                     h_points                  = nullptr;
    float3*                     h_queries                 = nullptr;
    unsigned int*               d_r2q_map                 = nullptr;
    float3**                    h_ndpoints                = nullptr;
    float3**                    h_ndqueries               = nullptr;
    int                         dim;

    int32_t                     device_id                 = 1;
    std::string                 searchMode                = "radius";
    std::string                 pfile;
    std::string                 qfile;
    float                       radius                    = 0.0;
    int                         qGasSortMode              = 2; // no GAS-based sort vs. 1D vs. ID
    int                         pointSortMode             = 1; // no sort vs. morton order vs. raster order vs. 1D order
    int                         querySortMode             = 1; // no sort vs. morton order vs. raster order vs. 1D order
    float                       crRatio                   = 8; // celSize = radius / crRatio
    float                       sortingGAS                = 1;
    bool                        toGather                  = false;
    bool                        reorderPoints             = false;
    bool                        samepq                    = false;

    unsigned int                numPoints                 = 0;
    unsigned int                numQueries                = 0;
    unsigned int                numTotalQueries           = 0;
    unsigned int                numActQueries[2]          = {0};
    float                       launchRadius[2]           = {0.0};

    float3*                     d_actQs[2]                = {nullptr};
    float3*                     h_actQs[2]                = {nullptr};

    bool                        partition                 = false;
    bool*                       cellMask                  = nullptr;
    int                         partThd                   = 1;

    float3                      Min;
    float3                      Max;

    OptixShaderBindingTable     sbt                       = {};
};
