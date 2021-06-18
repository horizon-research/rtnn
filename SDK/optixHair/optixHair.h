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
#pragma once

#include "whitted.h"
#include <sutil/CUDAOutputBuffer.h>
#include <cuda/BufferView.h>
#include <cuda_runtime.h>
#include <sutil/Aabb.h>
#include <sutil/Record.h>
#include <sutil/Camera.h>

typedef sutil::EmptyRecord                   RayGenRecord;
typedef sutil::EmptyRecord                   MissRecord;
typedef sutil::Record<whitted::HitGroupData> HitRecord;

//
// forward declarations
//

class Hair;
class Head;
class Camera;
class ShaderBindingTable;
class HairProgramGroups;

struct HairState
{
    OptixDeviceContext context = 0;

    Hair*         pHair;
    const Head*   pHead;
    sutil::Camera camera = {};

    unsigned int width  = 0;
    unsigned int height = 0;

    sutil::CUDAOutputBuffer<uchar4> outputBuffer = sutil::CUDAOutputBuffer<uchar4>(sutil::CUDAOutputBufferType::CUDA_DEVICE, 1, 1);
    sutil::CUDAOutputBuffer<float4> accumBuffer = sutil::CUDAOutputBuffer<float4>(sutil::CUDAOutputBufferType::CUDA_DEVICE, 1, 1);

    sutil::Aabb aabb;

    whitted::LaunchParams  params       = {};
    whitted::LaunchParams* deviceParams = nullptr;

    Light lights[2] = {};

    OptixTraversableHandle hHairGAS            = 0;
    CUdeviceptr            deviceBufferHairGAS = 0;

    OptixTraversableHandle hIAS            = 0;
    CUdeviceptr            deviceBufferIAS = 0;

    // for curves SBT record
    GeometryData::Curves curves = {};

    //ShaderBindingTable* pSBT           = nullptr;
    HairProgramGroups*  pProgramGroups = nullptr;
    OptixPipeline       pipeline       = 0;
    OptixShaderBindingTable SBT = {};

};

void makeHairGAS( HairState* pState );
void makeInstanceAccelerationStructure( HairState* pState );
void makePipeline( HairState* pState );
void makeProgramGroups( HairState* pState );
void makeSBT( HairState* pState );
void renderFrame( HairState* pState );

void initializeParams( HairState* pState );
void updateParams( HairState* pState );
void updateSize( HairState* pState, int width, int height );
