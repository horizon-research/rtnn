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

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>



static void context_log_cb( uint32_t level, const char* tag, const char* message, void* /*cbdata*/ )
{
    if( level < 4 )
        std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
                  << message << "\n";
}



class OptiXDenoiser
{
public:
    struct Data
    {
        uint32_t  width    = 0;
        uint32_t  height   = 0;
        float*    color    = nullptr;
        float*    albedo   = nullptr;
        float*    normal   = nullptr;
        float*    output   = nullptr;
    };

    // Initialize the API and push all data to the GPU -- normaly done only once per session
    void init( Data& data );

    // Execute the denoiser. In interactive sessions, this would be done once per frame/subframe
    void exec();

    // Cleanup state, deallocate memory -- normally done only once per render session
    void finish(); 


private:
    OptixDeviceContext    m_context      = nullptr;
    OptixDenoiser         m_denoiser     = nullptr;
    OptixDenoiserParams   m_params       = {};

    CUdeviceptr           m_intensity    = 0;
    CUdeviceptr           m_scratch      = 0;
    uint32_t              m_scratch_size = 0;
    CUdeviceptr           m_state        = 0;
    uint32_t              m_state_size   = 0;

    OptixImage2D          m_inputs[3]    = {};
    OptixImage2D          m_output;

    float*                m_host_output = nullptr;
};



void OptiXDenoiser::init( Data& data )
{
    SUTIL_ASSERT( data.color  );
    SUTIL_ASSERT( data.output );
    SUTIL_ASSERT( data.width  );
    SUTIL_ASSERT( data.height );
    SUTIL_ASSERT_MSG( !data.normal || data.albedo, "Currently albedo is required if normal input is given" );

    m_host_output = data.output;

    //
    // Initialize CUDA and create OptiX context
    //
    {
        // Initialize CUDA
        CUDA_CHECK( cudaFree( 0 ) );

        CUcontext cu_ctx = 0;  // zero means take the current context
        OPTIX_CHECK( optixInit() );
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &context_log_cb;
        options.logCallbackLevel          = 4;
        OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &m_context ) );
    }

    //
    // Create denoiser
    //
    {
        OptixDenoiserOptions options = {};
        options.inputKind   =
            data.normal ? OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL :
            data.albedo ? OPTIX_DENOISER_INPUT_RGB_ALBEDO        :
                          OPTIX_DENOISER_INPUT_RGB;
        OPTIX_CHECK( optixDenoiserCreate( m_context, &options, &m_denoiser ) );
        OPTIX_CHECK( optixDenoiserSetModel(
                    m_denoiser,
                    OPTIX_DENOISER_MODEL_KIND_HDR,
                    nullptr, // data
                    0        // size
                    ) );
    }


    //
    // Allocate device memory for denoiser
    //
    {
        OptixDenoiserSizes denoiser_sizes;
        OPTIX_CHECK( optixDenoiserComputeMemoryResources(
                    m_denoiser,
                    data.width,
                    data.height,
                    &denoiser_sizes
                    ) );
        
        // NOTE: if using tiled denoising, we would set scratch-size to 
        //       denoiser_sizes.withOverlapScratchSizeInBytes
        m_scratch_size = static_cast<uint32_t>( denoiser_sizes.withoutOverlapScratchSizeInBytes );

        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_intensity ),
                    sizeof( float )
                    ) );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_scratch ),
                    m_scratch_size 
                    ) );

        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_state ),
                    denoiser_sizes.stateSizeInBytes
                    ) );
        m_state_size = static_cast<uint32_t>( denoiser_sizes.stateSizeInBytes );

        const uint64_t frame_byte_size = data.width*data.height*sizeof(float4);
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_inputs[0].data ), frame_byte_size ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( m_inputs[0].data ),
                    data.color,
                    frame_byte_size,
                    cudaMemcpyHostToDevice
                    ) );
        m_inputs[0].width              = data.width;
        m_inputs[0].height             = data.height;
        m_inputs[0].rowStrideInBytes   = data.width*sizeof(float4);
        m_inputs[0].pixelStrideInBytes = sizeof(float4);
        m_inputs[0].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        m_inputs[1].data = 0;
        if( data.albedo )
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_inputs[1].data ), frame_byte_size ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( m_inputs[1].data ),
                        data.albedo,
                        frame_byte_size,
                        cudaMemcpyHostToDevice
                        ) );
            m_inputs[1].width              = data.width;
            m_inputs[1].height             = data.height;
            m_inputs[1].rowStrideInBytes   = data.width*sizeof(float4);
            m_inputs[1].pixelStrideInBytes = sizeof(float4);
            m_inputs[1].format             = OPTIX_PIXEL_FORMAT_FLOAT4;
        }

        m_inputs[2].data = 0;
        if( data.normal )
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_inputs[2].data ), frame_byte_size ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( m_inputs[2].data ),
                        data.normal,
                        frame_byte_size,
                        cudaMemcpyHostToDevice
                        ) );
            m_inputs[2].width              = data.width;
            m_inputs[2].height             = data.height;
            m_inputs[2].rowStrideInBytes   = data.width*sizeof(float4);
            m_inputs[2].pixelStrideInBytes = sizeof(float4);
            m_inputs[2].format             = OPTIX_PIXEL_FORMAT_FLOAT4;
        }

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_output.data ), frame_byte_size ) );
        m_output.width              = data.width;
        m_output.height             = data.height;
        m_output.rowStrideInBytes   = data.width*sizeof(float4);
        m_output.pixelStrideInBytes = sizeof(float4);
        m_output.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
    }

    //
    // Setup denoiser
    //
    {
        OPTIX_CHECK( optixDenoiserSetup(
                    m_denoiser,
                    0,  // CUDA stream
                    data.width,
                    data.height,
                    m_state,
                    m_state_size,
                    m_scratch,
                    m_scratch_size
                    ) );


        m_params.denoiseAlpha = 0;
        m_params.hdrIntensity = m_intensity;
        m_params.blendFactor  = 0.0f;
    }
}


void OptiXDenoiser::exec()
{
    OPTIX_CHECK( optixDenoiserComputeIntensity(
                m_denoiser,
                0, // CUDA stream
                m_inputs,
                m_intensity,
                m_scratch,
                m_scratch_size
                ) );

    OPTIX_CHECK( optixDenoiserInvoke(
                m_denoiser,
                0, // CUDA stream
                &m_params,
                m_state,
                m_state_size,
                m_inputs,
                m_inputs[2].data ? 3 : m_inputs[1].data ? 2 : 1, // num input channels
                0, // input offset X
                0, // input offset y
                &m_output,
                m_scratch,
                m_scratch_size
                ) );

    CUDA_SYNC_CHECK();
}


void OptiXDenoiser::finish() 
{
    const uint64_t frame_byte_size = m_output.width*m_output.height*sizeof(float4);
    CUDA_CHECK( cudaMemcpy(
                m_host_output,
                reinterpret_cast<void*>( m_output.data ),
                frame_byte_size,
                cudaMemcpyDeviceToHost
                ) );

    // Cleanup resources
    optixDenoiserDestroy( m_denoiser );
    optixDeviceContextDestroy( m_context );

    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_intensity)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_scratch)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_state)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_inputs[0].data)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_inputs[1].data)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_inputs[2].data)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_output.data)) ); 
}

