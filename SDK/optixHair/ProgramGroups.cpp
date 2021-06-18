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
#include "ProgramGroups.h"

#include <optix.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <cstring>


ProgramGroups::ProgramGroups( const OptixDeviceContext    context,
                              OptixPipelineCompileOptions pipeOptions,
                              OptixProgramGroupOptions    programGroupOptions = {} )
    : m_context( context )
    , m_pipeOptions( pipeOptions )
    , m_programGroupOptions( programGroupOptions )
{
    ;
}

void ProgramGroups::add( const OptixProgramGroupDesc& programGroupDescriptor, const std::string& name )
{
    // Only add a new program group, if one with `name` doesn't yet exist.
    if( m_nameToIndex.find( name ) == m_nameToIndex.end() )
    {
        size_t last         = m_programGroups.size();
        m_nameToIndex[name] = static_cast<unsigned int>( last );
        m_programGroups.resize( last + 1 );
        OPTIX_CHECK_LOG2( optixProgramGroupCreate( m_context, &programGroupDescriptor,
                                                   1,  // num program groups
                                                   &m_programGroupOptions, LOG, &LOG_SIZE, &m_programGroups[last] ) );
    }
}

const OptixProgramGroup& ProgramGroups::operator[]( const std::string& name ) const
{
    auto iter = m_nameToIndex.find( name );
    SUTIL_ASSERT( iter != m_nameToIndex.end() );
    size_t index = iter->second;
    return m_programGroups[index];
}

const OptixProgramGroup* ProgramGroups::data() const
{
    return &( m_programGroups[0] );
}

unsigned int ProgramGroups::size() const
{
    return static_cast<unsigned int>( m_programGroups.size() );
}

//
// HairProgramGroups
//
HairProgramGroups::HairProgramGroups( const OptixDeviceContext context, OptixPipelineCompileOptions pipeOptions )
    : ProgramGroups( context, pipeOptions )
{
    //
    // Create modules
    //
    const OptixModuleCompileOptions defaultOptions = {OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                                                      OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                                                      OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO};

    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixHair.cu" );
    OPTIX_CHECK_LOG2( optixModuleCreateFromPTX( context,
                                                &defaultOptions,
                                                &pipeOptions,
                                                ptx.c_str(),
                                                ptx.size(),
                                                LOG, &LOG_SIZE,
                                                &m_shadingModule ) );

    ptx = sutil::getPtxString( nullptr, nullptr, "whitted.cu" );
    OPTIX_CHECK_LOG2( optixModuleCreateFromPTX( context,
                                                &defaultOptions,
                                                &pipeOptions,
                                                ptx.c_str(),
                                                ptx.size(),
                                                LOG, &LOG_SIZE,
                                                &m_whittedModule ) );

    if( pipeOptions.usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE ) {
        OptixBuiltinISOptions builtinISOptions = {};
        builtinISOptions.builtinISModuleType   = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
        OPTIX_CHECK( optixBuiltinISModuleGet( context, &defaultOptions, &pipeOptions, &builtinISOptions, &m_quadraticCurveModule ) );
    }
    if( pipeOptions.usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE ) {
        OptixBuiltinISOptions builtinISOptions = {};
        builtinISOptions.builtinISModuleType   = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
        OPTIX_CHECK( optixBuiltinISModuleGet( context, &defaultOptions, &pipeOptions, &builtinISOptions, &m_cubicCurveModule ) );
    }
    if( pipeOptions.usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR ) {
        OptixBuiltinISOptions builtinISOptions = {};
        builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
        OPTIX_CHECK( optixBuiltinISModuleGet( context, &defaultOptions, &pipeOptions, &builtinISOptions, &m_linearCurveModule ) );
    }
}
