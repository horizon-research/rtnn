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

#include <optix.h>

#include "optixHair.h"

#include <string>
#include <vector>


// forward declarations
class Context;
class ProgramGroups;


class Head
{
  public:
    Head( const OptixDeviceContext context, const std::string& fileName );
    ~Head();

    virtual OptixTraversableHandle traversable() const;

    virtual void gatherProgramGroups( HairProgramGroups* pProgramGroups ) const;

    virtual std::vector<HitRecord> sbtHitRecords( const ProgramGroups& programs, size_t rayTypes ) const;

    size_t numberOfVertices() const { return m_vertices; }

    size_t numberOfTriangles() const { return m_triangles; }

    virtual sutil::Aabb aabb() const { return m_aabb; }

    virtual unsigned int usesPrimitiveTypes() const { return OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; }

  private:

    size_t                         m_vertices = 0;
    size_t                         m_triangles     = 0;
    CUdeviceptr                    m_buffer = 0;
    sutil::Aabb                    m_aabb;
    mutable OptixTraversableHandle m_hGAS            = 0;
    mutable CUdeviceptr            m_deviceBufferGAS = 0;
    GeometryData::TriangleMesh m_triangleMesh;

    friend std::ostream& operator<<( std::ostream& o, const Head& head );
};

// Ouput operator for Head
std::ostream& operator<<( std::ostream& o, const Head& head );
