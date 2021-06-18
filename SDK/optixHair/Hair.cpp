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
#include "Hair.h"

#include <sutil/sutil.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>

#include "ProgramGroups.h"
#include "Util.h"


Hair::Hair( const OptixDeviceContext context, const std::string& fileName )
    : m_context( context )
{
    std::ifstream input( fileName.c_str(), std::ios::binary );
    SUTIL_ASSERT_MSG( input.is_open(), "Unable to open " + fileName + "." );

    input.read( reinterpret_cast<char*>( &m_header ), sizeof( FileHeader ) );
    SUTIL_ASSERT( input );
    SUTIL_ASSERT_MSG( strncmp( m_header.magic, "HAIR", 4 ) == 0, "Hair-file error: Invalid file format." + fileName );
    m_header.fileInfo[87] = 0;

    // Segments array(unsigned short)
    // The segements array contains the number of linear segments per strand;
    // thus there are segments + 1 control-points/vertices per strand.
    auto strandSegments = std::vector<unsigned short>( numberOfStrands() );
    if( hasSegments() )
    {
        input.read( reinterpret_cast<char*>( strandSegments.data() ), numberOfStrands() * sizeof( unsigned short ) );
        SUTIL_ASSERT_MSG( input, "Hair-file error: Cannot read segments." );
    }
    else
    {
        std::fill( strandSegments.begin(), strandSegments.end(), defaultNumberOfSegments() );
    }

    // Compute strands vector<unsigned int>. Each element is the index to the
    // first point of the first segment of the strand. The last entry is the
    // index "one beyond the last vertex".
    m_strands    = std::vector<int>( strandSegments.size() + 1 );
    auto strand = m_strands.begin();
    *strand++   = 0;
    for( auto segments : strandSegments )
    {
        *strand = *( strand - 1 ) + 1 + segments;
        strand++;
    }

    // Points array(float)
    SUTIL_ASSERT_MSG( hasPoints(), "Hair-file error: File contains no points." );
    m_points = std::vector<float3>( numberOfPoints() );
    input.read( reinterpret_cast<char*>( m_points.data() ), numberOfPoints() * sizeof( float3 ) );
    SUTIL_ASSERT_MSG( input, "Hair-file error: Cannot read points." );

    // Thickness array(float)
    m_thickness = std::vector<float>( numberOfPoints() );
    if( hasThickness() )
    {
        input.read( reinterpret_cast<char*>( m_thickness.data() ), numberOfPoints() * sizeof( float ) );
        SUTIL_ASSERT_MSG( input, "Hair-file error: Cannot read thickness." );
    }
    else
    {
        std::fill( m_thickness.begin(), m_thickness.end(), defaultThickness() );
    }

    SUTIL_ASSERT_MSG( !hasAlpha(), "Not implemented: Alpha data." );
    SUTIL_ASSERT_MSG( !hasColor(), "Not implemented: Color data." );

    //
    // Compute the axis-aligned bounding box for this hair geometry.
    //
    for( auto point : m_points )
    {
        m_aabb.include( point );
    }
    // expand the aabb by the maximum hair radius
    float max_width = defaultThickness();
    if( hasThickness() )
    {
        max_width = *std::max_element( m_thickness.begin(), m_thickness.end() );
    }
    m_aabb.m_min = m_aabb.m_min - make_float3( max_width );
    m_aabb.m_max = m_aabb.m_max + make_float3( max_width );
}

Hair::~Hair() {}

void Hair::gatherProgramGroups( HairProgramGroups* pProgramGroups ) const
{
    OptixProgramGroupDesc programGroupDesc = {};
    //
    // Shader blending red and green via segment u.
    //
    programGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programGroupDesc.hitgroup.moduleCH            = pProgramGroups->m_shadingModule;
    programGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__curve_segment_u";
    if( QUADRATIC_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_quadraticCurveModule;
    else if( CUBIC_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_cubicCurveModule;
    else if( LINEAR_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_linearCurveModule;
    programGroupDesc.hitgroup.entryFunctionNameIS = 0;  // automatically supplied for built-in moduleq
    pProgramGroups->add( programGroupDesc, programName() + "SegmentU" );

    //
    // Shader blending red and green via with root-to-tip (strand) u.
    //
    memset( &programGroupDesc, 0, sizeof( OptixProgramGroupDesc ) );
    programGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programGroupDesc.hitgroup.moduleCH            = pProgramGroups->m_shadingModule;
    programGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__curve_strand_u";
    if( QUADRATIC_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_quadraticCurveModule;
    else if( CUBIC_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_cubicCurveModule;
    else if( LINEAR_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_linearCurveModule;
    programGroupDesc.hitgroup.entryFunctionNameIS = 0;  // automatically supplied for built-in moduleq
    pProgramGroups->add( programGroupDesc, programName() + "StrandU" );

    //
    // Shader switching color based on strand index..
    //
    memset( &programGroupDesc, 0, sizeof( OptixProgramGroupDesc ) );
    programGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programGroupDesc.hitgroup.moduleCH            = pProgramGroups->m_shadingModule;
    programGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__curve_strand_idx";
    if( QUADRATIC_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_quadraticCurveModule;
    else if( CUBIC_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_cubicCurveModule;
    else if( LINEAR_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_linearCurveModule;
    programGroupDesc.hitgroup.entryFunctionNameIS = 0;  // automatically supplied for built-in moduleq
    pProgramGroups->add( programGroupDesc, programName() + "StrandIndex" );

    // Occlusion shader for shadow rays
    memset( &programGroupDesc, 0, sizeof( OptixProgramGroupDesc ) );
    programGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programGroupDesc.hitgroup.moduleCH            = pProgramGroups->m_whittedModule;
    programGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
    if( QUADRATIC_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_quadraticCurveModule;
    else if( CUBIC_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_cubicCurveModule;
    else if( LINEAR_BSPLINE == m_splineMode )
        programGroupDesc.hitgroup.moduleIS = pProgramGroups->m_linearCurveModule;
    programGroupDesc.hitgroup.entryFunctionNameIS = 0;  // automatically supplied for built-in modul
    pProgramGroups->add( programGroupDesc, "occludeCurve" );
}

std::string Hair::programName() const
{
    switch( m_splineMode ) {
    case LINEAR_BSPLINE:
        return "hitLinearCurve";
    case QUADRATIC_BSPLINE:
        return "hitQuadraticCurve";
    case CUBIC_BSPLINE:
        return "hitCubicCurve";
    default:
        SUTIL_ASSERT_MSG( false, "Invalid b-spline mode" );
    }

    return "";
}

std::string Hair::programSuffix() const
{
    switch( m_shadeMode ) {
    case SEGMENT_U:
        return "SegmentU";
    case STRAND_U:
        return "StrandU";
    case STRAND_IDX:
        return "StrandIndex";
    default:
        SUTIL_ASSERT_MSG( false, "Invalid hair-shading mode" );
    }

    return "";
}

uint32_t Hair::numberOfStrands() const
{
    return m_header.numStrands;
}

uint32_t Hair::numberOfPoints() const
{
    return m_header.numPoints;
}

uint32_t Hair::defaultNumberOfSegments() const
{
    return m_header.defaultNumSegments;
}

float Hair::defaultThickness() const
{
    return m_header.defaultThickness;
}

float Hair::defaultAlpha() const
{
    return m_header.defaultAlpha;
}

float3 Hair::defaultColor() const
{
    return make_float3( m_header.defaultColor.x, m_header.defaultColor.y, m_header.defaultColor.z );
}

std::string Hair::fileInfo() const
{
    return std::string( m_header.fileInfo );
}

bool Hair::hasSegments() const
{
    return ( m_header.flags & ( 0x1 << 0 ) ) > 0;
}

bool Hair::hasPoints() const
{
    return ( m_header.flags & ( 0x1 << 1 ) ) > 0;
}

bool Hair::hasThickness() const
{
    return ( m_header.flags & ( 0x1 << 2 ) ) > 0;
}

bool Hair::hasAlpha() const
{
    return ( m_header.flags & ( 0x1 << 3 ) ) > 0;
}

bool Hair::hasColor() const
{
    return ( m_header.flags & ( 0x1 << 4 ) ) > 0;
}

std::vector<float3> Hair::points() const
{
    return m_points;
}

std::vector<float> Hair::widths() const
{
    return m_thickness;
}

int Hair::numberOfSegments() const
{
    return numberOfPoints() - numberOfStrands() * curveDegree();
}

// Compute a list of all segment indices making up the curves array.
//
// The structure of the list is as follows:
// * For each strand all segments are listed in order from root to tip.
// * Segment indices are identical to the index of the first control-point
//   of a segment.
// * The number of segments per strand is dependent on the curve degree; e.g.
//   a cubic segment requires four control points, thus a cubic strand with n
//   control points will have (n - 3) segments.
//
std::vector<int> Hair::segments() const
{
    std::vector<int> segments;
    // loop to one before end, as last strand value is the "past last valid vertex"
    // index
    for( auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand )
    {
        const int start = *( strand );                      // first vertex in first segment
        const int end   = *( strand + 1 ) - curveDegree();  // second vertex of last segment
        for( int i = start; i < end; ++i )
        {
            segments.push_back( i );
        }
    }

    return segments;
}

std::vector<float2> Hair::strandU() const
{
    std::vector<float2> strand_u;
    for( auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand )
    {
        const int   start    = *( strand );
        const int   end      = *( strand + 1 ) - curveDegree();
        const int   segments = end - start;  // number of strand's segments
        const float scale    = 1.0f / segments;
        for( int i = 0; i < segments; ++i )
        {
            strand_u.push_back( make_float2( i * scale, scale ) );
        }
    }

    return strand_u;
}

std::vector<int> Hair::strandIndices() const
{
    std::vector<int> strandIndices;
    int              strandIndex = 0;
    for( auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand )
    {
        const int start = *( strand );
        const int end   = *( strand + 1 ) - curveDegree();
        for( auto segment = start; segment != end; ++segment )
        {
            strandIndices.push_back( strandIndex );
        }
        ++strandIndex;
    }

    return strandIndices;
}

std::vector<uint2> Hair::strandInfo() const
{
    std::vector<uint2> strandInfo;
    unsigned int       firstPrimitiveIndex = 0;
    for( auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand )
    {
        uint2 info;
        info.x = firstPrimitiveIndex;                        // strand's start index
        info.y = *( strand + 1 ) - *(strand)-curveDegree();  // number of segments in strand
        firstPrimitiveIndex += info.y;                       // increment with number of primitives/segments in strand
        strandInfo.push_back( info );
    }
    return strandInfo;
}

void Hair::setRadiusMode( Radius radiusMode )
{
    if(  m_radiusMode != radiusMode )
    {
        m_radiusMode = radiusMode;
        if( CONSTANT_R == m_radiusMode )
        {
            // assign all radii the root radius
            const float r = m_thickness[0];
            for( auto ir = m_thickness.begin(); ir != m_thickness.end(); ++ir )
                *ir = r;
        }
        else if( TAPERED_R == m_radiusMode )
        {
            const float r = m_thickness[0];
            for( auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand )
            {
                const int rootVertex = *( strand );
                const int vertices   = *( strand + 1 ) - rootVertex;  // vertices in strand
                for( int i = 0; i < vertices; ++i )
                {
                    m_thickness[rootVertex + i] = r * ( vertices - 1 - i ) / static_cast<float>( vertices - 1 );
                }
            }
        }
    }
}

std::string toString( bool b )
{
    std::string result;
    if( b )
        result = "true";
    else
        result = "false";
    return result;
}

std::ostream& operator<<( std::ostream& o, Hair::SplineMode splineMode )
{
    switch( splineMode )
    {
        case Hair::LINEAR_BSPLINE:
            o <<  "LINEAR_BSPLINE";
            break;
        case Hair::QUADRATIC_BSPLINE:
            o << "QUADRATIC_BSPLINE";
            break;
        case Hair::CUBIC_BSPLINE:
            o <<  "CUBIC_BSPLINE";
            break;
        default:
            SUTIL_ASSERT_MSG( false, "Invalid spline mode." );
    }

    return o;
}

std::ostream& operator<<( std::ostream& o, const Hair& hair )
{
    o << "Hair: " << std::endl;
    o << "Number of strands:          " << hair.numberOfStrands() << std::endl;
    o << "Number of points:           " << hair.numberOfPoints() << std::endl;
    o << "Spline mode:                " << hair.m_splineMode << std::endl;
    o << "Contains segments:          " << toString( hair.hasSegments() ) << std::endl;
    o << "Contains points:            " << toString( hair.hasPoints() ) << std::endl;
    o << "Contains alpha:             " << toString( hair.hasAlpha() ) << std::endl;
    o << "Contains color:             " << toString( hair.hasColor() ) << std::endl;
    o << "Default number of segments: " << hair.defaultNumberOfSegments() << std::endl;
    o << "Default thickness:          " << hair.defaultThickness() << std::endl;
    o << "Default alpha:              " << hair.defaultAlpha() << std::endl;
    float3 color = hair.defaultColor();
    o << "Default color:              (" << color.x << ", " << color.y << ", " << color.z << ")" << std::endl;
    std::string fileInfo = hair.fileInfo();
    o << "File info:                  ";
    if( fileInfo.empty() )
        o << "n/a" << std::endl;
    else
        o << fileInfo << std::endl;

    o << "Strands: [" << hair.m_strands[0] << "..." << hair.m_strands[hair.m_strands.size() - 1] << "]" << std::endl;
    o << "Points: [" << hair.m_points[0] << "..." << hair.m_points[hair.m_points.size() - 1] << "]" << std::endl;
    o << "Thickness: [" << hair.m_thickness[0] << "..." << hair.m_thickness[hair.m_thickness.size() - 1] << "]" << std::endl;
    o << "Bounding box: [" << hair.m_aabb.m_min << ", " << hair.m_aabb.m_max << "]" << std::endl;

    return o;
}
