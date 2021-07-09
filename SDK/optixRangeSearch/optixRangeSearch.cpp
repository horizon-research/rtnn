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

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <sutil/Timing.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include <iomanip>
#include <cstring>
#include <fstream>
#include <string>
#include <random>
#include <cstdlib>

// the SDK cmake defines NDEBUG in the Release build, but we still want to use assert
#undef NDEBUG
#include <assert.h>

#include "optixRangeSearch.h"


//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

const int         max_trace = 12;

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT )

    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<GeomData>        RayGenRecord;
typedef Record<MissData>        MissRecord;
typedef Record<HitGroupData>    HitGroupRecord;

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

    CUstream                    stream                    = 0;
    Params                      params;
    Params*                     d_params                  = nullptr;

    float*                      d_key                     = nullptr;
    float3*                     h_points                  = nullptr;
    float3*                     h_queries                 = nullptr;
    unsigned int*               d_r2q_map                 = nullptr;
    float3**                    h_ndpoints                = nullptr;
    float3**                    h_ndqueries               = nullptr;
    int                         dim;

    std::string                 searchMode                = "radius";
    std::string                 pfile;
    std::string                 qfile;
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

    float3                      Min;
    float3                      Max;

    OptixShaderBindingTable     sbt                       = {};
};

enum ParticleType
{
    POINT = 0,
    QUERY = 1
};

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void sortByKey( thrust::device_vector<float>*, thrust::device_vector<unsigned int>* );
void sortByKey( thrust::device_vector<float>*, thrust::device_ptr<unsigned int>);
void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<unsigned int>, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_vector<unsigned int>*, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int>, unsigned int );
void sortByKey( thrust::device_vector<float>*, thrust::device_ptr<float3> );
void sortByKey( thrust::device_ptr<float>, thrust::device_ptr<float3>, unsigned int );
void sortByKey( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, unsigned int );
void gatherByKey ( thrust::device_vector<unsigned int>*, thrust::device_ptr<float3>, thrust::device_ptr<float3> );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float3>, thrust::device_ptr<float3>, unsigned int );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_vector<float>*, thrust::device_ptr<float>, unsigned int );
void gatherByKey ( thrust::device_ptr<unsigned int>, thrust::device_ptr<float>, thrust::device_ptr<float>, unsigned int );
thrust::device_ptr<unsigned int> getThrustDevicePtr(unsigned int);
thrust::device_ptr<unsigned int> genSeqDevice(unsigned int);
void exclusiveScan(thrust::device_ptr<unsigned int>, unsigned int, thrust::device_ptr<unsigned int>);
void fillByValue(thrust::device_ptr<unsigned int>, unsigned int, int);

void kComputeMinMax (unsigned int, unsigned int, float3*, unsigned int, int3*, int3*);
void kInsertParticles(unsigned int, unsigned int, GridInfo, float3*, unsigned int*, unsigned int*, unsigned int*, bool);
void kCountingSortIndices(unsigned int, unsigned int, GridInfo, unsigned int*, unsigned int*, unsigned int*, unsigned int*);
void computeMinMax(WhittedState&, ParticleType);
void gridSort(WhittedState&, ParticleType, bool);

int tokenize(std::string s, std::string del, float3** ndpoints, unsigned int lineId)
{
  int start = 0;
  int end = s.find(del);
  int dim = 0;

  std::vector<float> vcoords;
  while (end != -1) {
    float coord = std::stof(s.substr(start, end - start));
    //std::cout << coord << std::endl;
    if (ndpoints != nullptr) {
      vcoords.push_back(coord);
    }
    start = end + del.size();
    end = s.find(del, start);
    dim++;
  }
  float coord  = std::stof(s.substr(start, end - start));
  //std::cout << coord << std::endl;
  if (ndpoints != nullptr) {
    vcoords.push_back(coord);
  }
  dim++;

  assert(dim > 0);
  if ((dim % 3) != 0) dim = (dim/3+1)*3;

  if (ndpoints != nullptr) {
    for (int batch = 0; batch < dim/3; batch++) {
      float3 point = make_float3(vcoords[batch*3], vcoords[batch*3+1], vcoords[batch*3+2]);
      ndpoints[batch][lineId] = point;
    }
  }

  return dim;
}

float3** read_pc_data(const char* data_file, unsigned int* N, int* d) {
  std::ifstream file;

  file.open(data_file);
  if( !file.good() ) {
    std::cerr << "Could not read the frame data...\n";
    assert(0);
  }

  char line[1024];
  unsigned int lines = 0;
  int dim = 0;

  while (file.getline(line, 1024)) {
    if (lines == 0) {
      std::string str(line);
      dim = tokenize(str, ",", nullptr, 0);
    }
    lines++;
  }
  file.clear();
  file.seekg(0, std::ios::beg);

  *N = lines;
  *d = dim;

  float3** ndpoints = new float3*[dim/3];
  for (int i = 0; i < dim/3; i++) {
    ndpoints[i] = new float3[lines];
  }

  lines = 0;
  while (file.getline(line, 1024)) {
    std::string str(line);
    tokenize(str, ",", ndpoints, lines);

    //std::cerr << ndpoints[0][lines].x << "," << ndpoints[0][lines].y << "," << ndpoints[0][lines].z << std::endl;
    //std::cerr << ndpoints[1][lines].x << "," << ndpoints[1][lines].y << "," << ndpoints[1][lines].z << std::endl;
    lines++;
  }

  file.close();

  return ndpoints;
}

float3* read_pc_data(const char* data_file, unsigned int* N) {
  std::ifstream file;

  file.open(data_file);
  if( !file.good() ) {
    std::cerr << "Could not read the frame data...\n";
    assert(0);
  }

  char line[1024];
  unsigned int lines = 0;

  while (file.getline(line, 1024)) {
    lines++;
  }
  file.clear();
  file.seekg(0, std::ios::beg);
  *N = lines;

  float3* t_points = new float3[lines];

  lines = 0;
  while (file.getline(line, 1024)) {
    double x, y, z;

    sscanf(line, "%lf,%lf,%lf\n", &x, &y, &z);
    t_points[lines] = make_float3(x, y, z);
    //std::cerr << t_points[lines].x << ", " << t_points[lines].y << ", " << t_points[lines].z << std::endl;
    lines++;
  }

  file.close();

  return t_points;
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file          | -f <filename>   File for point cloud input\n";
    std::cerr << "         --searchmode    | -sm             Search mode; can only be \"knn\" or \"radius\" \n";
    std::cerr << "         --radius        | -r              Search radius\n";
    std::cerr << "         --knn           | -k              Max K returned\n";
    std::cerr << "         --gassort       | -s              GAS-based query sort mode\n";
    std::cerr << "         --pointsort     | -ps             Point sort mode\n";
    std::cerr << "         --querysort     | -qs             Query sort mode\n";
    std::cerr << "         --crratio       | -cr             cell/radius ratio\n";
    std::cerr << "         --sortingGAS    | -sg             Param for SortingGAS\n";
    std::cerr << "         --gather        | -g              Whether to gather queries after sort \n";
    std::cerr << "         --reorderpoints | -rp             Whether to reorder points after query sort \n";
    std::cerr << "         --help          | -h              Print this usage message\n";
    exit( 0 );
}

void oneDSort ( WhittedState& state, ParticleType type ) {
  // sort points/queries based on coordinates (x/y/z)
  unsigned int N;
  float3* particles;
  float3* h_particles;
  if (type == POINT) {
    N = state.numPoints;
    particles = state.params.points;
    h_particles = state.h_points;
  } else {
    N = state.numQueries;
    particles = state.params.queries;
    h_particles = state.h_queries;
  }

  // TODO: do this whole thing on GPU.
  // create 1d points as the sorting key and upload it to device memory
  thrust::host_vector<float> h_key(N);
  for(unsigned int i = 0; i < N; i++) {
    h_key[i] = h_particles[i].x;
  }

  float* d_key = nullptr;
  CUDA_CHECK( cudaMalloc(
      reinterpret_cast<void**>(&d_key),
      N * sizeof(float) ) );
  thrust::device_ptr<float> d_key_ptr = thrust::device_pointer_cast(d_key);
  thrust::copy(h_key.begin(), h_key.end(), d_key_ptr);

  // actual sort
  thrust::device_ptr<float3> d_particles_ptr = thrust::device_pointer_cast(particles);
  sortByKey( d_key_ptr, d_particles_ptr, N );
  CUDA_CHECK( cudaFree( (void*)d_key ) );

  // TODO: lift it outside of this function and combine with other sorts?
  // copy the sorted queries to host so that we build the GAS in the same order
  // note that the h_queries at this point still point to what h_points points to
  thrust::copy(d_particles_ptr, d_particles_ptr + N, h_particles);
}

void uploadData ( WhittedState& state ) {
  Timing::startTiming("upload points and/or queries");
    // Allocate device memory for points/queries
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.params.points ),
        state.numPoints * sizeof(float3) ) );
    
    CUDA_CHECK( cudaMemcpyAsync(
        reinterpret_cast<void*>( state.params.points ),
        state.h_points,
        state.numPoints * sizeof(float3),
        cudaMemcpyHostToDevice,
        state.stream
    ) );

    if (state.samepq) {
      // by default, params.queries and params.points point to the same device
      // memory. later if we decide to reorder the queries, we will allocate new
      // space in device memory and point params.queries to that space. this is
      // lazy query allocation.
      state.params.queries = state.params.points;
    } else {
      CUDA_CHECK( cudaMalloc(
          reinterpret_cast<void**>( &state.params.queries),
          state.numQueries * sizeof(float3) ) );
      
      CUDA_CHECK( cudaMemcpyAsync(
          reinterpret_cast<void*>( state.params.queries ),
          state.h_queries,
          state.numQueries * sizeof(float3),
          cudaMemcpyHostToDevice,
          state.stream
      ) );
    }
    CUDA_CHECK( cudaStreamSynchronize( state.stream ) ); // TODO: just so we can measure time
  Timing::stopTiming(true);
}

void initLaunchParams( WhittedState& state )
{
    state.params.frame_buffer = nullptr; // the result buffer
    state.params.d_r2q_map = nullptr; // contains the index to reorder rays

    state.params.max_depth = max_trace;
    state.params.scene_epsilon = 1.e-4f;
}

static void sphere_bound(float3 center, float radius, float result[6])
{
    OptixAabb *aabb = reinterpret_cast<OptixAabb*>(result);

    float3 m_min = center - radius;
    float3 m_max = center + radius;

    *aabb = {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

static void buildGas(
    const WhittedState &state,
    const OptixAccelBuildOptions &accel_options,
    const OptixBuildInput &build_input,
    OptixTraversableHandle &gas_handle,
    CUdeviceptr &d_gas_output_buffer
    )
{
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr d_temp_buffer_gas;

    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes));

    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        state.stream,
        &accel_options,
        &build_input,
        1,
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        &emitProperty,
        1) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        //OPTIX_CHECK( optixAccelCompact( state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );
        OPTIX_CHECK( optixAccelCompact( state.context, state.stream, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createGeometry( WhittedState &state, float sortingGAS )
{
    //
    // Build Custom Primitives
    //

    // Load AABB into device memory
    unsigned int numPrims = state.numPoints;
    OptixAabb* aabb = (OptixAabb*)malloc(numPrims * sizeof(OptixAabb));
    CUdeviceptr d_aabb;

    // Create an AABB whose volume is the same as the sphere
    //double sphere_volume = 4 / 3 * M_PI * state.params.radius * state.params.radius * state.params.radius;
    //double halfLength = std::cbrt(sphere_volume / 8);
    //std::cout << "\tAABB half length: " << halfLength << std::endl;
    //float radius = halfLength;

    float radius = state.params.radius/sortingGAS;

    for(unsigned int i = 0; i < numPrims; i++) {
      sphere_bound(
          state.h_points[i], radius,
          reinterpret_cast<float*>(&aabb[i]));
    }

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
        ), numPrims * sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( d_aabb ),
                aabb,
                numPrims * sizeof( OptixAabb ),
                cudaMemcpyHostToDevice,
                state.stream
    ) );

    // Setup AABB build input
    uint32_t* aabb_input_flags = (uint32_t*)malloc(numPrims * sizeof(uint32_t));

    for (unsigned int i = 0; i < numPrims; i++) {
      //aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
      aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
    }

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;
    aabb_input.customPrimitiveArray.numPrimitives = numPrims;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer         = 0;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes    = sizeof( uint32_t );
    aabb_input.customPrimitiveArray.primitiveIndexOffset         = 0;


    OptixAccelBuildOptions accel_options = {
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags
        OPTIX_BUILD_OPERATION_BUILD         // operation
    };


    buildGas(
        state,
        accel_options,
        aabb_input,
        state.gas_handle,
        state.d_gas_output_buffer);

    CUDA_CHECK( cudaFree( (void*)d_aabb) );
}

void createModules( WhittedState &state )
{
    OptixModuleCompileOptions module_compile_options = {
        100,                                    // maxRegisterCount
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,     // optLevel
        OPTIX_COMPILE_DEBUG_LEVEL_NONE          // debugLevel
    };
    char log[2048];
    size_t sizeof_log = sizeof(log);

    {
        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "geometry.cu" );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &state.geometry_module ) );
    }

    {
        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "camera.cu" );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &state.camera_module ) );
    }
}

static void createCameraProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           cam_prog_group;
    OptixProgramGroupOptions    cam_prog_group_options = {};
    OptixProgramGroupDesc       cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.camera_module;
    if (state.searchMode == "knn")
      cam_prog_group_desc.raygen.entryFunctionName = "__raygen__knn";
    else
      cam_prog_group_desc.raygen.entryFunctionName = "__raygen__radius";

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &cam_prog_group_desc,
        1,
        &cam_prog_group_options,
        log,
        &sizeof_log,
        &cam_prog_group ) );

    program_groups.push_back(cam_prog_group);
    state.raygen_prog_group = cam_prog_group;
}

static void createMetalSphereProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    char    log[2048];
    size_t  sizeof_log = sizeof( log );

    OptixProgramGroup           radiance_sphere_prog_group;
    OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
    radiance_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    radiance_sphere_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    if (state.searchMode == "knn")
      radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere_knn";
    else
      radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere_radius";
    radiance_sphere_prog_group_desc.hitgroup.moduleCH               = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.moduleAH               = state.geometry_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = "__anyhit__terminateRay";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_sphere_prog_group_desc,
        1,
        &radiance_sphere_prog_group_options,
        log,
        &sizeof_log,
        &radiance_sphere_prog_group ) );

    program_groups.push_back(radiance_sphere_prog_group);
    state.radiance_metal_sphere_prog_group = radiance_sphere_prog_group;
}

static void createMissProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroupOptions    miss_prog_group_options = {};
    OptixProgramGroupDesc       miss_prog_group_desc = {};
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module             = nullptr;
    miss_prog_group_desc.miss.entryFunctionName  = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        log,
        &sizeof_log,
        &state.radiance_miss_prog_group ) );

    program_groups.push_back(state.radiance_miss_prog_group);
}

void createPipeline( WhittedState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
        8,                                                      // numPayloadValues; need 8 for 7nn search
        0,                                                      // numAttributeValues
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createModules( state );
    createCameraProgram( state, program_groups );
    createMetalSphereProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
        max_trace,                          // maxTraceDepth
        OPTIX_COMPILE_DEBUG_LEVEL_FULL      // debugLevel
    };
    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG( optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        static_cast<unsigned int>( program_groups.size() ),
        log,
        &sizeof_log,
        &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace,
                                             0,  // maxCCDepth
                                             0,  // maxDCDepth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}

void createSBT( WhittedState &state )
{
    // Raygen program record
    {
        CUdeviceptr d_raygen_record;
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_raygen_record ),
            sizeof( RayGenRecord ) ) );

        RayGenRecord rg_sbt;
        optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_raygen_record ),
            &rg_sbt,
            sizeof(rg_sbt),
            cudaMemcpyHostToDevice
        ) );

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof( MissRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_miss_record ),
            sizeof_miss_record*RAY_TYPE_COUNT ) );

        MissRecord ms_sbt;
        optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_miss_record ),
            &ms_sbt,
            sizeof_miss_record,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.missRecordBase          = d_miss_record;
        state.sbt.missRecordCount         = 1;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record
    {
        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_hitgroup_records ),
            sizeof_hitgroup_record
        ) );

        HitGroupRecord hit_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.radiance_metal_sphere_prog_group,
            &hit_sbt ) );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_hitgroup_records ),
            &hit_sbt,
            sizeof_hitgroup_record,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.hitgroupRecordBase            = d_hitgroup_records;
        state.sbt.hitgroupRecordCount           = 1;
        state.sbt.hitgroupRecordStrideInBytes   = static_cast<uint32_t>( sizeof_hitgroup_record );
    }
}

//static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
//{
//    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
//              << message << "\n";
//}

void createContext( WhittedState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    //options.logCallbackFunction       = &context_log_cb;
    options.logCallbackFunction       = nullptr;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}

void launchSubframe( unsigned int* output_buffer, WhittedState& state )
{
    state.params.frame_buffer = output_buffer;

    // note cudamemset sets #count number of BYTES to value.
    CUDA_CHECK( cudaMemsetAsync ( state.params.frame_buffer, 0xFF,
                                  state.numQueries * state.params.limit * sizeof(unsigned int),
                                  state.stream ) );

    state.params.handle = state.gas_handle;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ),
                                 &state.params,
                                 sizeof( Params ),
                                 cudaMemcpyHostToDevice,
                                 state.stream
    ) );

    OPTIX_CHECK( optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>( state.d_params ),
        sizeof( Params ),
        &state.sbt,
        state.numQueries, // launch width
        1,                // launch height
        1                 // launch depth
    ) );
    // leave sync to the caller.
    //CUDA_SYNC_CHECK();
}


void cleanupState( WhittedState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.camera_module           ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer    ) ) );
    if (state.d_r2q_map)
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_r2q_map     ) ) );
    if (state.params.queries != state.params.points)
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.queries       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.points          ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
    if (state.d_key)
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_key                ) ) );

    if (state.h_queries != state.h_points) delete state.h_queries;
    delete state.h_points;
}

//class Compare
//{
//  public:
//    bool operator() (knn_point_t* a, knn_point_t* b)
//    {
//      return a->dist < b->dist;
//    }
//};
//
//typedef std::priority_queue<knn_point_t*, std::vector<knn_point_t*>, Compare> knn_queue;

// TODO: finish it.
void sanityCheck_knn( WhittedState& state, void* data ) {
  for (unsigned int q = 0; q < state.numQueries; q++) {
    for (unsigned int n = 0; n < state.params.limit; n++) {
      unsigned int p = static_cast<unsigned int*>( data )[ q * state.params.limit + n ];
      if (p == UINT_MAX) break;
      else {
        float3 diff = state.h_points[p] - state.h_queries[q];
        float dists = dot(diff, diff);
        std::cout << sqrt(dists) << " ";
      }
      //std::cout << p << " ";
    }
    std::cout << "\n";
  }
  std::cerr << "Sanity check done." << std::endl;
}

void sanityCheck( WhittedState& state, void* data ) {
  // this is stateful in that it relies on state.params.limit

  unsigned int totalNeighbors = 0;
  unsigned int totalWrongNeighbors = 0;
  double totalWrongDist = 0;
  for (unsigned int q = 0; q < state.numQueries; q++) {
    for (unsigned int n = 0; n < state.params.limit; n++) {
      unsigned int p = reinterpret_cast<unsigned int*>( data )[ q * state.params.limit + n ];
      //std::cout << p << std::endl; break;
      if (p == UINT_MAX) break;
      else {
        totalNeighbors++;
        float3 diff = state.h_points[p] - state.h_queries[q];
        float dists = dot(diff, diff);
        if (dists > state.params.radius*state.params.radius) {
          //fprintf(stdout, "Point %u [%f, %f, %f] is not a neighbor of query %u [%f, %f, %f]. Dist is %lf.\n", p, state.h_points[p].x, state.h_points[p].y, state.h_points[p].z, q, state.h_points[q].x, state.h_points[q].y, state.h_points[q].z, sqrt(dists));
          totalWrongNeighbors++;
          totalWrongDist += sqrt(dists);
          //exit(1);
        }
        std::cout << sqrt(dists) << " ";
      }
      //std::cout << p << " ";
    }
    std::cout << "\n";
  }
  std::cerr << "Sanity check done." << std::endl;
  std::cerr << "Avg neighbor/query: " << (float)totalNeighbors/state.numQueries << std::endl;
  std::cerr << "Total wrong neighbors: " << totalWrongNeighbors << std::endl;
  if (totalWrongNeighbors != 0) std::cerr << "Avg wrong dist: " << totalWrongDist / totalWrongNeighbors << std::endl;
}

thrust::device_ptr<unsigned int> sortQueriesByFHCoord( WhittedState& state, thrust::device_ptr<unsigned int> d_firsthit_idx_ptr ) {
  // this is sorting queries by the x/y/z coordinate of the first hit primitives.

  Timing::startTiming("gas-sort queries init");
    // allocate device memory for storing the keys, which will be generated by a gather and used in sort_by_keys
    float* d_key;
    cudaMalloc(reinterpret_cast<void**>(&d_key),
               state.numQueries * sizeof(float) );
    thrust::device_ptr<float> d_key_ptr = thrust::device_pointer_cast(d_key);
    state.d_key = d_key; // just so we have a handle to free it later
  
    // create indices for gather and upload to device
    thrust::host_vector<float> h_orig_points_1d(state.numQueries);
    // TODO: do this in CUDA
    for (unsigned int i = 0; i < state.numQueries; i++) {
      h_orig_points_1d[i] = state.h_points[i].z; // could be other dimensions
    }
    thrust::device_vector<float> d_orig_points_1d = h_orig_points_1d;

    // initialize a sequence to be sorted, which will become the r2q map.
    // TODO: need to free this.
    thrust::device_ptr<unsigned int> d_r2q_map_ptr = genSeqDevice(state.numQueries);
  Timing::stopTiming(true);
  
  Timing::startTiming("gas-sort queries");
    // TODO: do thrust work in a stream: https://forums.developer.nvidia.com/t/thrust-and-streams/53199
    // first use a gather to generate the keys, then sort by keys
    gatherByKey(d_firsthit_idx_ptr, &d_orig_points_1d, d_key_ptr, state.numQueries);
    sortByKey( d_key_ptr, d_r2q_map_ptr, state.numQueries );
    state.d_r2q_map = thrust::raw_pointer_cast(d_r2q_map_ptr);
  Timing::stopTiming(true);
  
  // if debug, copy the sorted keys and values back to host
  bool debug = false;
  if (debug) {
    thrust::host_vector<unsigned int> h_vec_val(state.numQueries);
    thrust::copy(d_r2q_map_ptr, d_r2q_map_ptr+state.numQueries, h_vec_val.begin());

    thrust::host_vector<float> h_vec_key(state.numQueries);
    thrust::copy(d_key_ptr, d_key_ptr+state.numQueries, h_vec_key.begin());
    
    for (unsigned int i = 0; i < h_vec_val.size(); i++) {
      std::cout << h_vec_key[i] << "\t" 
                << h_vec_val[i] << "\t" 
                << state.h_queries[h_vec_val[i]].x << "\t"
                << state.h_queries[h_vec_val[i]].y << "\t"
                << state.h_queries[h_vec_val[i]].z
                << std::endl;
    }
  }

  return d_r2q_map_ptr;
}

thrust::device_ptr<unsigned int> sortQueriesByFHIdx( WhittedState& state, thrust::device_ptr<unsigned int> d_firsthit_idx_ptr ) {
  // this is sorting queries just by the first hit primitive IDs

  // initialize a sequence to be sorted, which will become the r2q map
  Timing::startTiming("gas-sort queries init");
    thrust::device_ptr<unsigned int> d_r2q_map_ptr = genSeqDevice(state.numQueries);
  Timing::stopTiming(true);

  Timing::startTiming("gas-sort queries");
    sortByKey( d_firsthit_idx_ptr, d_r2q_map_ptr, state.numQueries );
    // thrust can't be used in kernel code since NVRTC supports only a
    // limited subset of C++, so we would have to explicitly cast a
    // thrust device vector to its raw pointer. See the problem discussed
    // here: https://github.com/cupy/cupy/issues/3728 and
    // https://github.com/cupy/cupy/issues/3408. See how cuNSearch does it:
    // https://github.com/InteractiveComputerGraphics/cuNSearch/blob/master/src/cuNSearchDeviceData.cu#L152
    state.d_r2q_map = thrust::raw_pointer_cast(d_r2q_map_ptr);
  Timing::stopTiming(true);

  bool debug = false;
  if (debug) {
    thrust::host_vector<unsigned int> h_vec_val(state.numQueries);
    thrust::copy(d_r2q_map_ptr, d_r2q_map_ptr+state.numQueries, h_vec_val.begin());

    thrust::host_vector<unsigned int> h_vec_key(state.numQueries);
    thrust::copy(d_firsthit_idx_ptr, d_firsthit_idx_ptr+state.numQueries, h_vec_key.begin());

    for (unsigned int i = 0; i < h_vec_val.size(); i++) {
      std::cout << h_vec_key[i] << "\t"
                << h_vec_val[i] << "\t"
                << state.h_queries[h_vec_val[i]].x << "\t"
                << state.h_queries[h_vec_val[i]].y << "\t"
                << state.h_queries[h_vec_val[i]].z
                << std::endl;
    }
  }

  return d_r2q_map_ptr;
}

void gatherQueries( WhittedState& state, thrust::device_ptr<unsigned int> d_indices_ptr ) {
  // Perform a device gather before launching the actual search, which by
  // itself is not useful, since we access each query only once (in the RG
  // program) anyways. in reality we see little gain by gathering queries. but
  // if queries and points point to the same device memory, gathering queries
  // effectively reorders the points too. we access points in the IS program
  // (get query origin using the hit primIdx), and so it would be nice to
  // coalesce memory by reordering points. but note two things. First, we
  // access only one point and only once in each IS program and the bulk of
  // memory access is to the BVH which is out of our control, so better memory
  // coalescing has less effect than in traditional grid search. Second, if the
  // points are already sorted in a good order (raster scan or z-order), this
  // reordering has almost zero effect. empirically, we get 10% search time
  // reduction for large point clouds and the points originally are poorly
  // ordered. but this comes at a chilling overhead that we need to rebuild the
  // GAS (to make sure the ID of a box in GAS is the ID of the sphere in device
  // memory; otherwise IS program is in correct), which is on the critical path
  // and whose overhead can't be hidden. so almost always this optimization
  // leads to performance degradation, both toGather and reorderPoints are
  // disabled by default.

  Timing::startTiming("gather queries");
    // allocate device memory for reordered/gathered queries
    float3* d_reordered_queries;
    // TODO: change this to thrust device vector
    cudaMalloc(reinterpret_cast<void**>(&d_reordered_queries),
               state.numQueries * sizeof(float3) );
    thrust::device_ptr<float3> d_reord_queries_ptr = thrust::device_pointer_cast(d_reordered_queries);

    // get pointer to original queries in device memory
    thrust::device_ptr<float3> d_orig_queries_ptr = thrust::device_pointer_cast(state.params.queries);

    // gather by key, which is generated by the previous sort
    gatherByKey(d_indices_ptr, d_orig_queries_ptr, d_reord_queries_ptr, state.numQueries);

    // if not samepq, then we can free the original query device memory
    if (!state.samepq) CUDA_CHECK( cudaFree( (void*)state.params.queries ) );
    state.params.queries = thrust::raw_pointer_cast(d_reord_queries_ptr);
    assert(state.params.points != state.params.queries);
  Timing::stopTiming(true);

  // Copy reordered queries to host for sanity check
  thrust::host_vector<float3> host_reord_queries(state.numQueries);
  // if not samepq, free the original query host memory first
  if (!state.samepq) delete state.h_queries;
  state.h_queries = new float3[state.numQueries]; // don't overwrite h_points
  thrust::copy(d_reord_queries_ptr, d_reord_queries_ptr+state.numQueries, state.h_queries);
  assert (state.h_points != state.h_queries);

  // if samepq, we could try reordering points according to the new query
  // layout. see caveats in the note above.
  if (state.samepq && state.reorderPoints) {
    // will rebuild the GAS later
    delete state.h_points;
    state.h_points = state.h_queries;
    CUDA_CHECK( cudaFree( (void*)state.params.points ) );
    state.params.points = state.params.queries;
  }
}

void parseArgs( WhittedState& state,  int argc, char* argv[] ) {
  for( int i = 1; i < argc; ++i )
  {
      const std::string arg = argv[i];
      if( arg == "--help" || arg == "-h" )
      {
          printUsageAndExit( argv[0] );
      }
      else if( arg == "--pfile" || arg == "-f" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.pfile = argv[++i];
      }
      else if( arg == "--qfile" || arg == "-q" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.qfile = argv[++i];
      }
      else if( arg == "--knn" || arg == "-k" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.params.knn = atoi(argv[++i]);
      }
      else if( arg == "--searchmode" || arg == "-sm" ) // need to be after --knn so that we can overwrite params.knn if needed
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.searchMode = argv[++i];
          if ((state.searchMode.compare("knn") != 0) && (state.searchMode.compare("radius") != 0))
              printUsageAndExit( argv[0] );
      }
      else if( arg == "--radius" || arg == "-r" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.params.radius = std::stof(argv[++i]);
      }
      else if( arg == "--samepq" || arg == "-spq" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.samepq = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--qgassort" || arg == "-s" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.qGasSortMode = atoi(argv[++i]);
      }
      else if( arg == "--pointsort" || arg == "-ps" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.pointSortMode = atoi(argv[++i]);
      }
      else if( arg == "--querysort" || arg == "-qs" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.querySortMode = atoi(argv[++i]);
      }
      else if( arg == "--crratio" || arg == "-cr" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.crRatio = std::stof(argv[++i]);
      }
      else if( arg == "--gather" || arg == "-g" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.toGather = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--reorderpoints" || arg == "-rp" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.reorderPoints = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--sortingGAS" || arg == "-sg" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.sortingGAS = std::stof(argv[++i]);
          if (state.sortingGAS <= 0)
              printUsageAndExit( argv[0] );
      }
      else
      {
          std::cerr << "Unknown option '" << argv[i] << "'\n";
          printUsageAndExit( argv[0] );
      }
  }

  if (state.searchMode.compare("knn") == 0) {
    state.params.knn = K; // a macro
  }
}

void setupCUDA( WhittedState& state, int32_t device_id ) {
  int32_t device_count = 0;
  CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
  std::cerr << "\tTotal GPUs visible: " << device_count << std::endl;
  
  cudaDeviceProp prop;
  CUDA_CHECK( cudaGetDeviceProperties ( &prop, device_id ) );
  CUDA_CHECK( cudaSetDevice( device_id ) );
  std::cerr << "\tUsing [" << device_id << "]: " << prop.name << std::endl;

  CUDA_CHECK( cudaStreamCreate( &state.stream ) );
}

void computeMinMax(WhittedState& state, ParticleType type)
{
  unsigned int N;
  float3* particles;
  if (type == POINT) {
    N = state.numPoints;
    particles = state.params.points;
  } else {
    N = state.numQueries;
    particles = state.params.queries;
  }

  // TODO: maybe use long since we are going to convert a float to its floor value?
  thrust::host_vector<int3> h_MinMax(2);
  h_MinMax[0] = make_int3(std::numeric_limits<int>().max(), std::numeric_limits<int>().max(), std::numeric_limits<int>().max());
  h_MinMax[1] = make_int3(std::numeric_limits<int>().min(), std::numeric_limits<int>().min(), std::numeric_limits<int>().min());

  thrust::device_vector<int3> d_MinMax = h_MinMax;

  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = N / threadsPerBlock + 1;
  // compare only the ints since atomicAdd has only int version
  kComputeMinMax(numOfBlocks,
                 threadsPerBlock,
                 particles,
                 N,
                 thrust::raw_pointer_cast(&d_MinMax[0]),
                 thrust::raw_pointer_cast(&d_MinMax[1])
                 );

  h_MinMax = d_MinMax;

  // minCell encloses the scene but maxCell doesn't (floor and int in the kernel) so increment by 1 to enclose the scene.
  // TODO: consider minus 1 for minCell too to avoid the numerical precision issue
  int3 minCell = h_MinMax[0];
  int3 maxCell = h_MinMax[1] + make_int3(1, 1, 1);
 
  state.Min.x = minCell.x;
  state.Min.y = minCell.y;
  state.Min.z = minCell.z;
 
  state.Max.x = maxCell.x;
  state.Max.y = maxCell.y;
  state.Max.z = maxCell.z;

  //fprintf(stdout, "\tcell boundary: (%d, %d, %d), (%d, %d, %d)\n", minCell.x, minCell.y, minCell.z, maxCell.x, maxCell.y, maxCell.z);
  //fprintf(stdout, "\tscene boundary: (%f, %f, %f), (%f, %f, %f)\n", state.Min.x, state.Min.y, state.Min.z, state.Max.x, state.Max.y, state.Max.z);
}

void gridSort(WhittedState& state, ParticleType type, bool morton) {
  unsigned int N;
  float3* particles;
  float3* h_particles;
  if (type == POINT) {
    N = state.numPoints;
    particles = state.params.points;
    h_particles = state.h_points;
  } else {
    N = state.numQueries;
    particles = state.params.queries;
    h_particles = state.h_queries;
  }

  float3 sceneMin = state.Min;
  float3 sceneMax = state.Max;

  GridInfo gridInfo;
  gridInfo.ParticleCount = N;
  gridInfo.GridMin = sceneMin;

  float cellSize = state.params.radius/state.crRatio; // TODO: change cellSize as a input parameter to the function
  float3 gridSize = sceneMax - sceneMin;
  gridInfo.GridDimension.x = static_cast<unsigned int>(ceilf(gridSize.x / cellSize));
  gridInfo.GridDimension.y = static_cast<unsigned int>(ceilf(gridSize.y / cellSize));
  gridInfo.GridDimension.z = static_cast<unsigned int>(ceilf(gridSize.z / cellSize));

  // Adjust grid size to multiple of cell size
  gridSize.x = gridInfo.GridDimension.x * cellSize;
  gridSize.y = gridInfo.GridDimension.y * cellSize;
  gridSize.z = gridInfo.GridDimension.z * cellSize;

  gridInfo.GridDelta.x = gridInfo.GridDimension.x / gridSize.x;
  gridInfo.GridDelta.y = gridInfo.GridDimension.y / gridSize.y;
  gridInfo.GridDelta.z = gridInfo.GridDimension.z / gridSize.z;

  // morton code can only be correctly calcuated for a cubic, where each
  // dimension is of the same size. currently we generate the largely meta_grid
  // possible, which would divice the entire grid into multiple meta grids.
  gridInfo.meta_grid_dim = std::min({gridInfo.GridDimension.x, gridInfo.GridDimension.y, gridInfo.GridDimension.z});
  gridInfo.meta_grid_size = gridInfo.meta_grid_dim * gridInfo.meta_grid_dim * gridInfo.meta_grid_dim;

  // One meta grid cell contains meta_grid_dim^3 cells. The morton curve is
  // calculated for each metagrid, and the order of metagrid is raster order.
  // So if meta_grid_dim is 1, this is basically the same as raster order
  // across all cells. If meta_grid_dim is the same as GridDimension, this
  // calculates one single morton curve for the entire grid.
  gridInfo.MetaGridDimension.x = static_cast<unsigned int>(ceilf(gridInfo.GridDimension.x / (float)gridInfo.meta_grid_dim));
  gridInfo.MetaGridDimension.y = static_cast<unsigned int>(ceilf(gridInfo.GridDimension.y / (float)gridInfo.meta_grid_dim));
  gridInfo.MetaGridDimension.z = static_cast<unsigned int>(ceilf(gridInfo.GridDimension.z / (float)gridInfo.meta_grid_dim));

  // metagrids will slightly increase the total cells
  unsigned int numberOfCells = (gridInfo.MetaGridDimension.x * gridInfo.MetaGridDimension.y * gridInfo.MetaGridDimension.z) * gridInfo.meta_grid_size;
  //fprintf(stdout, "\tGrid dimension (without meta grids): %u, %u, %u\n", gridInfo.GridDimension.x, gridInfo.GridDimension.y, gridInfo.GridDimension.z);
  //fprintf(stdout, "\tGrid dimension (with meta grids): %u, %u, %u\n", gridInfo.MetaGridDimension.x * gridInfo.meta_grid_dim, gridInfo.MetaGridDimension.y * gridInfo.meta_grid_dim, gridInfo.MetaGridDimension.z * gridInfo.meta_grid_dim);
  //fprintf(stdout, "\tMeta Grid dimension: %u, %u, %u\n", gridInfo.MetaGridDimension.x, gridInfo.MetaGridDimension.y, gridInfo.MetaGridDimension.z);
  //fprintf(stdout, "\tLength of a meta grid: %u\n", gridInfo.meta_grid_dim);
  fprintf(stdout, "\tNumber of cells: %u\n", numberOfCells);
  fprintf(stdout, "\tCell size: %f\n", cellSize);
 
  thrust::device_ptr<unsigned int> d_ParticleCellIndices_ptr = getThrustDevicePtr(N);
  thrust::device_ptr<unsigned int> d_CellParticleCounts_ptr = getThrustDevicePtr(numberOfCells); // this takes a lot of memory
  fillByValue(d_CellParticleCounts_ptr, numberOfCells, 0);
  thrust::device_ptr<unsigned int> d_LocalSortedIndices_ptr = getThrustDevicePtr(N);

  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = N / threadsPerBlock + 1;
  kInsertParticles(numOfBlocks,
                   threadsPerBlock,
                   gridInfo,
                   particles,
                   thrust::raw_pointer_cast(d_ParticleCellIndices_ptr),
                   thrust::raw_pointer_cast(d_CellParticleCounts_ptr),
                   thrust::raw_pointer_cast(d_LocalSortedIndices_ptr),
                   morton
                   );

  bool debug = false;
  if (debug) {
    thrust::host_vector<unsigned int> temp(numberOfCells);
    thrust::copy(d_CellParticleCounts_ptr, d_CellParticleCounts_ptr + numberOfCells, temp.begin());
    for (unsigned int i = 0; i < numberOfCells; i++) {
      fprintf(stdout, "%u\n", temp[i]);
    }
  }

  thrust::device_ptr<unsigned int> d_CellOffsets_ptr = getThrustDevicePtr(numberOfCells);
  fillByValue(d_CellOffsets_ptr, numberOfCells, 0); // need to initialize it even for exclusive scan
  exclusiveScan(d_CellParticleCounts_ptr, numberOfCells, d_CellOffsets_ptr);

  thrust::device_ptr<unsigned int> d_posInSortedPoints_ptr = getThrustDevicePtr(N);
  kCountingSortIndices(numOfBlocks,
                       threadsPerBlock,
                       gridInfo,
                       thrust::raw_pointer_cast(d_ParticleCellIndices_ptr),
                       thrust::raw_pointer_cast(d_CellOffsets_ptr),
                       thrust::raw_pointer_cast(d_LocalSortedIndices_ptr),
                       thrust::raw_pointer_cast(d_posInSortedPoints_ptr)
                       );

  // in-place sort; no new device memory is allocated
  sortByKey(d_posInSortedPoints_ptr, thrust::device_pointer_cast(particles), N);

  // TODO: do this in a stream
  thrust::device_ptr<float3> d_particles_ptr = thrust::device_pointer_cast(particles);
  thrust::copy(d_particles_ptr, d_particles_ptr + N, h_particles);

  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_ParticleCellIndices_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_posInSortedPoints_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_CellOffsets_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_LocalSortedIndices_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_CellParticleCounts_ptr) ) );

  debug = false;
  if (debug) {
    thrust::host_vector<uint> temp(N);
    thrust::copy(d_posInSortedPoints_ptr, d_posInSortedPoints_ptr + N, temp.begin());
    for (unsigned int i = 0; i < N; i++) {
      fprintf(stdout, "%u (%f, %f, %f)\n", temp[i], h_particles[i].x, h_particles[i].y, h_particles[i].z);
    }
  }
}

void sortParticles ( WhittedState& state, ParticleType type, int sortMode ) {
  if (!sortMode) return;

  // the semantices of the two sort functions are: sort data in device, and copy the sorted data back to host.
  std::string typeName = ((type == POINT) ? "points" : "queries");
  Timing::startTiming("sort " + typeName);
    if (sortMode == 3) {
      oneDSort(state, type);
    } else {
      computeMinMax(state, type);

      bool morton; // false for raster order
      if (sortMode == 1) morton = true;
      else {
        assert(sortMode == 2);
        morton = false;
      }
      gridSort(state, type, morton);
    }
  Timing::stopTiming(true);
}

void setupOptiX( WhittedState& state ) {
  Timing::startTiming("create context");
    createContext  ( state );
  Timing::stopTiming(true);

  // creating GAS can be done async with the rest two.
  Timing::startTiming("create and upload geometry");
    createGeometry ( state, state.sortingGAS );
  Timing::stopTiming(true);
 
  Timing::startTiming("create pipeline");
    createPipeline ( state );
  Timing::stopTiming(true);

  Timing::startTiming("create SBT");
    createSBT      ( state );
  Timing::stopTiming(true);
}

void nonsortedSearch( WhittedState& state, int32_t device_id ) {
  Timing::startTiming("total search time");
    Timing::startTiming("search compute");
      state.params.limit = state.params.knn;
      thrust::device_ptr<unsigned int> output_buffer = getThrustDevicePtr(state.numQueries * state.params.limit);

      assert((state.h_queries == state.h_points) ^ !state.samepq);
      assert((state.params.points == state.params.queries) ^ !state.samepq);
      assert(state.params.d_r2q_map == nullptr);
      launchSubframe( thrust::raw_pointer_cast(output_buffer), state );
      CUDA_CHECK( cudaStreamSynchronize( state.stream ) ); // TODO: just so we can measure time
    Timing::stopTiming(true);

    // cudaMallocHost is time consuming; must be hidden behind async launch
    Timing::startTiming("result copy D2H");
      void* data;
      cudaMallocHost(reinterpret_cast<void**>(&data), state.numQueries * state.params.limit * sizeof(unsigned int));

      // TODO: can a thrust copy
      CUDA_CHECK( cudaMemcpyAsync(
                      static_cast<void*>( data ),
                      thrust::raw_pointer_cast(output_buffer),
                      state.numQueries * state.params.limit * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost,
                      state.stream
                      ) );
      CUDA_CHECK( cudaStreamSynchronize( state.stream ) ); // TODO: just so we can measure time
    Timing::stopTiming(true);
  Timing::stopTiming(true);

  if (state.searchMode == "radius") sanityCheck( state, data );
  else sanityCheck_knn( state, data );
  CUDA_CHECK( cudaFreeHost(data) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(output_buffer) ) );
}

void searchTraversal(WhittedState& state, int32_t device_id) {
  Timing::startTiming("total search time");
    // create a new GAS if the sorting GAS is different, but we reordered points using the query order
    if ( (state.sortingGAS != 1) || (state.samepq && state.toGather && state.reorderPoints) ) {
      Timing::startTiming("create search GAS");
        createGeometry ( state, 1.0 );
        CUDA_CHECK( cudaStreamSynchronize( state.stream ) );
      Timing::stopTiming(true);
    }

    Timing::startTiming("search compute");
      state.params.limit = state.params.knn;
      thrust::device_ptr<unsigned int> output_buffer = getThrustDevicePtr(state.numQueries * state.params.limit);

      // TODO: this is just awkward. maybe we should just get rid of the gather mode and directl assign to params.d_r2q_map.
      assert(state.params.d_r2q_map == nullptr);
      // TODO: not sure why, but directly assigning state.params.d_r2q_map in sort routines has a huge perf hit.
      if (!state.toGather) state.params.d_r2q_map = state.d_r2q_map;

      launchSubframe( thrust::raw_pointer_cast(output_buffer), state );
      CUDA_CHECK( cudaStreamSynchronize( state.stream ) ); // comment this out for e2e measurement.
    Timing::stopTiming(true);

    Timing::startTiming("result copy D2H");
      void* data;
      cudaMallocHost(reinterpret_cast<void**>(&data), state.numQueries * state.params.limit * sizeof(unsigned int));

      CUDA_CHECK( cudaMemcpyAsync(
                      static_cast<void*>( data ),
                      thrust::raw_pointer_cast(output_buffer),
                      state.numQueries * state.params.limit * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost,
                      state.stream
                      ) );
      CUDA_CHECK( cudaStreamSynchronize( state.stream ) );
    Timing::stopTiming(true);
  Timing::stopTiming(true);

  if (state.searchMode == "radius") sanityCheck( state, data );
  else sanityCheck_knn( state, data );
  CUDA_CHECK( cudaFreeHost(data) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(output_buffer) ) );
}

thrust::device_ptr<unsigned int> initialTraversal(WhittedState& state, int32_t device_id) {
  Timing::startTiming("initial traversal");
    state.params.limit = 1;
    thrust::device_ptr<unsigned int> output_buffer = getThrustDevicePtr(state.numQueries * state.params.limit);

    assert((state.h_queries == state.h_points) ^ !state.samepq);
    assert((state.params.points == state.params.queries) ^ !state.samepq);
    assert(state.params.d_r2q_map == nullptr);

    launchSubframe( thrust::raw_pointer_cast(output_buffer), state );
    // TODO: could delay this until sort, but initial traversal is lightweight anyways
    CUDA_CHECK( cudaStreamSynchronize( state.stream ) );
  Timing::stopTiming(true);

  return output_buffer;
}

void readData(WhittedState& state) {
  // p and q files being the same dones't mean samepq have to be true. we can
  // still set it to be false to evaluate different reordering policies on
  // points and queries separately.

  state.h_points = read_pc_data(state.pfile.c_str(), &state.numPoints);
  //state.h_ndpoints = read_pc_data(state.pfile.c_str(), &state.numPoints, &state.dim);
  state.numQueries = state.numPoints;
  if (state.samepq) state.h_queries = state.h_points;
  else {
    state.h_queries = (float3*)malloc(state.numQueries * sizeof(float3));
    thrust::copy(state.h_points, state.h_points+state.numQueries, state.h_queries);
  }

  if (!state.qfile.empty()) {
    state.h_queries = read_pc_data(state.qfile.c_str(), &state.numQueries);
    assert(state.h_points != state.h_queries);
    // overwrite the samepq option from commandline
    state.samepq = false;

    //int query_dim;
    //state.h_ndqueries = read_pc_data(state.qfile.c_str(), &state.numQueries, &query_dim);
    //assert(query_dim == state.dim);
  }
}

int main( int argc, char* argv[] )
{
  WhittedState state;
  state.params.radius = 2;
  state.params.knn = 50;

  parseArgs( state, argc, argv );

  readData(state);

  std::cout << "========================================" << std::endl;
  std::cout << "numPoints: " << state.numPoints << std::endl;
  std::cout << "numQueries: " << state.numQueries << std::endl;
  std::cout << "searchMode: " << state.searchMode << std::endl;
  std::cout << "radius: " << state.params.radius << std::endl;
  std::cout << "K: " << state.params.knn << std::endl;
  std::cout << "Same P and Q? " << std::boolalpha << state.samepq << std::endl;
  std::cout << "qGasSortMode: " << state.qGasSortMode << std::endl;
  std::cout << "pointSortMode: " << std::boolalpha << state.pointSortMode << std::endl;
  std::cout << "querySortMode: " << std::boolalpha << state.querySortMode << std::endl;
  std::cout << "cellRadiusRatio: " << std::boolalpha << state.crRatio << std::endl; // only useful when preSort == 1/2
  std::cout << "sortingGAS: " << state.sortingGAS << std::endl; // only useful when qGasSortMode != 0
  std::cout << "Gather? " << std::boolalpha << state.toGather << std::endl;
  std::cout << "reorderPoints? " << std::boolalpha << state.reorderPoints << std::endl; // only useful under samepq and toGather
  std::cout << "========================================" << std::endl << std::endl;

  try
  {
    // Set up CUDA device and stream
    int32_t device_id = 1;
    setupCUDA(state, device_id);

    Timing::reset();

    uploadData(state);
    sortParticles(state, POINT, state.pointSortMode);

    // Set up OptiX state, which includes creating the GAS (using the current order of points).
    setupOptiX(state);

    initLaunchParams( state );

    // when samepq, queries are sorted using the point sort mode.
    if (!state.samepq) sortParticles(state, QUERY, state.querySortMode);

    if (!state.qGasSortMode) {
      nonsortedSearch(state, device_id);
    } else {
      // Initial traversal (to sort the queries)
      thrust::device_ptr<unsigned int> d_firsthit_idx_ptr = initialTraversal(state, device_id);

      // Sort the queries
      thrust::device_ptr<unsigned int> d_indices_ptr;
      if (state.qGasSortMode == 1)
        d_indices_ptr = sortQueriesByFHCoord(state, d_firsthit_idx_ptr);
      else if (state.qGasSortMode == 2)
        d_indices_ptr = sortQueriesByFHIdx(state, d_firsthit_idx_ptr);
      else assert(0);
      CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_firsthit_idx_ptr) ) );

      if (state.toGather) {
        gatherQueries( state, d_indices_ptr );
      }

      // Actual traversal with sorted queries
      searchTraversal(state, device_id);
    }

    cleanupState( state );
  }
  catch( std::exception& e )
  {
      std::cerr << "Caught exception: " << e.what() << "\n";
      return 1;
  }

  return 0;
}
