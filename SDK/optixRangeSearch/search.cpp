#include <sutil/Exception.h>
#include <sutil/Timing.h>
#include <thrust/device_vector.h>

#include "optixRangeSearch.h"
#include "state.h"
#include "func.h"

void nonsortedSearch( WhittedState& state) {
  Timing::startTiming("total search time");
    Timing::startTiming("search compute");
      state.params.limit = state.params.knn;
      thrust::device_ptr<unsigned int> output_buffer = getThrustDevicePtr(state.numQueries * state.params.limit);

      // TODO: not true if partition is enabled
      //assert((state.h_queries == state.h_points) ^ !state.samepq);
      //assert((state.params.points == state.params.queries) ^ !state.samepq);
      //assert(state.params.d_r2q_map == nullptr);

      state.params.d_r2q_map = nullptr; // contains the index to reorder rays

      // TODO: for radius search if the AABB is enclosed by the sphere we can safely approx it the search.
      state.params.isApprox = false;
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

void searchTraversal(WhittedState& state) {
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

      // TODO: this is just awkward. maybe we should just get rid of the gather mode and directly assign to params.d_r2q_map.
      //assert(state.params.d_r2q_map == nullptr);
      // TODO: not sure why, but directly assigning state.params.d_r2q_map in sort routines has a huge perf hit.
      if (!state.toGather) state.params.d_r2q_map = state.d_r2q_map;
      else state.params.d_r2q_map = nullptr;

      // TODO: for radius search if the AABB is enclosed by the sphere we can safely approx it the search.
      state.params.isApprox = false;
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

thrust::device_ptr<unsigned int> initialTraversal(WhittedState& state) {
  Timing::startTiming("initial traversal");
    state.params.limit = 1;
    thrust::device_ptr<unsigned int> output_buffer = getThrustDevicePtr(state.numQueries * state.params.limit);

    // TODO: not true if partition is enabled
    //assert((state.h_queries == state.h_points) ^ !state.samepq);
    //assert((state.params.points == state.params.queries) ^ !state.samepq);
    //assert(state.params.d_r2q_map == nullptr);

    state.params.d_r2q_map = nullptr; // contains the index to reorder rays

    state.params.isApprox = true;
    launchSubframe( thrust::raw_pointer_cast(output_buffer), state );
    // TODO: could delay this until sort, but initial traversal is lightweight anyways
    CUDA_CHECK( cudaStreamSynchronize( state.stream ) );
  Timing::stopTiming(true);

  return output_buffer;
}


