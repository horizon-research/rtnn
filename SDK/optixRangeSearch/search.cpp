#include <sutil/Exception.h>
#include <sutil/Timing.h>
#include <thrust/device_vector.h>

#include "optixRangeSearch.h"
#include "state.h"
#include "func.h"

void search(WhittedState& state, int batch_id) {
  Timing::startTiming("batch search time");
    Timing::startTiming("search compute");
      state.params.limit = state.params.knn;
      thrust::device_ptr<unsigned int> output_buffer = getThrustDevicePtr(state.numQueries * state.params.limit);

      if (state.qGasSortMode && !state.toGather) state.params.d_r2q_map = state.d_r2q_map;
      else state.params.d_r2q_map = nullptr; // if no GAS-sorting or has done gather, this map is null.

      state.params.isApprox = false;
      // approximate in the first batch of radius search. can't approximate in the knn search.
      // TODO: change it when the batch order changes.
      if ((state.searchMode == "radius") && state.partition && !batch_id) state.params.isApprox = true;

      // TODO: revisit this. create a new GAS if the sorting GAS is different,
      // or if we want to reorder points using the gas-sorted query order.
      //if ( (state.sortingGAS != 1) || (state.samepq && state.toGather && state.reorderPoints) ) {
      //  Timing::startTiming("create search GAS");
      //    createGeometry ( state );
      //    CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) );
      //  Timing::stopTiming(true);
      //}

      launchSubframe( thrust::raw_pointer_cast(output_buffer), state, batch_id );
      OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) ); // TODO: just so we can measure time
    Timing::stopTiming(true);

    // cudaMallocHost is time consuming; must be hidden behind async launch
    Timing::startTiming("result copy D2H");
      void* data;
      cudaMallocHost(reinterpret_cast<void**>(&data), state.numQueries * state.params.limit * sizeof(unsigned int));

      // TODO: do a thrust copy
      CUDA_CHECK( cudaMemcpyAsync(
                      static_cast<void*>( data ),
                      thrust::raw_pointer_cast(output_buffer),
                      state.numQueries * state.params.limit * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost,
                      state.stream[batch_id]
                      ) );
      OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) ); // TODO: just so we can measure time
    Timing::stopTiming(true);
  Timing::stopTiming(true);

  if (state.searchMode == "radius") OMIT_ON_E2EMSR( sanityCheck( state, data ) );
  else OMIT_ON_E2EMSR( sanityCheck_knn( state, data ) );
  CUDA_CHECK( cudaFreeHost(data) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(output_buffer) ) );
}

thrust::device_ptr<unsigned int> initialTraversal(WhittedState& state, int batch_id) {
  Timing::startTiming("initial traversal");
    state.params.limit = 1;
    thrust::device_ptr<unsigned int> output_buffer = getThrustDevicePtr(state.numQueries * state.params.limit);

    state.params.d_r2q_map = nullptr; // contains the index to reorder rays
    state.params.isApprox = true;

    launchSubframe( thrust::raw_pointer_cast(output_buffer), state, batch_id );
    // TODO: could delay this until sort, but initial traversal is lightweight anyways
    OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) ); // TODO: just so we can measure time
  Timing::stopTiming(true);

  return output_buffer;
}

void gasSortSearch(WhittedState& state, int batch_id) {
  // Initial traversal to aggregate the queries
  thrust::device_ptr<unsigned int> d_firsthit_idx_ptr = initialTraversal(state, batch_id);

  // Sort the queries
  thrust::device_ptr<unsigned int> d_indices_ptr;
  if (state.qGasSortMode == 1)
    d_indices_ptr = sortQueriesByFHCoord(state, d_firsthit_idx_ptr, batch_id);
  else if (state.qGasSortMode == 2)
    d_indices_ptr = sortQueriesByFHIdx(state, d_firsthit_idx_ptr, batch_id);
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_firsthit_idx_ptr) ) );

  if (state.toGather)
    gatherQueries( state, d_indices_ptr, batch_id );
}
