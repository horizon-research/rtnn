#include <sutil/Exception.h>
#include <sutil/Timing.h>
#include <thrust/device_vector.h>

#include "optixNSearch.h"
#include "state.h"
#include "func.h"

void search(RTNNState& state, int batch_id) {
  Timing::startTiming("batch search time");
    Timing::startTiming("search compute");
      unsigned int numQueries = state.numActQueries[batch_id];

      state.params.limit = state.knn;
      thrust::device_ptr<unsigned int> output_buffer;
      allocThrustDevicePtr(&output_buffer, numQueries * state.params.limit, &state.d_pointers);
      // unused slots will become UINT_MAX
      fillByValue(output_buffer, numQueries * state.params.limit, UINT_MAX);

      if (state.qGasSortMode && !state.toGather) state.params.d_r2q_map = state.d_r2q_map[batch_id];
      else state.params.d_r2q_map = nullptr; // if no GAS-sorting or has done gather, this map is null.

      state.params.mode = PRECISE;
      if ((state.searchMode == "radius") && state.partition && (batch_id < state.numOfBatches - 1)) {
        // note that hardware AABB test during traversal in the current OptiX
        // implementation is inherently approximate, so if we want to guarantee
        // that a point is inside an AABB we still have to do an explicit aabb
        // test (instead of using NOTEST). see:
        // https://forums.developer.nvidia.com/t/numerical-imprecision-in-intersection-test/183665/4.

        // in radius mode use AABBTEST except for the last batch. see how the
        // launchRadius is calculated in the |genBatches| function. AABBTEST is
        // faster than PRECISE since sphere test is much more costly then aabb test.
        state.params.mode = AABBTEST;
      }

      state.params.radius = state.launchRadius[batch_id];

      launchSubframe( thrust::raw_pointer_cast(output_buffer), state, batch_id );
      OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) );
    Timing::stopTiming(true);

    Timing::startTiming("result copy D2H");
      void* data;
      cudaMallocHost(reinterpret_cast<void**>(&data), numQueries * state.params.limit * sizeof(unsigned int));
      state.h_res[batch_id] = data;

      CUDA_CHECK( cudaMemcpyAsync(
                      static_cast<void*>( data ),
                      thrust::raw_pointer_cast(output_buffer),
                      numQueries * state.params.limit * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost,
                      state.stream[batch_id]
                      ) );
      OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) );
    Timing::stopTiming(true);
  Timing::stopTiming(true);

  // this frees device memory but will block until the previous optix launch finish and the res is written back.
  //CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(output_buffer) ) );
}

thrust::device_ptr<unsigned int> initialTraversal(RTNNState& state, int batch_id) {
  Timing::startTiming("initial traversal");
    unsigned int numQueries = state.numActQueries[batch_id];

    state.params.limit = 1;
    thrust::device_ptr<unsigned int> output_buffer;
    allocThrustDevicePtr(&output_buffer, numQueries * state.params.limit, &state.d_pointers);
    // for initial sort fill with 0. it's possible that a query has no
    // neighbors (no intersection with any of the AABB), in which case during
    // gas-sort using FHCoord, gather might use UINT_MAX as a key if filled
    // with UINT_MAX.
    fillByValue(output_buffer, numQueries * state.params.limit, 0);

    state.params.d_r2q_map = nullptr; // contains the index to reorder rays
    state.params.mode = NOTEST;
    state.params.radius = state.launchRadius[batch_id]; // doesn't quite matter since we never check radius in approx mode

    launchSubframe( thrust::raw_pointer_cast(output_buffer), state, batch_id );
    // TODO: could delay this until sort, but initial traversal is lightweight anyways
    OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) );
  Timing::stopTiming(true);

  return output_buffer;
}

void gasSortSearch(RTNNState& state, int batch_id) {
  // TODO: maybe we should have a third mode where we sort FH primitives in
  // z-order or raster order. This would improve the performance when no
  // pre-sorting is done, and might even out-perform it since we are sorting
  // fewer points.

  // Initial traversal to aggregate the queries
  thrust::device_ptr<unsigned int> d_firsthit_idx_ptr = initialTraversal(state, batch_id);

  // Generate the GAS-sorted query order
  thrust::device_ptr<unsigned int> d_indices_ptr;
  if (state.qGasSortMode == 1)
    d_indices_ptr = sortQueriesByFHCoord(state, d_firsthit_idx_ptr, batch_id);
  else if (state.qGasSortMode == 2)
    d_indices_ptr = sortQueriesByFHIdx(state, d_firsthit_idx_ptr, batch_id);

  // Actually sort queries in memory if toGather is enabled
  if (state.toGather)
    gatherQueries( state, d_indices_ptr, batch_id );
}
