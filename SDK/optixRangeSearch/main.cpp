#include <cuda_runtime.h>

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
#include <queue>
#include <unordered_set>

#include "optixRangeSearch.h"
#include "state.h"
#include "func.h"
#include "grid.h"

void setDevice ( WhittedState& state ) {
  int32_t device_count = 0;
  CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
  std::cerr << "\tTotal GPUs visible: " << device_count << std::endl;
  
  cudaDeviceProp prop;
  CUDA_CHECK( cudaGetDeviceProperties ( &prop, state.device_id ) );
  CUDA_CHECK( cudaSetDevice( state.device_id ) );
  std::cerr << "\tUsing [" << state.device_id << "]: " << prop.name << std::endl;
}

void initBatches(WhittedState& state) {
  // see |genMask| for the logic behind this.
  float cellSize = state.radius / state.crRatio;
  float maxWidth = state.radius / sqrt(2) * 2;
  int maxIter = (int)floorf(maxWidth / (2 * cellSize) - 1);
  int maxBatchCount = maxIter + 2; // could be fewer than this.
  state.maxBatchCount = maxBatchCount;

  state.gas_handle = new OptixTraversableHandle[maxBatchCount];
  state.d_gas_output_buffer = new CUdeviceptr[maxBatchCount];
  state.stream = new cudaStream_t[maxBatchCount];
  state.d_r2q_map = new unsigned int*[maxBatchCount];
  state.numActQueries = new unsigned int[maxBatchCount];
  state.launchRadius = new float[maxBatchCount];
  state.partThd = new float[maxBatchCount];
  state.h_res = new void*[maxBatchCount];
  state.d_actQs = new float3*[maxBatchCount];
  state.h_actQs = new float3*[maxBatchCount];
  state.d_aabb = new void*[maxBatchCount];
  state.d_firsthit_idx = new void*[maxBatchCount];
  state.d_temp_buffer_gas = new void*[maxBatchCount];
  state.d_buffer_temp_output_gas_and_compacted_size = new void*[maxBatchCount];
  state.pipeline = new OptixPipeline[maxBatchCount];

  for (unsigned int i = 0; i < maxBatchCount; i++)
      CUDA_CHECK( cudaStreamCreate( &state.stream[i] ) );
}

void setupSearch( WhittedState& state ) {
  if (!state.partition) {
    assert(state.numOfBatches == 1);
    initBatches(state);

    state.numActQueries[0] = state.numQueries;
    state.d_actQs[0] = state.params.queries;
    state.h_actQs[0] = state.h_queries;
    state.launchRadius[0] = state.radius;
  }
}

int main( int argc, char* argv[] )
{
  WhittedState state;

  parseArgs( state, argc, argv );

  readData(state);

  initBatches(state);

  std::cout << "========================================" << std::endl;
  std::cout << "numPoints: " << state.numPoints << std::endl;
  std::cout << "numQueries: " << state.numQueries << std::endl;
  std::cout << "searchMode: " << state.searchMode << std::endl;
  std::cout << "radius: " << state.radius << std::endl;
  std::cout << "K: " << state.knn << std::endl;
  std::cout << "Same P and Q? " << std::boolalpha << state.samepq << std::endl;
  std::cout << "Partition? " << std::boolalpha << state.partition << std::endl;
  std::cout << "qGasSortMode: " << state.qGasSortMode << std::endl;
  std::cout << "pointSortMode: " << std::boolalpha << state.pointSortMode << std::endl;
  std::cout << "querySortMode: " << std::boolalpha << state.querySortMode << std::endl;
  std::cout << "cellRadiusRatio: " << std::boolalpha << state.crRatio << std::endl; // only useful when preSort == 1/2
  std::cout << "sortingGAS: " << state.sortingGAS << std::endl; // only useful when qGasSortMode != 0
  std::cout << "Gather? " << std::boolalpha << state.toGather << std::endl;
  std::cout << "Max batch count: " << state.maxBatchCount << std::endl;
  std::cout << "========================================" << std::endl << std::endl;

  try
  {
    setDevice(state);

    Timing::reset();

    uploadData(state);

    // if partition is enabled, we do it here too, where state.numOfBatches is set and batch related data structures are allocated.
    sortParticles(state, POINT, state.pointSortMode);
    // when samepq, queries are sorted using the point sort mode so no need to sort queries again.
    if (!state.samepq) sortParticles(state, QUERY, state.querySortMode);

    setupSearch(state);

    setupOptiX(state);

    Timing::startTiming("total search time");
    //TODO: try a better scheduling here (reverse the order)?
    for (int i = 0; i < state.numOfBatches; i++) {
      fprintf(stdout, "\n************** Batch %u **************\n", i);

      // it's possible that certain batches have 0 query (e.g., state.partThd too low).
      if (state.numActQueries[i] == 0) continue;

      // create the GAS using the current order of points and the launchRadius of the current batch.
      createGeometry (state, i); // batch_id ignored if not partition.

      if (state.qGasSortMode) gasSortSearch(state, i);

      //search(state, i);
    }

    for (int i = 0; i < state.numOfBatches; i++) {
      search(state, i);
    }

    CUDA_SYNC_CHECK();
    Timing::stopTiming(true);

    sanityCheck(state);

    cleanupState(state);
  }
  catch( std::exception& e )
  {
      std::cerr << "Caught exception: " << e.what() << "\n";
      return 1;
  }

  return 0;
}
