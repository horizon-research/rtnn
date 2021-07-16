#include <sutil/Exception.h>
#include <sutil/Timing.h>

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

void setupSearch( WhittedState& state ) {
  if (!state.partition) {
    assert(state.numOfBatches == 1);

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

  std::cout << "========================================" << std::endl;
  std::cout << "numPoints: " << state.numPoints << std::endl;
  std::cout << "numQueries: " << state.numQueries << std::endl;
  std::cout << "searchMode: " << state.searchMode << std::endl;
  std::cout << "radius: " << state.radius << std::endl;
  std::cout << "E2E Measure? " << std::boolalpha << state.msr << std::endl;
  std::cout << "K: " << state.knn << std::endl;
  std::cout << "Same P and Q? " << std::boolalpha << state.samepq << std::endl;
  std::cout << "Partition? " << std::boolalpha << state.partition << std::endl;
  std::cout << "qGasSortMode: " << state.qGasSortMode << std::endl;
  std::cout << "pointSortMode: " << std::boolalpha << state.pointSortMode << std::endl;
  std::cout << "querySortMode: " << std::boolalpha << state.querySortMode << std::endl;
  std::cout << "cellRadiusRatio: " << std::boolalpha << state.crRatio << std::endl; // only useful when preSort == 1/2
  std::cout << "sortingGAS: " << state.sortingGAS << std::endl; // only useful when qGasSortMode != 0
  std::cout << "Gather? " << std::boolalpha << state.toGather << std::endl;
  std::cout << "========================================" << std::endl << std::endl;

  try
  {
    setDevice(state);

    // call this after set device.
    initBatches(state);

    setupOptiX(state);

    Timing::reset();
    Timing::startTiming("total search time");

    uploadData(state);

    // if partition is enabled, we do it here too, which generate batches.
    sortParticles(state, POINT, state.pointSortMode);
    // when samepq, queries have been are sorted using the point sort mode so
    // no need to sort queries again.
    if (!state.samepq) sortParticles(state, QUERY, state.querySortMode);

    setupSearch(state);

    bool interleave = false;
    if (interleave) {
      for (int i = 0; i < state.numOfBatches; i++) {
        if (state.numActQueries[i] == 0) continue;
        createGeometry (state, i); // batch_id ignored if not partition.
      }

      for (int i = 0; i < state.numOfBatches; i++) {
        if (state.numActQueries[i] == 0) continue;
        if (state.qGasSortMode) gasSortSearch(state, i);
      }

      for (int i = 0; i < state.numOfBatches; i++) {
        if (state.numActQueries[i] == 0) continue;
        search(state, i);
      }
    } else {
      for (int i = 0; i < state.numOfBatches; i++) {
      //for (int i = 0; i < 1; i++) {
        fprintf(stdout, "\n************** Batch %u **************\n", i);
        // it's possible that certain batches have 0 query (e.g., state.partThd too low).
        if (state.numActQueries[i] == 0) continue;

        // create the GAS using the current order of points and the launchRadius of the current batch.
        createGeometry (state, i); // batch_id ignored if not partition.

        if (state.qGasSortMode) gasSortSearch(state, i);

        search(state, i);
      }
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
