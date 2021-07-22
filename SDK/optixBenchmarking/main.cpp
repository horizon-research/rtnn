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
  std::cout << "isType: " << state.isType << std::endl;
  std::cout << "radius: " << state.radius << std::endl;
  std::cout << "E2E Measure? " << std::boolalpha << state.msr << std::endl;
  std::cout << "K: " << state.knn << std::endl;
  std::cout << "Same P and Q? " << std::boolalpha << state.samepq << std::endl;
  std::cout << "Partition? " << std::boolalpha << state.partition << std::endl;
  std::cout << "qGasSortMode: " << state.qGasSortMode << std::endl;
  std::cout << "pointSortMode: " << std::boolalpha << state.pointSortMode << std::endl;
  std::cout << "querySortMode: " << std::boolalpha << state.querySortMode << std::endl;
  std::cout << "cellRadiusRatio: " << std::boolalpha << state.crRatio << std::endl; // only useful when preSort == 1/2
  std::cout << "========================================" << std::endl << std::endl;

  try
  {
    setDevice(state);

    // call this after set device.
    initBatches(state);

    setupOptiX(state);

    Timing::reset();
    //Timing::startTiming("total search time");

    uploadData(state);

    /* GAS creation benchmarking */
    if (state.ubenchID == 3)
    {
      setupSearch(state);
      state.numOfBatches = 1;
      unsigned int numPoints = state.numPoints;
      for (int i = 1; i <= 100; i++) {
        float frac = (float)i/100;
        state.numPoints = numPoints * frac;
        createGeometry (state, 0); // batch_id ignored if not partition.
        CUDA_CHECK( cudaFree( state.d_aabb[0] ) );
        CUDA_CHECK( cudaFree( state.d_temp_buffer_gas[0] ) );
        if (reinterpret_cast<void*>(state.d_gas_output_buffer[0] ) != state.d_buffer_temp_output_gas_and_compacted_size[0])
          CUDA_CHECK( cudaFree( (void*)state.d_buffer_temp_output_gas_and_compacted_size[0] ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer[0] ) ) );
      }
    }

    // how knn/radius search time varies with the number of queries under different IS programs
    /* searchType: radius; isType: approx*/
    /* searchType: radius; isType: sphereTest*/
    if (state.ubenchID == 0)
    {
      sortParticles(state, POINT, 1);
      setupSearch(state);
      state.numOfBatches = 1;
      createGeometry (state, 0); // batch_id ignored if not partition.
      unsigned int numActQs = state.numActQueries[0];
      for (int i = 1; i <= 200; i++) {
        float frac = (float)i/200;
        state.numActQueries[0] = numActQs * frac;
        state.qGasSortMode = 2;
        gasSortSearch(state, 0);
        search(state, 0);
        CUDA_CHECK( cudaFreeHost(state.h_res[0] ) );
        CUDA_CHECK( cudaFree( state.d_res[0] ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>(state.d_r2q_map[0] )) );
        CUDA_CHECK( cudaFree( state.d_firsthit_idx[0] ) );
      }
      CUDA_CHECK( cudaFree( state.d_aabb[0] ) );
      CUDA_CHECK( cudaFree( state.d_temp_buffer_gas[0] ) );
      if (reinterpret_cast<void*>(state.d_gas_output_buffer[0] ) != state.d_buffer_temp_output_gas_and_compacted_size[0])
        CUDA_CHECK( cudaFree( state.d_buffer_temp_output_gas_and_compacted_size[0] ) );
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer[0] ) ) );
    }

    // how time varies with the # of IS calls and R
    // returns the IS calls for each query
    // fix total points and get the average statistics for all queries
    // rebuild the gas with every new R
    /* searchType: knn; isType: countIS */
    // TODO: check searchType and isType with ubenchID
    if (state.ubenchID == 1)
    {
      sortParticles(state, POINT, 1);
      setupSearch(state);
      state.numOfBatches = 1;
      float3 sceneSize = state.Max - state.Min;
      fprintf(stdout, "Scene size: %f, %f, %f\n", sceneSize.x, sceneSize.y, sceneSize.z);
      state.radius = std::min(sceneSize.x / 10, 15.0f);
      float radius = state.radius;
      for (unsigned int i = 1; i <= 100; i++) {
        float frac = (float)i/100;
        state.radius = radius * frac; // used for creating GAS
        state.launchRadius[0] = state.radius; // used for searching
        createGeometry (state, 0);
        state.qGasSortMode = 2;
        gasSortSearch(state, 0);
        search(state, 0, 1); // 1 entry per query in the result buffer

        float total = 0;
        for (unsigned int j = 0; j < state.numQueries; j++) {
          unsigned int p = reinterpret_cast<unsigned int*>( state.h_res[0] )[ j ];
          total += p;
        }
        fprintf(stdout, "\tAvg IS calls: %f\n\n", total/state.numActQueries[0]);

        CUDA_CHECK( cudaFreeHost(state.h_res[0] ) );
        CUDA_CHECK( cudaFree( state.d_res[0] ) );
        CUDA_CHECK( cudaFree( state.d_aabb[0] ) );
        CUDA_CHECK( cudaFree( state.d_temp_buffer_gas[0] ) );
        if (reinterpret_cast<void*>(state.d_gas_output_buffer[0] ) != state.d_buffer_temp_output_gas_and_compacted_size[0])
          CUDA_CHECK( cudaFree( (void*)state.d_buffer_temp_output_gas_and_compacted_size[0] ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer[0] ) ) );
      }
    }

    // same as above, but we now get the data for queries at different density fields
    /* searchType: knn; isType: countIS */
    if (state.ubenchID == 2)
    {
      state.partition = 1;
      sortParticles(state, POINT, 1);
      setupSearch(state);
      float3 sceneSize = state.Max - state.Min;
      fprintf(stdout, "Scene size: %f, %f, %f\n", sceneSize.x, sceneSize.y, sceneSize.z);
      state.radius = std::min(sceneSize.x / 10, 15.0f);
      float radius = state.radius;
      //for (int b = 0; b < state.numOfBatches; b++) {
      for (int b = state.numOfBatches - 1; b < state.numOfBatches; b++) {
        fprintf(stdout, "\tBatch %d\n", b);
        //unsigned int numActQs = state.numActQueries[b]; // uncomment this to sweep the # of queries in a batch
        //for (int j = 1; j <= 4; j++) {
        //  float frac1 = (float)j/40;
        //  state.numActQueries[b] = numActQs * frac1;
          for (unsigned int i = 1; i <= 100; i++) {
            float frac2 = (float)i/100;
            state.radius = radius * frac2; // used for creating GAS
            state.launchRadius[b] = state.radius; // used for searching
            createGeometry (state, b);
            state.qGasSortMode = 2;
            gasSortSearch(state, b);
            search(state, b, 1); // 1 entry per query in the result buffer

            float total = 0;
            for (unsigned int j = 0; j < state.numActQueries[b]; j++) {
              unsigned int p = reinterpret_cast<unsigned int*>( state.h_res[b] )[ j ];
              total += p;
            }
            fprintf(stdout, "\tAvg IS calls: %f\n\n", total/state.numActQueries[b]);

            CUDA_CHECK( cudaFreeHost(state.h_res[b] ) );
            CUDA_CHECK( cudaFree( state.d_res[b] ) );
            CUDA_CHECK( cudaFree( state.d_aabb[b] ) );
            CUDA_CHECK( cudaFree( state.d_temp_buffer_gas[b] ) );
            if (reinterpret_cast<void*>(state.d_gas_output_buffer[b] ) != state.d_buffer_temp_output_gas_and_compacted_size[b])
              CUDA_CHECK( cudaFree( (void*)state.d_buffer_temp_output_gas_and_compacted_size[b] ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer[b] ) ) );
          }
        //}
      }
    }

    // how radius search time varies with K
    // fix the GAS and the point cloud; change only params.limit. returns when the limit is met.
    // the radius is intentionally set to be large so that the limit is roughly the same as IS calls
    /* searchType: radius; isType: retOnLimit */
    //{
    //  sortParticles(state, POINT, 1);
    //  setupSearch(state);
    //  state.numOfBatches = 1;
    //  state.launchRadius[0] = 10; // an intentionally large radius.
    //  createGeometry (state, 0); // batch_id ignored if not partition.
    //  state.qGasSortMode = 2;
    //  gasSortSearch(state, 0);
    //  for (unsigned int i = 1; i <= state.numPoints; i+=100) {
    //    state.knn = i;
    //    search(state, 0, 1); // overloaded version where the res devive buffer is set to minimal size
    //    CUDA_CHECK( cudaFree( state.d_res[0] ) );
    //  }
    //  CUDA_CHECK( cudaFree( state.d_aabb[0] ) );
    //  CUDA_CHECK( cudaFree( state.d_temp_buffer_gas[0] ) );
    //  if (reinterpret_cast<void*>(state.d_gas_output_buffer[0] ) != state.d_buffer_temp_output_gas_and_compacted_size[0])
    //    CUDA_CHECK( cudaFree( (void*)state.d_buffer_temp_output_gas_and_compacted_size[0] ) );
    //  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer[0] ) ) );
    //}

    CUDA_SYNC_CHECK();
    //Timing::stopTiming(true);
    exit(1);

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
