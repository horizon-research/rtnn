#include <unistd.h>

#include <sutil/Exception.h>
#include <sutil/Timing.h>

#include "optixNSearch.h"
#include "state.h"
#include "func.h"
#include "grid.h"

void setDevice ( RTNNState& state ) {
  int32_t device_count = 0;
  CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
  std::cerr << "\tTotal GPUs visible: " << device_count << std::endl;
  
  cudaDeviceProp prop;
  CUDA_CHECK( cudaGetDeviceProperties ( &prop, state.device_id ) );
  CUDA_CHECK( cudaSetDevice( state.device_id ) );
  std::cerr << "\tUsing [" << state.device_id << "]: " << prop.name << std::endl;
  state.totDRAMSize = (double)prop.totalGlobalMem/1024/1024/1024;
  std::cerr << "\tMemory: " << state.totDRAMSize << " GB" << std::endl;
  // conservatively reduce dram size by 256 MB as the usable memory appears to
  // be that much smaller than what is reported, presumably to store data
  // structures that are hidden from us.
  state.totDRAMSize -= 0.25;
}

void freeGridPointers( RTNNState& state ) {
  for (auto it = state.d_gridPointers.begin(); it != state.d_gridPointers.end(); it++) {
    CUDA_CHECK( cudaFree( *it ) );
  }
  //fprintf(stdout, "Finish early free\n");
}

void setupSearch( RTNNState& state ) {
  if (!state.deferFree) freeGridPointers(state);

  if (state.partition) return;

  assert(state.numOfBatches == -1);
  state.numOfBatches = 1;

  state.numActQueries[0] = state.numQueries;
  state.d_actQs[0] = state.params.queries;
  state.h_actQs[0] = state.h_queries;
  state.launchRadius[0] = state.radius;
}

void executeSearch(RTNNState& state)
{

  try {
    setDevice(state);

    // Timing::reset();
    uploadData(state);

    // call this after set device.
    initBatches(state);

    setupOptiX(state);

    // TODO: streamline the logic of partition and sorting.
    sortParticles(state, QUERY, state.querySortMode);

    // samepq indicates same underlying data and sorting mode, in which case
    // queries have been sorted so no need to sort them again.
    if (!state.samepq) sortParticles(state, POINT, state.pointSortMode);

    // early free done here too
    setupSearch(state);

    if (state.interleave) {
      for (int i = 0; i < state.numOfBatches; i++) {
        // it's possible that certain batches have 0 query (e.g., state.partThd too low).
        if (state.numActQueries[i] == 0) continue;
	    // TODO: group buildGas together to allow overlapping; this would allow
	    // us to batch-free temp storages and non-compacted gas storages. right
	    // now free storage serializes gas building.
        createGeometry (state, i, state.launchRadius[i]/state.gsrRatio); // batch_id ignored if not partition.
      }

      for (int i = 0; i < state.numOfBatches; i++) {
        if (state.numActQueries[i] == 0) continue;
        if (state.qGasSortMode) gasSortSearch(state, i);
      }

      for (int i = 0; i < state.numOfBatches; i++) {
        if (state.numActQueries[i] == 0) continue;
        if (state.qGasSortMode && state.gsrRatio != 1)
          createGeometry (state, i, state.launchRadius[i]);
      }

      for (int i = 0; i < state.numOfBatches; i++) {
        if (state.numActQueries[i] == 0) continue;
        // TODO: when K is too big, we can't launch all rays together. split rays.
        search(state, i);
      }
    } else {
      for (int i = 0; i < state.numOfBatches; i++) {
        if (state.numActQueries[i] == 0) continue;

        // create the GAS using the current order of points and the launchRadius of the current batch.
        // TODO: does it make sense to have per-batch |gsrRatio|?
        createGeometry (state, i, state.launchRadius[i]/state.gsrRatio); // batch_id ignored if not partition.

        if (state.qGasSortMode) {
          gasSortSearch(state, i);
          if (state.gsrRatio != 1)
            createGeometry (state, i, state.launchRadius[i]);
        }

        search(state, i);
      }
    }

    CUDA_SYNC_CHECK();

    if(state.sanCheck) sanityCheck(state);
    cleanupState(state);
  }
  catch( std::exception& e )
  {
    std::cerr << "Caught exception: " << e.what() << "\n";
    exit(1);
  }
}

/**
@param points: null-terminated array of points in flat float array.
  for each pair of floats at index [i, i + 1, i + 2] in the points array,
  the three elements are the x, y, z of the point
@param numPoints: number of points in the points array
@param radius: radius limit for all points
@param radii: null-terminated array of radius limits for each point in points
  if radii is NULL, the constant radius is used.
  if radii is not NULL, the radius parameter is ignored and the radii array is used.
@param max_interactions: maximum number of interactions to return

@returns: null-terminated array of neighbor pairs in float array.
  for each float[] in the main array,
  the first three elements are the x, y, z of the first point
  and the last three elements are the x, y, z of the second point
  radii[i] is the radius limit for points[i]
**/
float** getNeighborList(float* points, int numPoints, float radius, float* radii, double max_interactions) {
  float3* t_points = new float3[numPoints];

  // turn flat array of floats into array of float3's
  for (int i = 0; i < numPoints; i++) {
    int index = i * 3;
    t_points[i] = make_float3(points[index], points[index + 1], points[index + 2]);
  }

  RTNNState state;

  char* argv[] = {};
  parseArgs(state, 0, argv);

  // manually set state parameters
  state.numPoints = numPoints;
  state.numQueries = numPoints;
  state.h_points = t_points;
  state.h_queries = t_points;
  state.h_radii = radii;
  state.searchMode = "radius";
  state.sanCheck = false;
  state.params.radius = radius;
  state.radius = radius;
  state.samepq = true;
  state.knn = max_interactions;

  // enable optimizations
  state.querySortMode = 1;
  state.pointSortMode = 1;
  state.interleave = true;
  state.qGasSortMode = 2;

  if (state.radii == NULL) {
    state.autoNB = false; // keep batching and partitioning off, as it doesn't make sense for variable radii
    state.numOfBatches = -1;
    state.partition = false;
  } else {
    state.autoNB = true;
    state.partition = true;
  }

  executeSearch(state);

  float** neighbors = new float*[numPoints * numPoints]; // size of max amount of neighbor pairs
  int processed_count = 0;

  // extract result from the state
  // rtnn stores as a flat array of ints, where each point has max k neighbors
  // this converts to an array of neighbor pairs, described above
  for (int batch_id = 0; batch_id < state.numOfBatches; batch_id++) {
    
    unsigned int* result = (unsigned int*) state.h_res[batch_id];
    for (int p = 0; p < numPoints; p++) {
      float3 point = state.h_points[p];
      unsigned int prev_q = -1;

      for (int n = 1; n <= state.knn; n++) { // start at 1 to skip index fields
        unsigned long index = p * state.knn + n;
        unsigned int q = static_cast<unsigned int*>(state.h_res[batch_id])[index];
 
        // if q == UINT_MAX, that point has no more neighbors, and the rest of the qs in that section will all be UINT_MAX
        if (q == UINT_MAX) {
          break;
        }

        // prevent in-order duplicates, which rtnn occasionally generates
        if (q == prev_q) {
          continue;
        }

        float3 query = state.h_points[q];
        neighbors[processed_count] = new float[6];
        neighbors[processed_count][0] = query.x;
        neighbors[processed_count][1] = query.y;
        neighbors[processed_count][2] = query.z;
        neighbors[processed_count][3] = point.x;
        neighbors[processed_count][4] = point.y;
        neighbors[processed_count][5] = point.z;
        
        processed_count++;
        prev_q = q;
      }
    }
  }

  // null-terminate the return array
  neighbors[processed_count] = NULL;
  return neighbors;
}