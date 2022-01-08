#include <cuda_runtime.h>

#include <sutil/Exception.h>
#include <sutil/Timing.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include <algorithm>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <string>
#include <random>
#include <cstdlib>
#include <queue>
#include <unordered_set>
#include <map>
#include <float.h>

#include "optixNSearch.h"
#include "func.h"
#include "state.h"
#include "grid.h"

#ifdef MEM_STATS
  extern std::map<void*, double> memmap;
  extern double tot_alloc_size;
#endif

void computeMinMax(unsigned int N, float3* particles, float3& min, float3& max)
{
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
 
  min.x = minCell.x;
  min.y = minCell.y;
  min.z = minCell.z;
 
  max.x = maxCell.x;
  max.y = maxCell.y;
  max.z = maxCell.z;

  //fprintf(stdout, "\tcell boundary: (%d, %d, %d), (%d, %d, %d)\n", minCell.x, minCell.y, minCell.z, maxCell.x, maxCell.y, maxCell.z);
  fprintf(stdout, "\tscene boundary: (%f, %f, %f), (%f, %f, %f)\n", min.x, min.y, min.z, max.x, max.y, max.z);
}

unsigned int genGridInfo(RTNNState& state, unsigned int N, GridInfo& gridInfo) {
  float3 sceneMin = state.Min;
  float3 sceneMax = state.Max;

  gridInfo.ParticleCount = N;
  gridInfo.GridMin = sceneMin;

  float cellSize = state.radius / state.crRatio;
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
  // dimension is of the same size and the dimension is a power of 2. if we
  // were to generate one single morton code for the entire grid, this would
  // waste a lot of space since a lot of empty cells will have to be padded.
  // the strategy is to divide the grid into smaller equal-dimension-power-of-2
  // smaller grids (meta_grid here). the order within each meta_grid is morton,
  // but the order across meta_grids is raster order. the current
  // implementation uses a heuristics. TODO: revisit this later.
  gridInfo.meta_grid_dim = (int)pow(2, floorf(log2(std::min({gridInfo.GridDimension.x, gridInfo.GridDimension.y, gridInfo.GridDimension.z}))))/4;
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
  fprintf(stdout, "\tGrid dimension (without meta grids): %u, %u, %u\n", gridInfo.GridDimension.x, gridInfo.GridDimension.y, gridInfo.GridDimension.z);
  fprintf(stdout, "\tGrid dimension (with meta grids): %u, %u, %u\n", gridInfo.MetaGridDimension.x * gridInfo.meta_grid_dim, gridInfo.MetaGridDimension.y * gridInfo.meta_grid_dim, gridInfo.MetaGridDimension.z * gridInfo.meta_grid_dim);
  //fprintf(stdout, "\tMeta Grid dimension: %u, %u, %u\n", gridInfo.MetaGridDimension.x, gridInfo.MetaGridDimension.y, gridInfo.MetaGridDimension.z);
  //fprintf(stdout, "\t# of cells in a meta grid: %u\n", gridInfo.meta_grid_dim);
  //fprintf(stdout, "\tGridDelta: %f, %f, %f\n", gridInfo.GridDelta.x, gridInfo.GridDelta.y, gridInfo.GridDelta.z);
  fprintf(stdout, "\tNumber of cells: %u\n", numberOfCells);
  fprintf(stdout, "\tCell size: %f\n", cellSize);

  // update GridDimension so that it can be used in the kernels (otherwise raster order is incorrect)
  gridInfo.GridDimension.x = gridInfo.MetaGridDimension.x * gridInfo.meta_grid_dim;
  gridInfo.GridDimension.y = gridInfo.MetaGridDimension.y * gridInfo.meta_grid_dim;
  gridInfo.GridDimension.z = gridInfo.MetaGridDimension.z * gridInfo.meta_grid_dim;
  return numberOfCells;
}

void test(GridInfo);

thrust::device_ptr<int> genCellMask (RTNNState& state, unsigned int* d_repQueries, float3* particles, unsigned int* d_CellParticleCounts, unsigned int numberOfCells, GridInfo gridInfo, unsigned int N, unsigned int numUniqQs, bool morton) {
  float cellSize = state.radius / state.crRatio;

  // |maxWidth| is the max width of a cube that can be enclosed by the sphere.
  // in radius search, we can generate an AABB of this size and be sure that
  // there are >= K points within this AABB, we don't have to calc the dist
  // since these points are gauranted to be in the search sphere (but see the
  // important caveats in the search function). in knn search, however, we
  // can't be sure the nearest K points are in this AABB (there are points that
  // are outside of the AABB that are closer to the centroid than points in the
  // AABB), but the K nearest neighbors are gauranteed to be in the sphere that
  // tightly encloses this cube, and given the way we calculate |maxWidth| we
  // know that the radius of that sphere won't be greater than state.radius, so
  // we still save time.
  float maxWidth = maxInscribedWidth(state.radius, 3);

  thrust::device_ptr<int> d_cellMask;
  allocThrustDevicePtr(&d_cellMask, numberOfCells); // no need to memset this since every single cell will be updated.
  //CUDA_CHECK( cudaMemset ( thrust::raw_pointer_cast(d_cellMask), 0xFF, numberOfCells * sizeof(int) ) );

  //test(gridInfo); // to demonstrate the weird parameter passing bug.

  bool gpu = true;
  if (gpu) {
    //thrust::host_vector<unsigned int> h_CellParticleCounts(numberOfCells);
    //thrust::copy(thrust::device_pointer_cast(d_CellParticleCounts), thrust::device_pointer_cast(d_CellParticleCounts) + numberOfCells, h_CellParticleCounts.begin());

    unsigned int threadsPerBlock = 64;
    unsigned int numOfBlocks = numUniqQs / threadsPerBlock + 1;
    kCalcSearchSize(numOfBlocks,
                    threadsPerBlock,
                    gridInfo,
                    morton, 
                    d_CellParticleCounts,
                    d_repQueries,
                    particles,
                    cellSize,
                    maxWidth,
                    state.knn,
                    thrust::raw_pointer_cast(d_cellMask)
                   );

    //thrust::host_vector<int> h_cellMask_t(numberOfCells);
    //thrust::copy(d_cellMask, d_cellMask + numberOfCells, h_cellMask_t.begin());
  } else {
    thrust::host_vector<unsigned int> h_part_seq(numUniqQs);
    thrust::copy(thrust::device_pointer_cast(d_repQueries), thrust::device_pointer_cast(d_repQueries) + numUniqQs, h_part_seq.begin());

    thrust::host_vector<unsigned int> h_CellParticleCounts(numberOfCells);
    thrust::copy(thrust::device_pointer_cast(d_CellParticleCounts), thrust::device_pointer_cast(d_CellParticleCounts) + numberOfCells, h_CellParticleCounts.begin());

    thrust::host_vector<int> h_cellMask(numberOfCells);

    for (unsigned int i = 0; i < numUniqQs; i++) {
      unsigned int qId = h_part_seq[i];
      float3 point = state.h_points[qId];
      float3 gridCellF = (point - gridInfo.GridMin) * gridInfo.GridDelta;
      int3 gridCell = make_int3(int(gridCellF.x), int(gridCellF.y), int(gridCellF.z));

      calcSearchSize(gridCell,
                     gridInfo,
                     morton,
                     h_CellParticleCounts.data(),
                     cellSize,
                     maxWidth,
                     state.knn,
                     h_cellMask.data()
                    );
    }
    thrust::copy(h_cellMask.begin(), h_cellMask.end(), d_cellMask);
  }

  //for (unsigned int i = 0; i < numberOfCells; i++) {
  //  if (h_cellMask[i] == 2)
  //    printf("%u, %u, %x\n", i, h_cellMask[i], h_cellMask_t[i]);
  //}

  return d_cellMask;
}

void autoBatchingRange(RTNNState& state, const thrust::host_vector<unsigned int>& h_rayHist, std::vector<int>& batches, int numAvailBatches) {
  // now that we allow AABBTEST in all but the last batch in radius search,
  // batching could save time, since doing sphere test is much more costly than
  // aabb test. build the cost model and find the optimal batching.

  // empirical coefficients on 2080Ti
  //const float kD2H_PerB = 6e-7; // D2H memcpy time in *ms* / byte (TODO)
  const float kBuildGas_PerAABB = 3.8e-6; // GAS building time in *ms* / AABB
  // higher values encourage batching.
  //const float kAABBTest_PerIS = 4e-5/50; // IS call time in *ms* if doing aabb test (by 50 as the data was ubenchmarked using K=50)
  //const float kSphereTest_PerIS = 4e-4/50; // IS call time in *ms* if doing sphere test

  // empirical coefficients on 2080
  const float kAABBTest_PerIS = 1e-5/50;
  const float kSphereTest_PerIS = 1e-4/50;

  //float tMemcpy = state.numQueries * state.knn * sizeof(unsigned int) * kD2H_PerB; // TODO: consider max(memcpy, compute)
  float tBuildGAS = state.numPoints * kBuildGas_PerAABB + 20; // 20 is the empirical intercept (TODO)
  //fprintf(stdout, "tBuildGAS: %f\n", tBuildGAS);

  // incrementally combine batch i with the last batch (assuming all other
  // batches are independent) and calculate the cost. choose the min cost.
  float overhead = 0;
  float maxOverhead = 0; // overhead must be negative for bundling to be useful
  int splitId = numAvailBatches - 1; // by default we don't bundle
  for (int i = numAvailBatches - 2; i >= 0; i--) {
    float extraTime = h_rayHist[i] * (kSphereTest_PerIS - kAABBTest_PerIS) * state.knn;
    overhead += extraTime - tBuildGAS;
    //fprintf(stdout, "i: %d, %u extraTime: %f, overhead: %f\n", i, h_rayHist[i], extraTime, overhead);
    if (overhead < maxOverhead) {
      maxOverhead = overhead;
      splitId = i;
    }
  }

  for (int i = 0; i <= splitId - 1; i++) {
    batches.push_back(i);
  }
  batches.push_back(numAvailBatches - 1);
}

float radiusFromMegacell(float width, int approxMode) {
  if (approxMode == 2) return radiusEquiVolume(width, 3); // 0.62; works well for uniform density
  // for a sphere to be of the same volume as the cube, its radius is width *
  // 0.62. if we use 2 in |minCircumscribedRadius|, the radius is width * 0.71.
  // so very likely the sphere will still have more than K neighbors.
  else if (approxMode == 1) return minCircumscribedRadius(width, 2); // 0.71
  else return minCircumscribedRadius(width, 3); // 0.87
}

void autoBatchingKNN(RTNNState& state, const thrust::host_vector<unsigned int>& h_rayHist, std::vector<int>& batches, int numAvailBatches) {
  // Logic: given CR (which has been decided beforehand), we know that max # of
  //   available batches (|numAvailBatches|). launching as many batches as
  //   available minimizes the work, but also introduces gas building overhead.
  // So we build a cost model = gas building time + searching time.
  // GAS building time is linear w.r.t. to the # of AABBs, which is the total amount of points.
  // Searching time = max(memcpy time, compute time).
  // The memcpy time is empirically observed to be linear w.r.t., to the # of queries
  // The compute time, without considering CKE, is the lump sum of the compute time of each batch, which is linear w.r.t. the # of queries in the batch and cubic w.r.t., to the radius in the batch.

  // empirical coefficients on 2080Ti
  //const float kD2H_PerB = 6e-7; // D2H memcpy time in *ms* / byte (TODO)
  const float kBuildGas_PerAABB = 3.8e-6; // GAS building time in *ms* / AABB
  // TODO: fit a better model for IS calls? N_tl * T_tl + N_is * T_is
  // TODO: this should depend K.
  const float kSearch_PerIS = 6e-2; // knn search time in *ms* per IS call

  //float tMemcpy = state.numQueries * state.knn * sizeof(unsigned int) * kD2H_PerB; // TODO: consider max(memcpy, compute)
  float tBuildGAS = state.numPoints * kBuildGas_PerAABB + 20; // 20 is the empirical intercept (TODO)
  float cellSize = state.radius / state.crRatio;
  //fprintf(stdout, "tBuildGAS: %f\n", tBuildGAS);

  float maxWidth = kGetWidthFromIter(numAvailBatches - 1, cellSize);
  float maxRadius = std::min(state.radius, radiusFromMegacell(maxWidth, state.approxMode));
  // incrementally combine batch i with the last batch (assuming all other
  // batches are independent) and calculate the cost. choose the min cost.
  float overhead = 0;
  float maxOverhead = 0;
  int splitId = numAvailBatches - 1;
  for (int i = numAvailBatches - 2; i >= 0; i--) {
    float curWidth = kGetWidthFromIter(i, cellSize);
    float curRadius = std::min(state.radius, radiusFromMegacell(curWidth, state.approxMode));
    float density = state.knn / ((curWidth - cellSize) * (curWidth - cellSize) * (curWidth - cellSize));

    // TODO: assuming density doesn't change dramatically; consider non-uniform density?
    float extraWork = h_rayHist[i] * 8 * (maxRadius * maxRadius * maxRadius - curRadius * curRadius * curRadius) * density;
    float extraTime = extraWork * kSearch_PerIS;
    overhead += extraTime - tBuildGAS;
    //fprintf(stdout, "i: %d, density: %f, extraWork: %f, extraTime: %f, overhead: %f\n", i, density, extraWork, extraTime, overhead);
    if (overhead < maxOverhead) {
      maxOverhead = overhead;
      splitId = i;
    }
  }

  for (int i = 0; i <= splitId - 1; i++) {
    batches.push_back(i);
  }
  batches.push_back(numAvailBatches - 1);
}

void prepBatches(RTNNState& state, std::vector<int>& batches, const thrust::host_vector<unsigned int>& h_rayHist) {
  int numAvailBatches = (int)h_rayHist.size();
  fprintf(stdout, "\tnumAvailBatches: %d\n", numAvailBatches);

  if (state.autoNB) {
    if (state.searchMode == "knn") autoBatchingKNN(state, h_rayHist, batches, numAvailBatches);
    else autoBatchingRange(state, h_rayHist, batches, numAvailBatches);
  } else {
    if (numAvailBatches == 1) {
      batches.push_back(0);
      return;
    }

    int numBatches;
    if (state.numOfBatches == -1) numBatches = (int)numAvailBatches;
    else numBatches = std::min(state.numOfBatches, (int)numAvailBatches);

    for (int i = 0; i < numAvailBatches; i++) {
      if (i <= numBatches - 2 || i == numAvailBatches - 1) batches.push_back(i);
    }
    assert(batches.size() <= (unsigned int)numBatches);
  }
}

void genBatches(RTNNState& state,
                std::vector<int>& batches,
                thrust::host_vector<unsigned int> h_rayHist,
                float3* particles,
                unsigned int N,
                thrust::device_ptr<int> d_rayMask)
{
  float cellSize = state.radius / state.crRatio;

  int lastMask = -1;
  for (int batchId = 0; batchId < state.numOfBatches; batchId++) {
    int maxMask = batches[batchId];
    unsigned int numActQs = 0;
    for (int j = lastMask + 1; j <= maxMask; j++) {
      numActQs += h_rayHist[j];
    }
    state.numActQueries[batchId] = numActQs;
    //printf("[%d, %d]: %u\n", lastMask + 1, maxMask, numActQs);

    // see comments in how maxWidth is calculated in |genCellMask|.
    float partThd = kGetWidthFromIter(maxMask, cellSize); // partThd depends on the max mask.
    if (state.searchMode == "knn")
      state.launchRadius[batchId] = radiusFromMegacell(partThd, state.approxMode);
    else
      state.launchRadius[batchId] = partThd / 2;
    if (batchId == (state.numOfBatches - 1)) state.launchRadius[batchId] = state.radius;
    //printf("%u, %f\n", maxMask, state.launchRadius[batchId]);

    // can't free |particles|, because it points to the points too.
    // same applies to state.h_queries. |particles| from this point
    // on will only be used to point to device queries used in kernels, and
    // will be set right before launch using d_actQs.
    thrust::device_ptr<float3> d_actQs;
    allocThrustDevicePtr(&d_actQs, numActQs);
    copyIfIdInRange(particles, N, d_rayMask, d_actQs, lastMask + 1, maxMask);
    state.d_actQs[batchId] = thrust::raw_pointer_cast(d_actQs);

    // Copy the active queries to host (for sanity check).
    // TODO: is this redundant given the copy at the end of |gridSort|?
    state.h_actQs[batchId] = new float3[numActQs];
    thrust::copy(d_actQs, d_actQs + numActQs, state.h_actQs[batchId]);

    lastMask = maxMask;
  }
}

void sortGenBatch(RTNNState& state,
                  unsigned int N,
                  bool morton,
                  unsigned int numberOfCells,
                  unsigned int numOfBlocks,
                  unsigned int threadsPerBlock,
                  GridInfo gridInfo,
                  float3* particles,
                  thrust::device_ptr<unsigned int> d_CellParticleCounts_ptr,
                  thrust::device_ptr<unsigned int> d_ParticleCellIndices_ptr,
                  thrust::device_ptr<unsigned int> d_CellOffsets_ptr,
                  thrust::device_ptr<unsigned int> d_LocalSortedIndices_ptr,
                  thrust::device_ptr<unsigned int> d_posInSortedPoints_ptr
                 )
{
    // pick one particle from each cell, and store all their indices in |d_repQueries|
    thrust::device_ptr<unsigned int> d_ParticleCellIndices_ptr_copy;
    allocThrustDevicePtr(&d_ParticleCellIndices_ptr_copy, N);
    thrustCopyD2D(d_ParticleCellIndices_ptr_copy, d_ParticleCellIndices_ptr, N);
    thrust::device_ptr<unsigned int> d_repQueries;
    allocThrustDevicePtr(&d_repQueries, N);
    genSeqDevice(d_repQueries, N);
    sortByKey(d_ParticleCellIndices_ptr_copy, d_repQueries, N);
    unsigned int numUniqQs = uniqueByKey(d_ParticleCellIndices_ptr_copy, N, d_repQueries);
    fprintf(stdout, "\tNum of Rep queries: %u\n", numUniqQs);

    // generate the cell mask
    thrust::device_ptr<int> d_cellMask = genCellMask(state,
            thrust::raw_pointer_cast(d_repQueries),
            particles,
            thrust::raw_pointer_cast(d_CellParticleCounts_ptr),
            numberOfCells,
            gridInfo,
            N,
            numUniqQs,
            morton
           );

    thrust::device_ptr<int> d_rayMask;
    allocThrustDevicePtr(&d_rayMask, N);

    // generate the sorted indices, and also set the rayMask according to cellMask.
    kCountingSortIndices_setRayMask(numOfBlocks,
                                    threadsPerBlock,
                                    gridInfo,
                                    thrust::raw_pointer_cast(d_ParticleCellIndices_ptr),
                                    thrust::raw_pointer_cast(d_CellOffsets_ptr),
                                    thrust::raw_pointer_cast(d_LocalSortedIndices_ptr),
                                    thrust::raw_pointer_cast(d_posInSortedPoints_ptr),
                                    thrust::raw_pointer_cast(d_cellMask),
                                    thrust::raw_pointer_cast(d_rayMask)
                                   );

    // get a histogram of d_rayMask, which won't be mutated. this needs to happen before sorting |d_rayMask|.
    // the last mask in the histogram indicates the number of rays that need full search.
    thrust::device_vector<unsigned int> d_rayHist;
    unsigned int numMasks = thrustGenHist(d_rayMask, d_rayHist, N);
    thrust::host_vector<unsigned int> h_rayHist(numMasks);
    thrust::copy(d_rayHist.begin(), d_rayHist.end(), h_rayHist.begin());

    // Sort the queries if sorting is enabled, in which case sort the ray masks
    // the same way as query sorting. Sorting particles MUST happen right after
    // sorting the masks so that queries and masks are consistent!!!
    if (state.pointSortMode) {
      // make a copy of the keys since they are useless after the first sort. no
      // need to use stable sort since the keys are unique, so masks and the
      // queries are gauranteed to be sorted in exactly the same way.
      // TODO: Can we do away with the extra copy by replacing sort by key with scatter? That'll need new space too...
      thrust::device_ptr<unsigned int> d_posInSortedPoints_ptr_copy;
      allocThrustDevicePtr(&d_posInSortedPoints_ptr_copy, N);
      thrustCopyD2D(d_posInSortedPoints_ptr_copy, d_posInSortedPoints_ptr, N);

      sortByKey(d_posInSortedPoints_ptr_copy, d_rayMask, N);
      sortByKey(d_posInSortedPoints_ptr, thrust::device_pointer_cast(particles), N);
      state.d_pointers.insert((void*)thrust::raw_pointer_cast(d_posInSortedPoints_ptr_copy));
    }

    // |batches| will contain the last mask of each batch.
    std::vector<int> batches;
    prepBatches(state, batches, h_rayHist);
    state.numOfBatches = batches.size();
    fprintf(stdout, "\tNumber of batches: %d\n", state.numOfBatches);

    genBatches(state, batches, h_rayHist, particles, N, d_rayMask);

    state.d_pointers.insert((void*)thrust::raw_pointer_cast(d_rayMask));
    state.d_pointers.insert((void*)thrust::raw_pointer_cast(d_cellMask));
}

void gridSort(RTNNState& state, unsigned int N, float3* particles, float3* h_particles, bool morton, bool toPartition) {
  GridInfo gridInfo;
  unsigned int numberOfCells = genGridInfo(state, N, gridInfo);

  thrust::device_ptr<unsigned int> d_ParticleCellIndices_ptr;
  allocThrustDevicePtr(&d_ParticleCellIndices_ptr, N);
  thrust::device_ptr<unsigned int> d_CellParticleCounts_ptr;
  allocThrustDevicePtr(&d_CellParticleCounts_ptr, numberOfCells); // this takes a lot of memory
  fillByValue(d_CellParticleCounts_ptr, numberOfCells, 0);
  thrust::device_ptr<unsigned int> d_LocalSortedIndices_ptr;
  allocThrustDevicePtr(&d_LocalSortedIndices_ptr, N);

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

  thrust::device_ptr<unsigned int> d_CellOffsets_ptr;
  allocThrustDevicePtr(&d_CellOffsets_ptr, numberOfCells);
  fillByValue(d_CellOffsets_ptr, numberOfCells, 0); // need to initialize it even for exclusive scan
  exclusiveScan(d_CellParticleCounts_ptr, numberOfCells, d_CellOffsets_ptr);

  thrust::device_ptr<unsigned int> d_posInSortedPoints_ptr;
  allocThrustDevicePtr(&d_posInSortedPoints_ptr, N);
  // if partition is enabled, do it here. we are partitioning points, but it's the same as queries.
  if (toPartition) {
    // normal particle sorting is done here too.
    sortGenBatch(state,
                 N,
                 morton,
                 numberOfCells,
                 numOfBlocks,
                 threadsPerBlock,
                 gridInfo,
                 particles,
                 d_CellParticleCounts_ptr,
                 d_ParticleCellIndices_ptr,
                 d_CellOffsets_ptr,
                 d_LocalSortedIndices_ptr,
                 d_posInSortedPoints_ptr
                );
  } else {
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
  }

  // copy particles to host, regardless of partition. for POINT, this makes
  // sure the points in device are consistent with the host points used to
  // build the GAS. for QUERY and POINT, this sets up data for sanity check.
  thrust::device_ptr<float3> d_particles_ptr = thrust::device_pointer_cast(particles);
  thrust::copy(d_particles_ptr, d_particles_ptr + N, h_particles);

  state.d_pointers.insert((void*)thrust::raw_pointer_cast(d_ParticleCellIndices_ptr));
  state.d_pointers.insert((void*)thrust::raw_pointer_cast(d_posInSortedPoints_ptr));
  state.d_pointers.insert((void*)thrust::raw_pointer_cast(d_CellOffsets_ptr));
  state.d_pointers.insert((void*)thrust::raw_pointer_cast(d_LocalSortedIndices_ptr));
  state.d_pointers.insert((void*)thrust::raw_pointer_cast(d_CellParticleCounts_ptr));
}

void oneDSort ( RTNNState& state, unsigned int N, float3* particles, float3* h_particles ) {
  // sort points/queries based on coordinates (x/y/z)

  // TODO: do this whole thing on GPU.
  // create 1d points as the sorting key and upload it to device memory
  thrust::host_vector<float> h_key(N);
  for(unsigned int i = 0; i < N; i++) {
    h_key[i] = h_particles[i].x;
  }

  thrust::device_ptr<float> d_key_ptr;
  state.d_1dsort_key = allocThrustDevicePtr(&d_key_ptr, state.numQueries);
  thrust::copy(h_key.begin(), h_key.end(), d_key_ptr);

  // actual sort
  thrust::device_ptr<float3> d_particles_ptr = thrust::device_pointer_cast(particles);
  sortByKey( d_key_ptr, d_particles_ptr, N );

  // TODO: lift it outside of this function and combine with other sorts?
  // copy the sorted queries to host so that we build the GAS in the same order
  // note that the h_queries at this point still point to what h_points points to
  thrust::copy(d_particles_ptr, d_particles_ptr + N, h_particles);
}

void sortParticles ( RTNNState& state, ParticleType type, int sortMode ) {
  // 0: no sort
  // 1: z-order sort
  // 2: raster sort
  // 3: 1D sort

  if (!sortMode) {
    // even if sortMode == 0, but if partition is enabled for POINT, go ahead
    // since we need to partition; won't actually sort since we will check this
    // again later. ugly logic.
    if ((type == POINT) && !state.partition) return;
    else if (type == QUERY) return;
  }

  unsigned int N;
  float3* particles;
  float3* h_particles;
  if (type == POINT) {
    N = state.numPoints;
    particles = state.params.points;
    h_particles = state.h_points;
    state.Min = state.pMin;
    state.Max = state.pMax;
  } else {
    N = state.numQueries;
    particles = state.params.queries;
    h_particles = state.h_queries;
    state.Min = state.qMin;
    state.Max = state.qMax;
  }

  // the semantices of the two sort functions are: sort data in device, and copy the sorted data back to host.
  std::string typeName = ((type == POINT) ? "points" : "queries");
  Timing::startTiming("sort " + typeName);
    if (sortMode == 3) {
      oneDSort(state, N, particles, h_particles);
    } else {
      bool morton; // false for raster order
      if (sortMode == 1) morton = true;
      else morton = false;
      gridSort(state, N, particles, h_particles, morton, state.partition&&(type==POINT));
    }
  Timing::stopTiming(true);
}

thrust::device_ptr<unsigned int> sortQueriesByFHCoord( RTNNState& state, thrust::device_ptr<unsigned int> d_firsthit_idx_ptr, int batch_id ) {
  // this is sorting queries by the x/y/z coordinate of the first hit primitives.
  unsigned int numQueries = state.numActQueries[batch_id];

  Timing::startTiming("gas-sort queries init");
    // allocate device memory for storing the keys, which will be generated by a gather and used in sort_by_keys
    thrust::device_ptr<float> d_key_ptr;
    state.d_fhsort_key = allocThrustDevicePtr(&d_key_ptr, numQueries);
  
    // create keys (1d coordinate), which will become the source of gather, the
    // result of which will be the keys for sort; the size must be
    // state.numPoints rather than numQueries. without point/query sorting,
    // Coord-sort can be better than ID-sort since the IDs of the FH primitives
    // will be arbitrary.
    thrust::host_vector<float> h_orig_points_1d(state.numPoints);
    // TODO: do this in CUDA
    for (unsigned int i = 0; i < state.numPoints; i++) {
      h_orig_points_1d[i] = state.h_points[i].z; // could be other dimensions
    }
    thrust::device_vector<float> d_orig_points_1d = h_orig_points_1d;

    // initialize a sequence to be sorted, which will become the r2q map.
    thrust::device_ptr<unsigned int> d_r2q_map_ptr;
    allocThrustDevicePtr(&d_r2q_map_ptr, numQueries);
    genSeqDevice(d_r2q_map_ptr, numQueries, state.stream[batch_id]);
  Timing::stopTiming(true);
 
  Timing::startTiming("gas-sort queries");
    // first use a gather to generate the keys, then sort by keys
    gatherByKey(d_firsthit_idx_ptr, &d_orig_points_1d, d_key_ptr, numQueries, state.stream[batch_id]);
    sortByKey( d_key_ptr, d_r2q_map_ptr, numQueries, state.stream[batch_id] );
    state.d_r2q_map[batch_id] = thrust::raw_pointer_cast(d_r2q_map_ptr);
  Timing::stopTiming(true);
 
  // if debug, copy the sorted keys and values back to host
  bool debug = false;
  if (debug) {
    thrust::host_vector<unsigned int> h_vec_val(numQueries);
    thrust::copy(d_r2q_map_ptr, d_r2q_map_ptr+numQueries, h_vec_val.begin());

    thrust::host_vector<float> h_vec_key(numQueries);
    thrust::copy(d_key_ptr, d_key_ptr+numQueries, h_vec_key.begin());

    float3* h_queries = state.h_actQs[batch_id];
    for (unsigned int i = 0; i < h_vec_val.size(); i++) {
      std::cout << h_vec_key[i] << "\t" 
                << h_vec_val[i] << "\t" 
                << h_queries[h_vec_val[i]].x << "\t"
                << h_queries[h_vec_val[i]].y << "\t"
                << h_queries[h_vec_val[i]].z
                << std::endl;
    }
  }

  return d_r2q_map_ptr;
}

thrust::device_ptr<unsigned int> sortQueriesByFHIdx( RTNNState& state, thrust::device_ptr<unsigned int> d_firsthit_idx_ptr, int batch_id ) {
  // this is sorting queries just by the first hit primitive IDs
  unsigned int numQueries = state.numActQueries[batch_id];

  // initialize a sequence to be sorted, which will become the r2q map
  Timing::startTiming("gas-sort queries init");
    thrust::device_ptr<unsigned int> d_r2q_map_ptr;
    allocThrustDevicePtr(&d_r2q_map_ptr, numQueries);
    genSeqDevice(d_r2q_map_ptr, numQueries, state.stream[batch_id]);
  Timing::stopTiming(true);

  Timing::startTiming("gas-sort queries");
    sortByKey( d_firsthit_idx_ptr, d_r2q_map_ptr, numQueries, state.stream[batch_id] );
    unsigned int uniqFHs = countUniq(d_firsthit_idx_ptr, numQueries);
    fprintf(stdout, "\tUnique FH AABBs: %u\n", uniqFHs);

    // thrust can't be used in kernel code since NVRTC supports only a
    // limited subset of C++, so we would have to explicitly cast a
    // thrust device vector to its raw pointer. See the problem discussed
    // here: https://github.com/cupy/cupy/issues/3728 and
    // https://github.com/cupy/cupy/issues/3408. See how cuNSearch does it:
    // https://github.com/InteractiveComputerGraphics/cuNSearch/blob/master/src/cuNSearchDeviceData.cu#L152
    state.d_r2q_map[batch_id] = thrust::raw_pointer_cast(d_r2q_map_ptr);
    //printf("%d, %p\n", batch_id, state.d_r2q_map[batch_id]);
  Timing::stopTiming(true);

  bool debug = false;
  if (debug) {
    thrust::host_vector<unsigned int> h_vec_val(numQueries);
    thrust::copy(d_r2q_map_ptr, d_r2q_map_ptr+numQueries, h_vec_val.begin());

    thrust::host_vector<unsigned int> h_vec_key(numQueries);
    thrust::copy(d_firsthit_idx_ptr, d_firsthit_idx_ptr+numQueries, h_vec_key.begin());

    float3* h_queries = state.h_actQs[batch_id];
    for (unsigned int i = 0; i < h_vec_val.size(); i++) {
      std::cout << h_vec_key[i] << "\t"
                << h_vec_val[i] << "\t"
                << h_queries[h_vec_val[i]].x << "\t"
                << h_queries[h_vec_val[i]].y << "\t"
                << h_queries[h_vec_val[i]].z
                << std::endl;
    }
  }

  return d_r2q_map_ptr;
}

void gatherQueries( RTNNState& state, thrust::device_ptr<unsigned int> d_indices_ptr, int batch_id ) {
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
  // leads to performance degradation, both |toGather| and |reorderPoints| are
  // disabled by default. |reorderPoints| are now removed.

  Timing::startTiming("gather queries");
    unsigned int numQueries = state.numActQueries[batch_id];

    // allocate device memory for reordered/gathered queries
    thrust::device_ptr<float3> d_reord_queries_ptr;
    allocThrustDevicePtr(&d_reord_queries_ptr, numQueries);

    // get pointer to original queries in device memory
    thrust::device_ptr<float3> d_orig_queries_ptr = thrust::device_pointer_cast(state.d_actQs[batch_id]);

    // gather by key, which is generated by the previous sort
    gatherByKey(d_indices_ptr, d_orig_queries_ptr, d_reord_queries_ptr, numQueries, state.stream[batch_id]);

    // if not samepq or partition is enabled (which will copy queries), then we can free the original query device memory
    if (!state.samepq || state.partition) CUDA_CHECK( cudaFree( (void*)state.d_actQs[batch_id] ) );
    state.d_actQs[batch_id] = thrust::raw_pointer_cast(d_reord_queries_ptr);
    //assert(state.params.points != state.params.queries);
  Timing::stopTiming(true);

  // Copy reordered queries to host for sanity check
  // if not samepq, free the original query host memory first
  if (!state.samepq || state.partition) delete state.h_actQs[batch_id];
  state.h_actQs[batch_id] = new float3[numQueries]; // don't overwrite h_points
  thrust::host_vector<float3> host_reord_queries(numQueries);
  thrust::copy(d_reord_queries_ptr, d_reord_queries_ptr+numQueries, state.h_actQs[batch_id]);
  //assert (state.h_points != state.h_queries);
}

