#include <cuda_runtime.h>

#include <sutil/Exception.h>
#include <sutil/vec_math.h>
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

#include "optixRangeSearch.h"
#include "func.h"
#include "state.h"
#include "grid.h"

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

unsigned int genGridInfo(WhittedState& state, unsigned int N, GridInfo& gridInfo) {
  float3 sceneMin = state.Min;
  float3 sceneMax = state.Max;

  gridInfo.ParticleCount = N;
  gridInfo.GridMin = sceneMin;

  // TODO: maybe crRatio should be automatically determined based on memory?
  float cellSize = state.radius/state.crRatio;
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
  gridInfo.meta_grid_dim = (int)pow(2, floorf(log2(std::min({gridInfo.GridDimension.x, gridInfo.GridDimension.y, gridInfo.GridDimension.z}))))/2;
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
  fprintf(stdout, "\tMeta Grid dimension: %u, %u, %u\n", gridInfo.MetaGridDimension.x, gridInfo.MetaGridDimension.y, gridInfo.MetaGridDimension.z);
  fprintf(stdout, "\t# of cells in a meta grid: %u\n", gridInfo.meta_grid_dim);
  //fprintf(stdout, "\tGridDelta: %f, %f, %f\n", gridInfo.GridDelta.x, gridInfo.GridDelta.y, gridInfo.GridDelta.z);
  fprintf(stdout, "\tNumber of cells: %u\n", numberOfCells);
  fprintf(stdout, "\tCell size: %f\n", cellSize);

  // update GridDimension so that it can be used in the kernels (otherwise raster order is incorrect)
  gridInfo.GridDimension.x = gridInfo.MetaGridDimension.x * gridInfo.meta_grid_dim;
  gridInfo.GridDimension.y = gridInfo.MetaGridDimension.y * gridInfo.meta_grid_dim;
  gridInfo.GridDimension.z = gridInfo.MetaGridDimension.z * gridInfo.meta_grid_dim;
  return numberOfCells;
}

float getWidthFromIter(int iter, float cellSize) {
  // to be absolutely certain, we add 2 (not 1) to iter to accommodate points
  // at the edges of the central cell. width means there are K points within
  // the width^3 AABB, whose center is the center point of the current cell.
  // for corner points in the cell, its width^3 AABB might have less than count
  // # of points if the point distrition becomes dramatically sparse outside of
  // the current AABB. we empirically observe no issue with >1M points, but
  // with about ~100K points this could be an issue.

  return (iter * 2 + 2) * cellSize;
}

uint kToCellIndex_MortonMetaGrid(const GridInfo&, int3);

unsigned int getCellIdx(GridInfo gridInfo, int ix, int iy, int iz, bool morton) {
  if (morton) // z-order sort
    return kToCellIndex_MortonMetaGrid(gridInfo, make_int3(ix, iy, iz));
  else // raster order
    return (ix * gridInfo.GridDimension.y + iy) * gridInfo.GridDimension.z + iz;
}

bool oob(GridInfo gridInfo, int ix, int iy, int iz) {
  if (ix < 0 || ix >= gridInfo.GridDimension.x
   || iy < 0 || iy >= gridInfo.GridDimension.y
   || iz < 0 || iz >= gridInfo.GridDimension.z)
    return true;
  else return false;
}

void addCount(int& count, unsigned int* h_CellParticleCounts, GridInfo gridInfo, int ix, int iy, int iz, bool morton) {
    if (oob(gridInfo, ix, iy, iz)) return;

    unsigned int iCellIdx = getCellIdx(gridInfo, ix, iy, iz, morton);
    count += h_CellParticleCounts[iCellIdx];
    //if (ix == 87 && iy == 22 && iz == 358) printf("[%d, %d, %d]\n", ix, iy, iz, iCellIdx);
}

void genMask (WhittedState& state, unsigned int* h_CellParticleCounts, unsigned int numberOfCells, GridInfo& gridInfo, unsigned int N, bool morton) {
  // TODO: this whole thing needs to be done in CUDA.

  std::vector<unsigned int> cellSearchSize(numberOfCells, 0);
  float cellSize = state.radius / state.crRatio;

  // this it the max width of a square that can be enclosed by the sphere. if
  // beyond this, knn will fall back to the original radius and radius search
  // can't be approximated.
  float maxWidth = state.radius / sqrt(2) * 2;

  int maxIter = (int)floorf(maxWidth / (2 * cellSize) - 1);
  int histCount = maxIter + 3; // 0: empty cell counts; 1 -- maxIter+1: real counts; maxIter+2: full search counts.

  //printf("%d, %f\n", maxIter, maxWidth);

  unsigned int* searchSizeHist = new unsigned int[histCount];
  memset(searchSizeHist, 0, sizeof (unsigned int) * histCount);

  for (int x = 0; x < gridInfo.GridDimension.x; x++) {
    for (int y = 0; y < gridInfo.GridDimension.y; y++) {
      for (int z = 0; z < gridInfo.GridDimension.z; z++) {
        // now let's check;
        int cellIndex = getCellIdx(gridInfo, x, y, z, morton);
        //if (x == 87 && y == 22 && z == 358) printf("cell %d has %d particles\n", cellIndex, h_CellParticleCounts[cellIndex]);
        assert(cellIndex <= numberOfCells);
        if (h_CellParticleCounts[cellIndex] == 0) continue;

        int iter = 0;
        int count = 0;
        addCount(count, h_CellParticleCounts, gridInfo, x, y, z, morton);

	    // in radius search we want to completely skip dist calc and sphere
	    // check in GPU (bottleneck) so we constrain the maxWidth such that the
	    // AABB is completely enclosed by the sphere. in knn search dist calc
	    // can't be skipped and is not the bottleneck anyway (invoking IS
	    // programs is) so we relax the maxWidth to give points more
	    // opportunity to find a smaller search radius.
        int xmin = x;
        int xmax = x;
        int ymin = y;
        int ymax = y;
        int zmin = z;
        int zmax = z;

        while(1) {
	      // TODO: there could be corner cases here, e.g., maxWidth is very
	      // small, cellSize will be 0 (same as uninitialized).
          float width = getWidthFromIter(iter, cellSize);

          //if (iter > maxIter) {
          if (width > maxWidth) {
            cellSearchSize[cellIndex] = iter + 1; // if width > maxWidth, we need to do a full search.
            searchSizeHist[iter + 1]++;
            break;
          }
          else if (count >= (state.knn + 1)) {
            // + 1 because the count in h_CellParticleCounts includes the point
            // itself whereas our KNN search isn't going to return itself!
            cellSearchSize[cellIndex] = iter + 1; // + 1 so that iter being 0 doesn't become full search.
            searchSizeHist[iter + 1]++;
            break;
          }
          else {
            iter++;
            //count = 0;
          }
          //if (x == 87 && y == 22 && z == 358) printf("%d, %d\n", iter, count);

          int ix, iy, iz;

          iz = zmin - 1;
          for (ix = xmin; ix <= xmax; ix++) {
            for (iy = ymin; iy <= ymax; iy++) {
              addCount(count, h_CellParticleCounts, gridInfo, ix, iy, iz, morton);
            }
          }

          iz = zmax + 1;
          for (ix = xmin; ix <= xmax; ix++) {
            for (iy = ymin; iy <= ymax; iy++) {
              addCount(count, h_CellParticleCounts, gridInfo, ix, iy, iz, morton);
            }
          }

          ix = xmin - 1;
          for (iy = ymin; iy <= ymax; iy++) {
            for (iz = zmin; iz <= zmax; iz++) {
              addCount(count, h_CellParticleCounts, gridInfo, ix, iy, iz, morton);
            }
          }

          ix = xmax + 1;
          for (iy = ymin; iy <= ymax; iy++) {
            for (iz = zmin; iz <= zmax; iz++) {
              addCount(count, h_CellParticleCounts, gridInfo, ix, iy, iz, morton);
            }
          }

          iy = ymin - 1;
          for (ix = xmin; ix <= xmax; ix++) {
            for (iz = zmin; iz <= zmax; iz++) {
              addCount(count, h_CellParticleCounts, gridInfo, ix, iy, iz, morton);
            }
          }

          iy = ymax + 1;
          for (ix = xmin; ix <= xmax; ix++) {
            for (iz = zmin; iz <= zmax; iz++) {
              addCount(count, h_CellParticleCounts, gridInfo, ix, iy, iz, morton);
            }
          }

          xmin--;
          xmax++;
          ymin--;
          ymax++;
          zmin--;
          zmax++;

          addCount(count, h_CellParticleCounts, gridInfo, xmin, ymin, zmin, morton);
          addCount(count, h_CellParticleCounts, gridInfo, xmin, ymin, zmax, morton);
          addCount(count, h_CellParticleCounts, gridInfo, xmin, ymax, zmin, morton);
          addCount(count, h_CellParticleCounts, gridInfo, xmin, ymax, zmax, morton);
          addCount(count, h_CellParticleCounts, gridInfo, xmax, ymin, zmin, morton);
          addCount(count, h_CellParticleCounts, gridInfo, xmax, ymin, zmax, morton);
          addCount(count, h_CellParticleCounts, gridInfo, xmax, ymax, zmin, morton);
          addCount(count, h_CellParticleCounts, gridInfo, xmax, ymax, zmax, morton);

          //for (int ix = x-iter; ix <= x+iter; ix++) {
          //  for (int iy = y-iter; iy <= y+iter; iy++) {
          //    for (int iz = z-iter; iz <= z+iter; iz++) {
          //      if (ix < 0 || ix >= gridInfo.GridDimension.x || iy < 0 || iy >= gridInfo.GridDimension.y || iz < 0 || iz >= gridInfo.GridDimension.z) continue;
          //      else {
          //        unsigned int iCellIdx;
          //        if (morton) // z-order sort
          //          iCellIdx = kToCellIndex_MortonMetaGrid(gridInfo, make_int3(ix, iy, iz));
          //        else // raster order
          //          iCellIdx = (ix * gridInfo.GridDimension.y + iy) * gridInfo.GridDimension.z + iz;
          //        count += h_CellParticleCounts[iCellIdx];
          //      }
          //    }
          //  }
          //}
        }
      }
    }
  }

  // setup the batches
  state.numOfBatches = histCount - 1;
  fprintf(stdout, "\tNumber of batches: %d\n", state.numOfBatches);

  // the last partThd won't be used -- radius will be state.radius for the last batch.
  for (unsigned int i = 0; i < state.numOfBatches; i++) {
    state.partThd[i] = getWidthFromIter(i, cellSize); 

    //fprintf(stdout, "%u, %u, %f\n", i, searchSizeHist[i + 1], state.partThd[i]);
  }

  state.cellMask = new char[numberOfCells];
  for (unsigned int i = 0; i < numberOfCells; i++) {
    //if (cellSearchSize[i] != 0) fprintf(stdout, "%u, %u\n",
    //                                    cellSearchSize[i],
    //                                    h_CellParticleCounts[i]
    //                                   );

    // TODO: need a better decision logic, e.g., through a histogram and some sort of cost model.
    if (cellSearchSize[i] != 0) {
      //if (i == 6054598) printf("search size for cell 6054598: %d\n", cellSearchSize[i]);
      state.cellMask[i] = cellSearchSize[i] - 1;
    }
  }
}

void sortGenPartInfo(WhittedState& state,
                 unsigned int N,
                 bool morton,
                 unsigned int numberOfCells,
                 unsigned int numOfBlocks,
                 unsigned int threadsPerBlock,
                 GridInfo gridInfo,
                 thrust::device_ptr<unsigned int> d_CellParticleCounts_ptr,
                 thrust::device_ptr<unsigned int> d_ParticleCellIndices_ptr,
                 thrust::device_ptr<unsigned int> d_CellOffsets_ptr,
                 thrust::device_ptr<unsigned int> d_LocalSortedIndices_ptr,
                 thrust::device_ptr<unsigned int> d_posInSortedPoints_ptr
                )
{
    thrust::host_vector<unsigned int> h_CellParticleCounts(numberOfCells);
    thrust::copy(d_CellParticleCounts_ptr, d_CellParticleCounts_ptr + numberOfCells, h_CellParticleCounts.begin());

    genMask(state, h_CellParticleCounts.data(), numberOfCells, gridInfo, N, morton);

    thrust::device_ptr<char> d_rayMask = getThrustDeviceCharPtr(N);
    thrust::device_ptr<char> d_cellMask = getThrustDeviceCharPtr(numberOfCells);
    thrust::copy(state.cellMask, state.cellMask + numberOfCells, d_cellMask);
    delete state.cellMask;

    kCountingSortIndices_genMask(numOfBlocks,
                                 threadsPerBlock,
                                 gridInfo,
                                 thrust::raw_pointer_cast(d_ParticleCellIndices_ptr),
                                 thrust::raw_pointer_cast(d_CellOffsets_ptr),
                                 thrust::raw_pointer_cast(d_LocalSortedIndices_ptr),
                                 thrust::raw_pointer_cast(d_posInSortedPoints_ptr),
                                 thrust::raw_pointer_cast(d_cellMask),
                                 thrust::raw_pointer_cast(d_rayMask)
                                );

    // make a copy of the keys since they are useless after the first sort. no
    // need to use stable sort since the keys are unique, so masks and the
    // queries are gauranteed to be sorted in exactly the same way.
    // TODO: Can we do away with th extra copy by replacing sort by key with scatter? That'll need new space too...
    thrust::device_ptr<unsigned int> d_posInSortedPoints_ptr_copy = getThrustDevicePtr(N);
    //thrust::copy(d_posInSortedPoints_ptr, d_posInSortedPoints_ptr + N, d_posInSortedPoints_ptr_copy); // not sure why this doesn't link.
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( thrust::raw_pointer_cast(d_posInSortedPoints_ptr_copy) ),
                thrust::raw_pointer_cast(d_posInSortedPoints_ptr),
                N * sizeof( unsigned int ),
                cudaMemcpyDeviceToDevice
    ) );


    thrust::host_vector<char> h_rayMask(N);
    thrust::copy(d_rayMask, d_rayMask + N, h_rayMask.begin());
    thrust::host_vector<unsigned int> h_ParticleCellIndices(N);
    thrust::copy(d_ParticleCellIndices_ptr, d_ParticleCellIndices_ptr + N, h_ParticleCellIndices.begin());

    //for (unsigned int i = 0; i < N; i++) {
    //  float3 query = state.h_queries[i];
    //  if (isClose(query, make_float3(-57.230999, 2.710000, 9.608000))) {
    //    printf("particle [%f, %f, %f], %d, in cell %u\n", query.x, query.y, query.z, h_rayMask[i], h_ParticleCellIndices[i]);
    //    break;
    //  }
    //}
    //exit(1);

    // sort the ray masks as well the same way as query sorting.
    sortByKey(d_posInSortedPoints_ptr_copy, d_rayMask, N);
    CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_posInSortedPoints_ptr_copy) ) );

    // this MUST happen right after sorting the masks and before copy so that the queries and the masks are consistent!!!
    sortByKey(d_posInSortedPoints_ptr, thrust::device_pointer_cast(state.params.queries), N);

    for (int i = 0; i < state.numOfBatches; i++) {
      state.numActQueries[i] = countById(d_rayMask, N, i);

      // the min check is technically not needed except for the last batch, for
      // which we should never need to search beyond state.radius. float
      // conversion is because std::sqrt returns a double for an integral input
      // (https://en.cppreference.com/w/cpp/numeric/math/sqrt), and std::min
      // can't compare float with double.
      // TODO: last batch needs to be 2

      if (state.searchMode == "knn") state.launchRadius[i] = std::min((float)(state.partThd[i] / 2 * sqrt(2)), state.radius);
      else state.launchRadius[i] = std::min(state.partThd[i] / 2, state.radius);

      // can't free state.params.queries, because it points to the points too.
      // same applies to state.h_queries. state.params.queries from this point
      // on will only be used to point to device queries used in kernels, and
      // will be set right before launch using d_actQs.
      thrust::device_ptr<float3> d_actQs = getThrustDeviceF3Ptr(state.numActQueries[i]);
      copyIfIdMatch(state.params.queries, N, d_rayMask, d_actQs, i);
      state.d_actQs[i] = thrust::raw_pointer_cast(d_actQs);

      // Copy the active queries to host.
      state.h_actQs[i] = new float3[state.numActQueries[i]];
      thrust::copy(d_actQs, d_actQs + state.numActQueries[i], state.h_actQs[i]);
    }

    CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_rayMask) ) );
    CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_cellMask) ) );
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

  GridInfo gridInfo;
  unsigned int numberOfCells = genGridInfo(state, N, gridInfo);

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

  thrust::device_ptr<unsigned int> d_CellOffsets_ptr = getThrustDevicePtr(numberOfCells);
  fillByValue(d_CellOffsets_ptr, numberOfCells, 0); // need to initialize it even for exclusive scan
  exclusiveScan(d_CellParticleCounts_ptr, numberOfCells, d_CellOffsets_ptr);

  thrust::device_ptr<unsigned int> d_posInSortedPoints_ptr = getThrustDevicePtr(N);
  // if samepq and partition is enabled, do it here. we are partitioning points, but it's the same as queries.
  if (state.partition) {
    // normal particle sorting is done here too.
    sortGenPartInfo(state,
                    N,
                    morton,
                    numberOfCells,
                    numOfBlocks,
                    threadsPerBlock,
                    gridInfo,
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

  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_ParticleCellIndices_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_posInSortedPoints_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_CellOffsets_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_LocalSortedIndices_ptr) ) );
  CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(d_CellParticleCounts_ptr) ) );

  bool debug = false;
  if (debug) {
    thrust::host_vector<uint> temp(N);
    thrust::copy(d_posInSortedPoints_ptr, d_posInSortedPoints_ptr + N, temp.begin());
    for (unsigned int i = 0; i < N; i++) {
      fprintf(stdout, "%u (%f, %f, %f)\n", temp[i], h_particles[i].x, h_particles[i].y, h_particles[i].z);
    }

    //for (unsigned int i = 0; i < numberOfCells; i++) {
    //  if (h_CellParticleCounts[i] != 0) fprintf(stdout, "%u, %u\n", i, h_CellParticleCounts[i]);
    //}
  }
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

  thrust::device_ptr<float> d_key_ptr = getThrustDeviceF1Ptr(state.numQueries);
  state.d_1dsort_key = thrust::raw_pointer_cast(d_key_ptr); // just so we have a handle to free it later
  thrust::copy(h_key.begin(), h_key.end(), d_key_ptr);

  // actual sort
  thrust::device_ptr<float3> d_particles_ptr = thrust::device_pointer_cast(particles);
  sortByKey( d_key_ptr, d_particles_ptr, N );

  // TODO: lift it outside of this function and combine with other sorts?
  // copy the sorted queries to host so that we build the GAS in the same order
  // note that the h_queries at this point still point to what h_points points to
  thrust::copy(d_particles_ptr, d_particles_ptr + N, h_particles);
}

void sortParticles ( WhittedState& state, ParticleType type, int sortMode ) {
  // 0: no sort
  // 1: z-order sort
  // 2: raster sort
  // 3: 1D sort
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

thrust::device_ptr<unsigned int> sortQueriesByFHCoord( WhittedState& state, thrust::device_ptr<unsigned int> d_firsthit_idx_ptr, int batch_id ) {
  // this is sorting queries by the x/y/z coordinate of the first hit primitives.
  unsigned int numQueries = state.numActQueries[batch_id];

  Timing::startTiming("gas-sort queries init");
    // allocate device memory for storing the keys, which will be generated by a gather and used in sort_by_keys
    thrust::device_ptr<float> d_key_ptr = getThrustDeviceF1Ptr(numQueries);
    state.d_fhsort_key = thrust::raw_pointer_cast(d_key_ptr); // just so we have a handle to free it later
  
    // create indices for gather and upload to device
    thrust::host_vector<float> h_orig_points_1d(numQueries);
    // TODO: do this in CUDA
    for (unsigned int i = 0; i < numQueries; i++) {
      h_orig_points_1d[i] = state.h_points[i].z; // could be other dimensions
    }
    thrust::device_vector<float> d_orig_points_1d = h_orig_points_1d;

    // initialize a sequence to be sorted, which will become the r2q map.
    thrust::device_ptr<unsigned int> d_r2q_map_ptr = getThrustDevicePtr(numQueries);
    genSeqDevice(d_r2q_map_ptr, numQueries, state.stream[batch_id]);
  Timing::stopTiming(true);
  
  Timing::startTiming("gas-sort queries");
    // TODO: do thrust work in a stream: https://forums.developer.nvidia.com/t/thrust-and-streams/53199
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

thrust::device_ptr<unsigned int> sortQueriesByFHIdx( WhittedState& state, thrust::device_ptr<unsigned int> d_firsthit_idx_ptr, int batch_id ) {
  // this is sorting queries just by the first hit primitive IDs
  unsigned int numQueries = state.numActQueries[batch_id];

  // initialize a sequence to be sorted, which will become the r2q map
  Timing::startTiming("gas-sort queries init");
    thrust::device_ptr<unsigned int> d_r2q_map_ptr = getThrustDevicePtr(numQueries);
    genSeqDevice(d_r2q_map_ptr, numQueries, state.stream[batch_id]);
  Timing::stopTiming(true);

  Timing::startTiming("gas-sort queries");
    sortByKey( d_firsthit_idx_ptr, d_r2q_map_ptr, numQueries, state.stream[batch_id] );
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

void gatherQueries( WhittedState& state, thrust::device_ptr<unsigned int> d_indices_ptr, int batch_id ) {
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
    thrust::device_ptr<float3> d_reord_queries_ptr = getThrustDeviceF3Ptr(numQueries);

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

