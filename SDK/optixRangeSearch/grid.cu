#include "helper_mortonCode.h"
#include "helper_linearIndex.h"
#include "optixRangeSearch.h"

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/* GPU code */
inline __device__ uint ToCellIndex_MortonMetaGrid(const GridInfo &GridInfo, int3 gridCell)
{
  int3 metaGridCell = make_int3(
    gridCell.x / CUDA_META_GRID_GROUP_SIZE,
    gridCell.y / CUDA_META_GRID_GROUP_SIZE,
    gridCell.z / CUDA_META_GRID_GROUP_SIZE);

  gridCell.x %= CUDA_META_GRID_GROUP_SIZE;
  gridCell.y %= CUDA_META_GRID_GROUP_SIZE;
  gridCell.z %= CUDA_META_GRID_GROUP_SIZE;
  uint metaGridIndex = CellIndicesToLinearIndex(GridInfo.MetaGridDimension, metaGridCell);
  return metaGridIndex * CUDA_META_GRID_BLOCK_SIZE + MortonCode3(gridCell.x, gridCell.y, gridCell.z);
}

__global__ void kComputeMinMax(
  const float3 *particles,
  uint particleCount,
  float searchRadius,
  int3 *minCell,
  int3 *maxCell
)
{
  uint particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= particleCount) return;
  const float3 particle = particles[particleIndex];

  int3 cell;
  cell.x = (int)floor(particle.x / searchRadius);
  cell.y = (int)floor(particle.y / searchRadius);
  cell.z = (int)floor(particle.z / searchRadius);

  atomicMin(&(minCell->x), cell.x);
  atomicMin(&(minCell->y), cell.y);
  atomicMin(&(minCell->z), cell.z);

  atomicMax(&(maxCell->x), cell.x);
  atomicMax(&(maxCell->y), cell.y);
  atomicMax(&(maxCell->z), cell.z);

  //printf("%d %d %d Min: %d %d %d Max: %d %d %d \n", cell.x, cell.y, cell.z, minCell->x, minCell->y, minCell->z, maxCell->x, maxCell->y, maxCell->z);
}

__global__ void kInsertParticles_Raster(
  const GridInfo GridInfo,
  const float3 *particles,
  unsigned int *particleCellIndices,
  unsigned int *cellParticleCounts,
  unsigned int *localSortedIndices
)
{
  unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= GridInfo.ParticleCount) return;

  float3 gridCellF = (particles[particleIndex] - GridInfo.GridMin) * GridInfo.GridDelta;
  int3 gridCell = make_int3(int(gridCellF.x), int(gridCellF.y), int(gridCellF.z));
  unsigned int cellIndex = (gridCell.x * GridInfo.GridDimension.y + gridCell.y) * GridInfo.GridDimension.z + gridCell.z;
  particleCellIndices[particleIndex] = cellIndex;

  // this stores the within-cell sorted indices of particles
  localSortedIndices[particleIndex] = atomicAdd(&cellParticleCounts[cellIndex], 1);

  //printf("%u, %u, (%d, %d, %d)\n", particleIndex, cellIndex, gridCell.x, gridCell.y, gridCell.z);
}

__global__ void kInsertParticles_Morton(
  const GridInfo GridInfo,
  const float3 *particles,
  unsigned int *particleCellIndices,
  unsigned int *cellParticleCounts,
  unsigned int *localSortedIndices
)
{
  unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= GridInfo.ParticleCount) return;

  float3 gridCellF = (particles[particleIndex] - GridInfo.GridMin) * GridInfo.GridDelta;
  int3 gridCell = make_int3(int(gridCellF.x), int(gridCellF.y), int(gridCellF.z));
  unsigned int cellIndex = ToCellIndex_MortonMetaGrid(GridInfo, gridCell);
  particleCellIndices[particleIndex] = cellIndex;

  // this stores the within-cell sorted indices of particles
  localSortedIndices[particleIndex] = atomicAdd(&cellParticleCounts[cellIndex], 1);

  //printf("%u, %u, (%d, %d, %d)\n", particleIndex, cellIndex, gridCell.x, gridCell.y, gridCell.z);
}

__global__ void kCountingSortIndices(
  const GridInfo GridInfo,
  const uint *particleCellIndices,
  const uint *cellOffsets,
  const uint *localSortedIndices,
  uint *sortIndicesDest,
  uint *posInSortedPoints
)
{
  uint particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= GridInfo.ParticleCount) return;

  uint gridCellIndex = particleCellIndices[particleIndex];

  uint sortIndex = localSortedIndices[particleIndex] + cellOffsets[gridCellIndex];
  sortIndicesDest[sortIndex] = particleIndex; // use this if we gather later
  posInSortedPoints[particleIndex] = sortIndex; // use this if we sort by key later

  //printf("%u, %u, %u, %u, %u\n", particleIndex, gridCellIndex, localSortedIndices[particleIndex], cellOffsets[gridCellIndex], sortIndex);
}



/* CPU code */
void kComputeMinMax (unsigned int numOfBlocks, unsigned int threadsPerBlock, float3* points, unsigned int numPrims, float radius, int3* d_MinMax_0, int3* d_MinMax_1) {
  kComputeMinMax <<<numOfBlocks, threadsPerBlock>>> (
      points,
      numPrims,
      radius,
      d_MinMax_0,
      d_MinMax_1
      );
}

void kInsertParticles_Morton(unsigned int numOfBlocks, unsigned int threadsPerBlock, GridInfo gridInfo, float3* points, unsigned int* d_ParticleCellIndices, unsigned int* d_CellParticleCounts, unsigned int* d_TempSortIndices, bool morton) {
  if (morton) {
    kInsertParticles_Morton <<<numOfBlocks, threadsPerBlock>>> (
        gridInfo,
        points,
        d_ParticleCellIndices,
        d_CellParticleCounts,
        d_TempSortIndices
        );
  } else {
    kInsertParticles_Raster<<<numOfBlocks, threadsPerBlock>>> (
        gridInfo,
        points,
        d_ParticleCellIndices,
        d_CellParticleCounts,
        d_TempSortIndices
        );
  }
}

void kCountingSortIndices(unsigned int numOfBlocks, unsigned int threadsPerBlock,
      GridInfo gridInfo,
      unsigned int* d_ParticleCellIndices,
      unsigned int* d_CellOffsets,
      unsigned int* d_TempSortIndices,
      unsigned int* d_SortIndices,
      unsigned int* d_posInSortedPoints
      ) {
  kCountingSortIndices <<<numOfBlocks, threadsPerBlock>>> (
      gridInfo,
      d_ParticleCellIndices,
      d_CellOffsets,
      d_TempSortIndices,
      d_SortIndices,
      d_posInSortedPoints
      );
}

void exclusiveScan(thrust::device_ptr<unsigned int> d_CellParticleCounts_ptr, unsigned int N, thrust::device_ptr<unsigned int> d_CellOffsets_ptr) {
  thrust::exclusive_scan(
    d_CellParticleCounts_ptr,
    d_CellParticleCounts_ptr + N,
    d_CellOffsets_ptr);
}

void fillByValue(thrust::device_ptr<unsigned int> d_CellParticleCounts_ptr, unsigned int N, int value) {
  thrust::fill(d_CellParticleCounts_ptr, d_CellParticleCounts_ptr + N, value);
}
