#include "helper_mortonCode.h"
#include "helper_linearIndex.h"
#include "optixRangeSearch.h"

#include <stdio.h>

/* GPU code */
inline __device__ uint ToCellIndex_MortonMetaGrid(const GridInfo &GridInfo, int3 gridCell)
{
  int3 metaGridCell = make_int3(
    gridCell.x / GridInfo.meta_grid_dim,
    gridCell.y / GridInfo.meta_grid_dim,
    gridCell.z / GridInfo.meta_grid_dim);

  gridCell.x %= GridInfo.meta_grid_dim;
  gridCell.y %= GridInfo.meta_grid_dim;
  gridCell.z %= GridInfo.meta_grid_dim;
  uint metaGridIndex = CellIndicesToLinearIndex(GridInfo.MetaGridDimension, metaGridCell);

  //printf("(%d, %d, %d), (%d, %d, %d), %u, %u, %u\n", metaGridCell.x, metaGridCell.y, metaGridCell.z, gridCell.x, gridCell.y, gridCell.z, metaGridIndex, metaGridIndex * GridInfo.meta_grid_size, MortonCode3(gridCell.x, gridCell.y, gridCell.z));

  return metaGridIndex * GridInfo.meta_grid_size + MortonCode3(gridCell.x, gridCell.y, gridCell.z);
}

__global__ void kComputeMinMax(
  const float3 *particles,
  unsigned int particleCount,
  int3 *minCell,
  int3 *maxCell
)
{
  unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= particleCount) return;
  const float3 particle = particles[particleIndex];

  int3 cell;
  // convert float to int since atomicMin/Max has no native float version
  // TODO: mind the float to int conversion issue
  cell.x = (int)floorf(particle.x); // floorf returns a float
  cell.y = (int)floorf(particle.y);
  cell.z = (int)floorf(particle.z);

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
  uint *posInSortedPoints
)
{
  uint particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= GridInfo.ParticleCount) return;

  uint gridCellIndex = particleCellIndices[particleIndex];

  uint sortIndex = localSortedIndices[particleIndex] + cellOffsets[gridCellIndex];
  posInSortedPoints[particleIndex] = sortIndex;

  //printf("%u, %u, %u, %u, %u\n", particleIndex, gridCellIndex, localSortedIndices[particleIndex], cellOffsets[gridCellIndex], sortIndex);
}



/* CPU code */
void kComputeMinMax (unsigned int numOfBlocks, unsigned int threadsPerBlock, float3* points, unsigned int numPrims, int3* d_MinMax_0, int3* d_MinMax_1) {
  kComputeMinMax <<<numOfBlocks, threadsPerBlock>>> (
      points,
      numPrims,
      d_MinMax_0,
      d_MinMax_1
      );
}

void kInsertParticles(unsigned int numOfBlocks, unsigned int threadsPerBlock, GridInfo gridInfo, float3* points, unsigned int* d_ParticleCellIndices, unsigned int* d_CellParticleCounts, unsigned int* d_TempSortIndices, bool morton) {
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
      unsigned int* d_LocalSortedIndices,
      unsigned int* d_posInSortedPoints
      ) {
  kCountingSortIndices <<<numOfBlocks, threadsPerBlock>>> (
      gridInfo,
      d_ParticleCellIndices,
      d_CellOffsets,
      d_LocalSortedIndices,
      d_posInSortedPoints
      );
}

