#include <sutil/vec_math.h>

#include "helper_mortonCode.h"
#include "helper_linearIndex.h"
#include "grid.h"

#include <stdio.h>

/* GPU code */
inline __host__ __device__ uint ToCellIndex_MortonMetaGrid(const GridInfo &GridInfo, int3 gridCell)
{
  //int3 temp = gridCell;

  int3 metaGridCell = make_int3(
    gridCell.x / GridInfo.meta_grid_dim,
    gridCell.y / GridInfo.meta_grid_dim,
    gridCell.z / GridInfo.meta_grid_dim);

  gridCell.x %= GridInfo.meta_grid_dim;
  gridCell.y %= GridInfo.meta_grid_dim;
  gridCell.z %= GridInfo.meta_grid_dim;
  uint metaGridIndex = CellIndicesToLinearIndex(GridInfo.MetaGridDimension, metaGridCell);

  //if (temp.x == 503 && temp.y == 33 && temp.z == 645)
  //  printf("(%d, %d, %d), (%d, %d, %d), %u, %u, %u\n", metaGridCell.x, metaGridCell.y, metaGridCell.z, gridCell.x, gridCell.y, gridCell.z, metaGridIndex, metaGridIndex * GridInfo.meta_grid_size, MortonCode3(gridCell.x, gridCell.y, gridCell.z));

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
  //printf("%u, %u\n", particleIndex, GridInfo.ParticleCount);

  float3 gridCellF = (particles[particleIndex] - GridInfo.GridMin) * GridInfo.GridDelta;
  int3 gridCell = make_int3(int(gridCellF.x), int(gridCellF.y), int(gridCellF.z));

  unsigned int cellIndex = (gridCell.x * GridInfo.GridDimension.y + gridCell.y) * GridInfo.GridDimension.z + gridCell.z;
  particleCellIndices[particleIndex] = cellIndex;

  //float3 query = particles[particleIndex];
  //float3 b = make_float3(-14.238000, 1.946000, 3.575000);
  //if (fabs(query.x - b.x) < 0.001 && fabs(query.y - b.y) < 0.001 && fabs(query.z - b.z) < 0.001) {
  //  printf("particle [%f, %f, %f], [%d, %d, %d] in cell %u\n", query.x, query.y, query.z, gridCell.x, gridCell.y, gridCell.z, cellIndex);
  //}

  // this stores the within-cell sorted indices of particles
  localSortedIndices[particleIndex] = atomicAdd(&cellParticleCounts[cellIndex], 1);

  //if (cellIndex == 6054598)
  //  printf("cell 6054598 has %u particles [%f, %f, %f]. Dist: %f\n", cellParticleCounts[cellIndex], query.x, query.y, query.z, sqrt((query.x - b.x) * (query.x - b.x) + (query.y - b.y) * (query.y - b.y) + (query.z - b.z) * (query.z - b.z)));

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

  //float3 query = particles[particleIndex];
  //float3 b = make_float3(21.618000, -0.005000, -13.505000);
  //if (fabs(query.x - b.x) < 0.001 && fabs(query.y - b.y) < 0.001 && fabs(query.z - b.z) < 0.001) {
  //  printf("particle [%f, %f, %f], [%d, %d, %d] in cell %u\n", query.x, query.y, query.z, gridCell.x, gridCell.y, gridCell.z, cellIndex);
  //}

  // this stores the within-cell sorted indices of particles
  localSortedIndices[particleIndex] = atomicAdd(&cellParticleCounts[cellIndex], 1);

  //printf("%u, %u, (%d, %d, %d)\n", particleIndex, cellIndex, gridCell.x, gridCell.y, gridCell.z);
}

__global__ void kCountingSortIndices(
  const GridInfo GridInfo,
  const uint* particleCellIndices,
  const uint* cellOffsets,
  const uint* localSortedIndices,
  uint* posInSortedPoints
)
{
  uint particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= GridInfo.ParticleCount) return;

  uint gridCellIndex = particleCellIndices[particleIndex];

  uint sortIndex = localSortedIndices[particleIndex] + cellOffsets[gridCellIndex];
  posInSortedPoints[particleIndex] = sortIndex;

  //printf("%u, %u, %u, %u, %u\n", particleIndex, gridCellIndex, localSortedIndices[particleIndex], cellOffsets[gridCellIndex], sortIndex);
}

__global__ void kCountingSortIndices_genMask(
  const GridInfo GridInfo,
  const uint* particleCellIndices,
  const uint* cellOffsets,
  const uint* localSortedIndices,
  uint* posInSortedPoints,
  char* cellMask,
  char* rayMask
)
{
  uint particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= GridInfo.ParticleCount) return;

  uint gridCellIndex = particleCellIndices[particleIndex];

  uint sortIndex = localSortedIndices[particleIndex] + cellOffsets[gridCellIndex];
  posInSortedPoints[particleIndex] = sortIndex;

  rayMask[particleIndex] = cellMask[gridCellIndex];

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

void kCountingSortIndices_genMask(unsigned int numOfBlocks, unsigned int threadsPerBlock,
      GridInfo gridInfo,
      unsigned int* d_ParticleCellIndices,
      unsigned int* d_CellOffsets,
      unsigned int* d_LocalSortedIndices,
      unsigned int* d_posInSortedPoints,
      char* cellMask,
      char* rayMask
      ) {
  kCountingSortIndices_genMask <<<numOfBlocks, threadsPerBlock>>> (
      gridInfo,
      d_ParticleCellIndices,
      d_CellOffsets,
      d_LocalSortedIndices,
      d_posInSortedPoints,
      cellMask,
      rayMask
      );
}

uint kToCellIndex_MortonMetaGrid(const GridInfo& gridInfo, int3 cell) {
  if (cell.x == 503 && cell.y == 33 && cell.z == 645) printf("here\n");
  return ToCellIndex_MortonMetaGrid(gridInfo, cell);
}

