#include "helper_mortonCode.h"
#include "helper_linearIndex.h"
#include "optixRangeSearch.h"
//#include "Types.h"

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//typedef unsigned int uint;

//struct GridInfo
//{
//  float3 GridMin;
//  uint ParticleCount;
//  float3 GridDelta;
//  uint3 GridDimension;
//  uint3 MetaGridDimension;
//  float SquaredSearchRadius;
//};

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
  //unsigned int cellIndex = ToCellIndex_MortonMetaGrid(GridInfo, gridCell);
  unsigned int cellIndex = (gridCell.x * GridInfo.GridDimension.y + gridCell.y) * GridInfo.GridDimension.z + gridCell.z;
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

//void computeMinMax(WhittedState& state)
//{
//  thrust::host_vector<int3> h_MinMax(2);
//  h_MinMax[0] = make_int3(std::numeric_limits<int>().max(), std::numeric_limits<int>().max(), std::numeric_limits<int>().max());
//  h_MinMax[1] = make_int3(std::numeric_limits<int>().min(), std::numeric_limits<int>().min(), std::numeric_limits<int>().min());
//  thrust::device_vector<int3> d_MinMax(2);
//  //d_MinMax.resize(2);
//
//  //CudaHelper::MemcpyHostToDevice(data, CudaHelper::GetPointer(d_MinMax), 2);
//
//  int threadsPerBlock = 64;
//  int numOfBlocks = state.params.numPrims / threadsPerBlock + 1;
//  kComputeMinMax <<<numOfBlocks, threadsPerBlock>>> (
//      state.params.points,
//      state.params.numPrims,
//      state.params.radius,
//      thrust::raw_pointer_cast(&d_MinMax[0]),
//      thrust::raw_pointer_cast(&d_MinMax[1])
//      );
//  //CudaHelper::CheckLastError();
//  //CudaHelper::DeviceSynchronize();
// 
//  //CudaHelper::MemcpyDeviceToHost(CudaHelper::GetPointer(d_MinMax), data, 2);
//  thrust::copy(d_MinMax.begin(), d_MinMax.end(), h_MinMax.begin());
//
//  int3 minCell = h_MinMax[0];
//  int3 maxCell = h_MinMax[1];
// 
//  state.Min.x = minCell.x * state.params.radius;
//  state.Min.y = minCell.y * state.params.radius;
//  state.Min.z = minCell.z * state.params.radius;
// 
//  state.Max.x = maxCell.x * state.params.radius;
//  state.Max.y = maxCell.y * state.params.radius;
//  state.Max.z = maxCell.z * state.params.radius;
//}

void kInsertParticles_Morton(unsigned int numOfBlocks, unsigned int threadsPerBlock, GridInfo gridInfo, float3* points, unsigned int* d_ParticleCellIndices, unsigned int* d_CellParticleCounts, unsigned int* d_TempSortIndices) {
  kInsertParticles_Morton <<<numOfBlocks, threadsPerBlock>>> (
      gridInfo,
      points,
      d_ParticleCellIndices,
      d_CellParticleCounts,
      d_TempSortIndices
      );
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

//void computeCellInformation(WhittedState& state) {
//  float3 sceneMin = state.Min;
//  float3 sceneMax = state.Max;
//
//  GridInfo gridInfo;
//  gridInfo.ParticleCount = static_cast<uint>(state.params.numPrims);
//  gridInfo.SquaredSearchRadius = state.params.radius * state.params.radius;
//  gridInfo.GridMin = sceneMin;
//
//  float cellSize = state.params.radius;
//  float3 gridSize = sceneMax - sceneMin;
//  gridInfo.GridDimension.x = static_cast<unsigned int>(ceil(gridSize.x / cellSize));
//  gridInfo.GridDimension.y = static_cast<unsigned int>(ceil(gridSize.y / cellSize));
//  gridInfo.GridDimension.z = static_cast<unsigned int>(ceil(gridSize.z / cellSize));
//
//  //Increase grid by 2 cells in each direciton (+4 in each dimension) to skip bounds checks in the kernel
//  gridInfo.GridDimension.x += 4;
//  gridInfo.GridDimension.y += 4;
//  gridInfo.GridDimension.z += 4;
//  gridInfo.GridMin -= make_float3(cellSize, cellSize, cellSize) * (float)2;
//
//  //One meta grid cell contains 8x8x8 grild cells. (512)
//  gridInfo.MetaGridDimension.x = static_cast<unsigned int>(ceil(gridInfo.GridDimension.x / (float)CUDA_META_GRID_GROUP_SIZE));
//  gridInfo.MetaGridDimension.y = static_cast<unsigned int>(ceil(gridInfo.GridDimension.y / (float)CUDA_META_GRID_GROUP_SIZE));
//  gridInfo.MetaGridDimension.z = static_cast<unsigned int>(ceil(gridInfo.GridDimension.z / (float)CUDA_META_GRID_GROUP_SIZE));
//
//  // Adjust grid size to multiple of cell size
//  gridSize.x = gridInfo.GridDimension.x * cellSize;
//  gridSize.y = gridInfo.GridDimension.y * cellSize;
//  gridSize.z = gridInfo.GridDimension.z * cellSize;
//
//  gridInfo.GridDelta.x = gridInfo.GridDimension.x / gridSize.x;
//  gridInfo.GridDelta.y = gridInfo.GridDimension.y / gridSize.y;
//  gridInfo.GridDelta.z = gridInfo.GridDimension.z / gridSize.z;
//
//  uint numberOfCells = (gridInfo.MetaGridDimension.x * gridInfo.MetaGridDimension.y * gridInfo.MetaGridDimension.z) * CUDA_META_GRID_BLOCK_SIZE;
//  thrust::device_vector<unsigned int> d_ParticleCellIndices(state.params.numPrims);
//  thrust::device_vector<unsigned int> d_SortIndices(state.params.numPrims);
//
//  thrust::device_vector<unsigned int> d_CellParticleCounts(numberOfCells);
//  thrust::device_vector<unsigned int> d_TempSortIndices(state.params.numPrims);
//
//  //CudaHelper::CheckLastError();
//  //CudaHelper::DeviceSynchronize();
//
//  cudaMemset(thrust::raw_pointer_cast(&d_CellParticleCounts[0]), 0, state.params.numPrims * sizeof(unsigned int));
//
//  //CudaHelper::CheckLastError();
//  //CudaHelper::DeviceSynchronize();
//
//  int threadsPerBlock = 64;
//  int numOfBlocks = state.params.numPrims / threadsPerBlock + 1;
//  kInsertParticles_Morton <<<numOfBlocks, threadsPerBlock>> > (
//      gridInfo,
//      state.params.points,
//      thrust::raw_pointer_cast(&d_ParticleCellIndices[0]),
//      thrust::raw_pointer_cast(&d_CellParticleCounts[0]),
//      thrust::raw_pointer_cast(&d_TempSortIndices[0])
//      );
//
//  //CudaHelper::CheckLastError();
//  //CudaHelper::DeviceSynchronize();
//
//  thrust::device_vector<unsigned int> d_CellOffsets(numberOfCells);
//  thrust::exclusive_scan(
//    d_CellParticleCounts.begin(),
//    d_CellParticleCounts.end(),
//    d_CellOffsets.begin());
//  //CudaHelper::DeviceSynchronize();
//  
//  kCountingSortIndices <<<numOfBlocks, threadsPerBlock>>> (
//      gridInfo,
//      thrust::raw_pointer_cast(&d_ParticleCellIndices[0]),
//      thrust::raw_pointer_cast(&d_CellOffsets[0]),
//      thrust::raw_pointer_cast(&d_TempSortIndices[0]),
//      thrust::raw_pointer_cast(&d_SortIndices[0])
//      );
//
//  thrust::device_vector<unsigned int> d_ReversedSortIndices = d_SortIndices;
//  //CudaHelper::DeviceSynchronize();
//
//  thrust::sort_by_key(d_ReversedSortIndices.begin(), d_ReversedSortIndices.end(), thrust::device_pointer_cast(state.params.points));
//}
