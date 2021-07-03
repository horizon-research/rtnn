#pragma once
#include <cuda_runtime.h>

__host__ __device__ inline uint CellIndicesToLinearIndex(
	uint3 &cellDimensions, 
	uint3 &xyz
)
{
	return xyz.z * cellDimensions.y * cellDimensions.x + xyz.y * cellDimensions.x + xyz.x;
}

__host__ __device__ inline uint CellIndicesToLinearIndex(
	const uint3&cellDimensions, 
	int3 &xyz
)
{
	return xyz.z * cellDimensions.y * cellDimensions.x + xyz.y * cellDimensions.x + xyz.x;
}

__host__ __device__ inline void LinearCellIndexTo3DIndices(
	const uint3 &cellDimensions,
	const uint linearIndex,
	uint3 &xyz
)
{
	xyz.z = linearIndex / (cellDimensions.y * cellDimensions.x);
	xyz.y = (linearIndex % (cellDimensions.y * cellDimensions.x)) / (cellDimensions.x);
	xyz.x = (linearIndex % (cellDimensions.y * cellDimensions.x)) % cellDimensions.x;
}

__host__ __device__ inline uint3 LinearCellIndexTo3DIndices(
	const uint3 &cellDimensions,
	const uint linearIndex)
{
	uint3 xyz;
	xyz.z = linearIndex / (cellDimensions.y * cellDimensions.x);
	xyz.y = (linearIndex % (cellDimensions.y * cellDimensions.x)) / (cellDimensions.x);
	xyz.x = (linearIndex % (cellDimensions.y * cellDimensions.x)) % cellDimensions.x;
	return xyz;
}

__host__ __device__ inline int3 LinearCellIndexTo3DIndicesint3(
	const uint3 &cellDimensions, 
	const uint &linearIndex)
{
	int3 xyz;
	xyz.z = linearIndex / (cellDimensions.y * cellDimensions.x);
	xyz.y = (linearIndex % (cellDimensions.y * cellDimensions.x)) / (cellDimensions.x);
	xyz.x = (linearIndex % (cellDimensions.y * cellDimensions.x)) % cellDimensions.x;
	return xyz;
}
