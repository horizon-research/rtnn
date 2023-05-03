#include <optix.h>
#include <sutil/vec_math.h>

__global__ void kGenAABB_t (
      const float3* points,
      float radius,
      float* radii,
      unsigned int N,
      OptixAabb* aabb
)
{
  unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= N) return;

  float3 center = points[particleIndex];
  float actual_radius = radii == NULL ? radius : radii[particleIndex];

  float3 m_min = center - actual_radius;
  float3 m_max = center + actual_radius;

  aabb[particleIndex] =
  {
    m_min.x, m_min.y, m_min.z,
    m_max.x, m_max.y, m_max.z
  };
}

void kGenAABB(
  float3* points, 
  float radius,
  float* radii, 
  unsigned int numPrims, 
  OptixAabb* d_aabb, 
  cudaStream_t stream
) {
  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = numPrims / threadsPerBlock + 1;

  kGenAABB_t <<<numOfBlocks, threadsPerBlock, 0, stream>>> (
      points,
      radius,
      radii,
      numPrims,
      d_aabb
     );
}
