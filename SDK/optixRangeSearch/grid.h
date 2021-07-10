#pragma once

struct GridInfo
{
  float3 GridMin;
  unsigned int ParticleCount;
  float3 GridDelta;
  uint3 GridDimension;
  uint3 MetaGridDimension;
  unsigned int meta_grid_dim;
  unsigned int meta_grid_size;
};
