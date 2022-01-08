#include <iomanip>
#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>

#include <sutil/Exception.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "func.h"
#include "state.h"

int tokenize(std::string s, std::string del, float3** ndpoints, unsigned int lineId)
{
  int start = 0;
  int end = s.find(del);
  int dim = 0;

  std::vector<float> vcoords;
  while (end != -1) {
    float coord = std::stof(s.substr(start, end - start));
    //std::cout << coord << std::endl;
    if (ndpoints != nullptr) {
      vcoords.push_back(coord);
    }
    start = end + del.size();
    end = s.find(del, start);
    dim++;
  }
  float coord  = std::stof(s.substr(start, end - start));
  //std::cout << coord << std::endl;
  if (ndpoints != nullptr) {
    vcoords.push_back(coord);
  }
  dim++;

  assert(dim > 0);
  if ((dim % 3) != 0) dim = (dim/3+1)*3;

  if (ndpoints != nullptr) {
    for (int batch = 0; batch < dim/3; batch++) {
      float3 point = make_float3(vcoords[batch*3], vcoords[batch*3+1], vcoords[batch*3+2]);
      ndpoints[batch][lineId] = point;
    }
  }

  return dim;
}

float3** read_pc_data(const char* data_file, unsigned int* N, int* d) {
  std::ifstream file;

  file.open(data_file);
  if( !file.good() ) {
    std::cerr << "Could not read the frame data...\n";
    assert(0);
  }

  char line[1024];
  unsigned int lines = 0;
  int dim = 0;

  while (file.getline(line, 1024)) {
    if (lines == 0) {
      std::string str(line);
      dim = tokenize(str, ",", nullptr, 0);
    }
    lines++;
  }
  file.clear();
  file.seekg(0, std::ios::beg);

  *N = lines;
  *d = dim;

  float3** ndpoints = new float3*[dim/3];
  for (int i = 0; i < dim/3; i++) {
    ndpoints[i] = new float3[lines];
  }

  lines = 0;
  while (file.getline(line, 1024)) {
    std::string str(line);
    tokenize(str, ",", ndpoints, lines);

    //std::cerr << ndpoints[0][lines].x << "," << ndpoints[0][lines].y << "," << ndpoints[0][lines].z << std::endl;
    //std::cerr << ndpoints[1][lines].x << "," << ndpoints[1][lines].y << "," << ndpoints[1][lines].z << std::endl;
    lines++;
  }

  file.close();

  return ndpoints;
}

float3* read_pc_data(const char* data_file, unsigned int* N) {
  std::ifstream file;

  file.open(data_file);
  if( !file.good() ) {
    std::cerr << "Could not read the frame data...\n";
    assert(0);
  }

  char line[1024];
  unsigned int lines = 0;

  while (file.getline(line, 1024)) {
    lines++;
  }
  file.clear();
  file.seekg(0, std::ios::beg);
  *N = lines;

  float3* t_points = new float3[lines];

  lines = 0;
  while (file.getline(line, 1024)) {
    double x, y, z;

    sscanf(line, "%lf,%lf,%lf\n", &x, &y, &z);
    t_points[lines] = make_float3(x, y, z);
    //std::cerr << t_points[lines].x << ", " << t_points[lines].y << ", " << t_points[lines].z << std::endl;
    lines++;
  }

  file.close();

  return t_points;
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "\e[1mUsage:\e[0m " << argv0 << " [options]\n\n";
    

    std::cerr << "\e[1mBasic Options:\e[0m\n";
    std::cerr << "  --pfile           | -f      File for search points. By default it's also used as queries unless -q is speficied.\n";
    std::cerr << "  --qfile           | -q      File for queries.\n";
    std::cerr << "  --searchmode      | -sm     Search mode; can only be \"knn\" or \"radius\". Default is \"radius\". \n";
    std::cerr << "  --radius          | -r      Search radius. Default is 2.\n";
    std::cerr << "  --knn             | -k      Max K returned. Default is 50.\n";
    std::cerr << "  --device          | -d      Specify GPU ID. Default is 0.\n";
    std::cerr << "  --interleave      | -i      Allow interleaving kernel launches? Enable it for better performance. Default is true.\n";
    std::cerr << "  --msr             | -m      Enable end-to-end measurement? If true, disable CUDA synchronizations for more accurate time measurement (and higher performance). Default is true.\n";
    std::cerr << "  --check           | -c      Enable sanity check? Default is false.\n";

    std::cerr << "  --help            | -h      Print this usage message\n";


    std::cerr << "\n\e[1mAdvanced Options:\e[0m\n";

    std::cerr << "  --partition       | -p      Allow query partitioning? Enable it for better performance. Default is true.\n";
    std::cerr << "  --approx          | -a      Approximate query partitioning mode for KNN search. Range search is always exact. {0: no approx, i.e., 3D circumRadius for 3D search; 1: 2D circumRadius for 3D search; 2: equiVol approx in query partitioning)} See |radiusFromMegacell| function. Default is 2.\n";

    std::cerr << "  --autobatch       | -ab     Automatically determining how to batch partitions? Default is true.\n";
    std::cerr << "  --numbatch        | -nb     Specify the number of batches when batching partitions. It's used only if -ab is false. Default nb is -1, which uses the max available batch; otherwise the numebr of batches to launch = min(avail batches, nb).\n";

    std::cerr << "  --gassort         | -s      GAS-based query sort mode. {0: no sort. 1: 1D order. 2: ID order.} Default is 2.\n";
    std::cerr << "  --gsrRatio        | -sg     Radius ratio used in GAS sort. Default is 1.\n";
    std::cerr << "  --gather          | -g      Whether to gather queries after GAS sort? Default is false.\n";

    std::cerr << "  --pointsort       | -ps     Grid-based point sort mode. {0: no sort. 1: morton order. 2: raster order. 3: 1D order.} Default 1.\n";
    std::cerr << "  --querysort       | -qs     Grid-based query sort mode. {0: no sort. 1: morton order. 2: raster order. 3: 1D order.} Default 1.\n";

    std::cerr << "  --autocrratio     | -ac     Automatically determining crRatio (cell/radius ratio)? cellSize = radius / crRatio. cellSize is used to create the grid for sorting queries. Default is true.\n";
    std::cerr << "  --crratio         | -cr     Specify crRatio. It's used only if \'-ac\' is false. Default is 8.\n";
    std::cerr << "  --gpumemused      | -gmu    Specify GPU memory that's occupied by other jobs. This allows a better estimation of crRatio to avoid OOM errors. Default is 0.\n";

    exit( 0 );
}

void parseArgs( RTNNState& state,  int argc, char* argv[] ) {
  for( int i = 1; i < argc; ++i )
  {
      const std::string arg = argv[i];
      if( arg == "--help" || arg == "-h" )
      {
          printUsageAndExit( argv[0] );
      }
      else if( arg == "--pfile" || arg == "-f" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.pfile = argv[++i];
      }
      else if( arg == "--qfile" || arg == "-q" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.qfile = argv[++i];
      }
      else if( arg == "--knn" || arg == "-k" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.knn = atoi(argv[++i]);
      }
      else if( arg == "--searchmode" || arg == "-sm" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.searchMode = argv[++i];
          if ((state.searchMode != "knn") && (state.searchMode != "radius"))
              printUsageAndExit( argv[0] );
      }
      else if( arg == "--radius" || arg == "-r" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.radius = std::stof(argv[++i]);
          state.params.radius = state.radius; // this indicates the search radius of a launch
      }
      else if( arg == "--msr" || arg == "-m" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.msr = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--numbatch" || arg == "-nb" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.numOfBatches = atoi(argv[++i]); // used only if ab is not enabled; nb==-1 means using the max available batch, otherwise the # of batches to launch = min(avail batches, nb)
      }
      else if( arg == "--autobatch" || arg == "-ab" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.autoNB = (bool)(atoi(argv[++i])); // if enabled, ignore nb
      }
      else if( arg == "--autocrratio" || arg == "-ac" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.autoCR = (bool)(atoi(argv[++i])); // if enabled, ignore crRatio
      }
      else if( arg == "--partition" || arg == "-p" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.partition = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--interleave" || arg == "-i" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.interleave = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--device" || arg == "-d" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.device_id = atoi(argv[++i]);
      }
      else if( arg == "--gassort" || arg == "-s" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.qGasSortMode = atoi(argv[++i]);
          if (state.qGasSortMode > 2 || state.qGasSortMode < 0)
              printUsageAndExit( argv[0] );
      }
      else if( arg == "--pointsort" || arg == "-ps" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.pointSortMode = atoi(argv[++i]);
      }
      else if( arg == "--querysort" || arg == "-qs" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.querySortMode = atoi(argv[++i]);
      }
      else if( arg == "--crratio" || arg == "-cr" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.crRatio = std::stof(argv[++i]);
      }
      else if( arg == "--gpumemused" || arg == "-gmu" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.gpuMemUsed = std::stof(argv[++i]);
      }
      else if( arg == "--gather" || arg == "-g" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.toGather = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--approx" || arg == "-a" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.approxMode = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--check" || arg == "-c" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.sanCheck = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--gsrRatio" || arg == "-sg" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.gsrRatio = std::stof(argv[++i]);
          if (state.gsrRatio <= 0)
              printUsageAndExit( argv[0] );
      }
      else
      {
          std::cerr << "Unknown option '" << argv[i] << "'\n";
          printUsageAndExit( argv[0] );
      }
  }

  // if search mode is knn, overwrite knn
  if (state.searchMode == "knn")
    state.knn = K; // a macro

  bool sameData = (state.qfile.empty() || (state.qfile == state.pfile));
  bool sameSortMode = (state.pointSortMode == state.querySortMode);

  // TODO: currently p and q must be the same to support query partitioning
  if (state.partition) {
    assert(sameData);
  }

  // samepq indicates whether queries and points share the same host and device
  // memory (for now; partitioning will change it). even if p and q are the
  // same data, but if they have different sorting modes we can't have them
  // share the same device memory since sorting is in-place; they can't share
  // the same host memory either since host memory layout will be mutated to be
  // in-sync with device memory layout for sanity check purpose.
  if (sameSortMode && sameData) {
    state.samepq = true;
  }
}

void readData(RTNNState& state) {
  state.h_points = read_pc_data(state.pfile.c_str(), &state.numPoints);
  state.h_queries = state.h_points;
  state.numQueries = state.numPoints;

  if (!state.samepq) { // if can't share the host memory
    if (!state.qfile.empty() && (state.qfile != state.pfile)) {
      // if the underlying data are different, read it
      state.h_queries = read_pc_data(state.qfile.c_str(), &state.numQueries);
    } else {
      // if underlying data are the same, copy it
      state.h_queries = (float3*)malloc(state.numQueries * sizeof(float3));
      thrust::copy(state.h_points, state.h_points+state.numQueries, state.h_queries);
    }
  }
}

// this function returns the width of the inscribed cube (square) of a sphere (circle)
float maxInscribedWidth(float radius, int dim) {
  if (dim == 2) return radius/sqrt(2)*2;
  else if (dim == 3) return radius/sqrt(3)*2;
  else assert(0);
}

// this function returns the radius of the circumsphere (circumcircle) of a cube (square)
float minCircumscribedRadius(float width, int dim) {
  if (dim == 2) return width/2*sqrt(2);
  else if (dim == 3) return width/2*sqrt(3);
  else assert(0);
}

// this function returns the radius of a sphere (circle) with same volume of a cube (square)
float radiusEquiVolume(float width, int dim) {
  if (dim == 2) return width*sqrt(1/M_PI);
  else if (dim == 3) return width*cbrt(3/(4*M_PI));
  else assert(0);
}

float calcCRRatio(RTNNState& state) {
  //TODO: the crratio is determined from the points, not queries, for now. that
  //is, when sorting queries we will use the same ratio. ideally points and
  //queries should have different ratios if they are diferent. so we could
  //still have an OOM when sorting queries. maybe we shouldn't have the notion
  //of crRatio; we should determine individually for points and queries what
  //the cell size is.
  unsigned int N = state.numPoints;
  unsigned int Q = state.numQueries;

  int scale = (state.samepq ? 1 : 2);
  // Note: scale_ms is to loosen the aggressive estimation of the numOfCells here
  // and also actual # of cells will be greater due to meta cell alignment.

  // conservatively include both points and queries and one more copy for partitioned queries
  float pointDataSize = 3 * N * sizeof(float3);
  float returnDataSize = Q * state.knn * sizeof(unsigned int);

  // for sorting and partitioning, we will have to allocate 3 arrays that have
  // numOfCell elements and 7 arrays that have N elements.
  float particleArraysSize = 7 * N * sizeof(unsigned int) * scale;
  // TODO: conservatively estimate the gas size as 1.5 times the point size (better fit?)
  float gasSize = state.numPoints * sizeof(float3) * 1.5;
  float spaceAvail = state.totDRAMSize * 1024 * 1024 * 1024 - particleArraysSize - pointDataSize - returnDataSize - state.gpuMemUsed * 1024 * 1024;
  fprintf(stdout, "\tspaceAval: %.3f\n\tparticleArray: %.3f\n\tpointData: %.3f\n\treturnData: %.3f\n\tgpuMemused: %.3f\n\tgasSize: %.3f\n",
                   spaceAvail/1024/1024,
                   particleArraysSize/1024/1024,
                   pointDataSize/1024/1024,
                   returnDataSize/1024/1024,
                   state.gpuMemUsed,
                   gasSize/1014/1024);

  // total gas size + total sorting structure size <= avail mem
  // total gas size = (state.radius / (sqrt(3) * cellSize) + 1) * gasSize; // TODO (sqrt(2) for 2D)
  // total sorting structure size = sceneVolume / power(cellSize, 3) * (3 * sizeof(unsigned int) * scale);
  // it's a cubic equation. don't want to solve it analytically. let's do that
  // iteratively. the initial size is the smallest cell size that can
  // accommodate the entire gas or the entire sorting structures.

  float numOfBatches = spaceAvail / gasSize;
  float cellSizeLimitedByGAS = state.radius / (sqrt(3) * (numOfBatches - 1));

  // could |genGridInfo| too but doesn't matter
  float sceneVolume = (state.pMax.x - state.pMin.x) * (state.pMax.y - state.pMin.y) * (state.pMax.z - state.pMin.z);
  float numOfSortingCells = spaceAvail / (3 * sizeof(unsigned int) * scale);
  float cellSizeLimitedBySort = cbrt(sceneVolume / numOfSortingCells);

  float cellSize = std::max(cellSizeLimitedBySort, cellSizeLimitedByGAS);

  float curGASSize = 0;
  float curSortingSize = 0;
  float curTotalSize = 0;
  while (1) {
    numOfBatches = state.radius / (sqrt(3) * cellSize) + 1;
    curGASSize = numOfBatches * gasSize;

    GridInfo gridInfo;
    state.crRatio = state.radius / cellSize;
    state.Min = state.pMin;
    state.Max = state.pMax;
    numOfSortingCells = genGridInfo(state, N, gridInfo);
    curSortingSize = numOfSortingCells * (3 * sizeof(unsigned int) * scale);

    curTotalSize = curGASSize + curSortingSize;
    fprintf(stdout, "%f+%f=%f, %f\n", curGASSize/1024/1024, curSortingSize/1024/1024, curTotalSize/1024/1024, spaceAvail/1024/1024);
    if (curTotalSize < spaceAvail) break;
    cellSize *= 1.05;
  }

  float ratio = state.radius / cellSize;
  fprintf(stdout, "\tCalculated cellRadiusRatio: %f (%f, %f)\n", ratio, curGASSize/1024/1024, curSortingSize/1024/1024);
  fprintf(stdout, "\tMemory utilization: %.3f\n", 1 - (spaceAvail-curTotalSize)/(state.totDRAMSize*1024*1024*1024));

  return ratio;
}

void initBatches(RTNNState& state) {
  if (state.autoCR) {
    // TODO: should we just use a fixed cell size/ratio? an overly small cell
    // increases the sort cost, but probably mean little for range search. need
    // some exhaustive testing.
    state.crRatio = calcCRRatio(state);
  }

  // see |genCellMask| for the logic behind this.
  float cellSize = state.radius / state.crRatio;
  float maxWidth = maxInscribedWidth(state.radius, 3); // for 3D
  int maxIter = (int)floorf(maxWidth / (2 * cellSize) - 1);
  int maxBatchCount = maxIter + 2; // could be fewer than this.
  state.maxBatchCount = maxBatchCount;

  state.gas_handle = new OptixTraversableHandle[maxBatchCount];
  state.d_gas_output_buffer = new CUdeviceptr[maxBatchCount]();
  state.stream = new cudaStream_t[maxBatchCount];
  state.d_r2q_map = new unsigned int*[maxBatchCount]();
  state.numActQueries = new unsigned int[maxBatchCount];
  state.launchRadius = new float[maxBatchCount];
  state.partThd = new float[maxBatchCount];
  state.h_res = new void*[maxBatchCount]();
  state.d_res = new void*[maxBatchCount]();
  state.d_actQs = new float3*[maxBatchCount]();
  state.h_actQs = new float3*[maxBatchCount]();
  state.d_aabb = new void*[maxBatchCount]();
  state.d_firsthit_idx = new void*[maxBatchCount]();
  state.d_temp_buffer_gas = new void*[maxBatchCount]();
  state.d_buffer_temp_output_gas_and_compacted_size = new void*[maxBatchCount]();
  state.pipeline = new OptixPipeline[maxBatchCount];

  for (int i = 0; i < maxBatchCount; i++)
      CUDA_CHECK( cudaStreamCreate( &state.stream[i] ) );
}

bool isClose(float3 a, float3 b) {
  if (fabs(a.x - b.x) < 0.001 && fabs(a.y - b.y) < 0.001 && fabs(a.z - b.z) < 0.001) return true;
  else return false;
}
