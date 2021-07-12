#include <iomanip>
#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>

#include <sutil/Exception.h>
#include <sutil/vec_math.h>

#include <thrust/copy.h>

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
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file          | -f <filename>   File for point cloud input\n";
    std::cerr << "         --searchmode    | -sm             Search mode; can only be \"knn\" or \"radius\" \n";
    std::cerr << "         --radius        | -r              Search radius\n";
    std::cerr << "         --knn           | -k              Max K returned\n";
    std::cerr << "         --samepq        | -spq            Same points and queries?\n";
    std::cerr << "         --device        | -d              Which GPU to use?\n";
    std::cerr << "         --gassort       | -s              GAS-based query sort mode\n";
    std::cerr << "         --pointsort     | -ps             Point sort mode\n";
    std::cerr << "         --querysort     | -qs             Query sort mode\n";
    std::cerr << "         --crratio       | -cr             cell/radius ratio\n";
    std::cerr << "         --sortingGAS    | -sg             Param for SortingGAS\n";
    std::cerr << "         --gather        | -g              Whether to gather queries after sort \n";
    std::cerr << "         --reorderpoints | -rp             Whether to reorder points after query sort \n";
    std::cerr << "         --help          | -h              Print this usage message\n";
    exit( 0 );
}

void parseArgs( WhittedState& state,  int argc, char* argv[] ) {
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
          state.params.knn = atoi(argv[++i]);
      }
      else if( arg == "--searchmode" || arg == "-sm" ) // need to be after --knn so that we can overwrite params.knn if needed
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.searchMode = argv[++i];
          if ((state.searchMode.compare("knn") != 0) && (state.searchMode.compare("radius") != 0))
              printUsageAndExit( argv[0] );
      }
      else if( arg == "--radius" || arg == "-r" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.params.radius = std::stof(argv[++i]);
      }
      else if( arg == "--partition" || arg == "-p" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.partition = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--samepq" || arg == "-spq" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.samepq = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--device" || arg == "-d" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.device_id = atoi(argv[++i]);
      }
      else if( arg == "--qgassort" || arg == "-s" )
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
      else if( arg == "--gather" || arg == "-g" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.toGather = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--reorderpoints" || arg == "-rp" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.reorderPoints = (bool)(atoi(argv[++i]));
      }
      else if( arg == "--sortingGAS" || arg == "-sg" )
      {
          if( i >= argc - 1 )
              printUsageAndExit( argv[0] );
          state.sortingGAS = std::stof(argv[++i]);
          if (state.sortingGAS <= 0)
              printUsageAndExit( argv[0] );
      }
      else
      {
          std::cerr << "Unknown option '" << argv[i] << "'\n";
          printUsageAndExit( argv[0] );
      }
  }

  // do a round of sanity check here
  if (state.searchMode.compare("knn") == 0) {
    state.params.knn = K; // a macro
  }

  if (state.partition) assert(state.samepq);
}

void setupCUDA( WhittedState& state) {
  int32_t device_count = 0;
  CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
  std::cerr << "\tTotal GPUs visible: " << device_count << std::endl;
  
  cudaDeviceProp prop;
  CUDA_CHECK( cudaGetDeviceProperties ( &prop, state.device_id ) );
  CUDA_CHECK( cudaSetDevice( state.device_id ) );
  std::cerr << "\tUsing [" << state.device_id << "]: " << prop.name << std::endl;

  CUDA_CHECK( cudaStreamCreate( &state.stream ) );
}

void readData(WhittedState& state) {
  // p and q files being the same dones't mean samepq have to be true. we can
  // still set it to be false to evaluate different reordering policies on
  // points and queries separately.

  state.h_points = read_pc_data(state.pfile.c_str(), &state.numPoints);
  //state.h_ndpoints = read_pc_data(state.pfile.c_str(), &state.numPoints, &state.dim);
  state.numQueries = state.numPoints;
  if (state.samepq) state.h_queries = state.h_points;
  else {
    state.h_queries = (float3*)malloc(state.numQueries * sizeof(float3));
    thrust::copy(state.h_points, state.h_points+state.numQueries, state.h_queries);
  }

  if (!state.qfile.empty()) {
    state.h_queries = read_pc_data(state.qfile.c_str(), &state.numQueries);
    assert(state.h_points != state.h_queries);
    // overwrite the samepq option from commandline
    state.samepq = false;

    //int query_dim;
    //state.h_ndqueries = read_pc_data(state.qfile.c_str(), &state.numQueries, &query_dim);
    //assert(query_dim == state.dim);
  }

  state.numTotalQueries = state.numQueries;
}

