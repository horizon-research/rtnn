#include <iostream>
#include <queue>
#include <algorithm>
#include <unordered_set>

#include "state.h"

typedef std::pair<float, unsigned int> knn_res_t;
class Compare
{
  public:
    bool operator() (knn_res_t a, knn_res_t b)
    {
      return a.first < b.first;
    }
};
typedef std::priority_queue<knn_res_t, std::vector<knn_res_t>, Compare> knn_queue;

void sanityCheck_knn( WhittedState& state, void* data ) {
  bool printRes = false;
  srand(time(NULL));
  std::vector<unsigned int> randQ {rand() % state.numQueries, rand() % state.numQueries, rand() % state.numQueries, rand() % state.numQueries, rand() % state.numQueries, 97808, 1192803};

  for (unsigned int q = 0; q < state.numQueries; q++) {
    if (std::find(randQ.begin(), randQ.end(), q) == randQ.end()) continue;
    std::cout << "Sanity check for query " << q << std::endl;

    // generate ground truth res
    float3 query = state.h_queries[q];
    knn_queue topKQ;
    unsigned int size = 0;
    for (unsigned int p = 0; p < state.numPoints; p++) {
      float3 point = state.h_points[p];
      float3 diff = query - point;
      float dists = dot(diff, diff);
      if ((dists > 0) && (dists < state.params.radius*state.params.radius)) {
        knn_res_t res = std::make_pair(dists, p);
        if (size < state.params.limit) {
          topKQ.push(res);
          size++;
        } else if (dists < topKQ.top().first) {
          topKQ.pop();
          topKQ.push(res);
        }
      }
    }

    if (printRes) std::cout << "GT: ";
    std::unordered_set<unsigned int> gt_idxs;
    for (unsigned int i = 0; i < size; i++) {
      if (printRes) std::cout << "[" << sqrt(topKQ.top().first) << ", " << topKQ.top().second << "] ";
      gt_idxs.insert(topKQ.top().second);
      topKQ.pop();
    }
    //std::cout << std::endl;

    // get the GPU data and check
    if (printRes) std::cout << "RTX: ";
    std::unordered_set<unsigned int> gpu_idxs;
    for (unsigned int n = 0; n < state.params.limit; n++) {
      unsigned int p = static_cast<unsigned int*>( data )[ q * state.params.limit + n ];
      if (p == UINT_MAX) break;
      else {
        gpu_idxs.insert(p);
        if (printRes) {
          float3 diff = state.h_points[p] - query;
          float dists = dot(diff, diff);
          std::cout << "[" << sqrt(dists) << ", " << p << "] ";
        }
      }
    }
    if (printRes) std::cout << std::endl;

    // TODO: there could some numerical precision issue.
    if (gt_idxs != gpu_idxs) {std::cout << "Incorrect!" << std::endl;}
  }
  std::cerr << "Sanity check done." << std::endl;
}

void sanityCheck( WhittedState& state, void* data ) {
  // this is stateful in that it relies on state.params.limit

  unsigned int totalNeighbors = 0;
  unsigned int totalWrongNeighbors = 0;
  double totalWrongDist = 0;
  for (unsigned int q = 0; q < state.numQueries; q++) {
    for (unsigned int n = 0; n < state.params.limit; n++) {
      unsigned int p = reinterpret_cast<unsigned int*>( data )[ q * state.params.limit + n ];
      //std::cout << p << std::endl; break;
      if (p == UINT_MAX) break;
      else {
        totalNeighbors++;
        float3 diff = state.h_points[p] - state.h_queries[q];
        float dists = dot(diff, diff);
        if (dists > state.params.radius*state.params.radius) {
          //fprintf(stdout, "Point %u [%f, %f, %f] is not a neighbor of query %u [%f, %f, %f]. Dist is %lf.\n", p, state.h_points[p].x, state.h_points[p].y, state.h_points[p].z, q, state.h_points[q].x, state.h_points[q].y, state.h_points[q].z, sqrt(dists));
          totalWrongNeighbors++;
          totalWrongDist += sqrt(dists);
          //exit(1);
        }
        //std::cout << sqrt(dists) << " ";
      }
      //std::cout << p << " ";
    }
    //std::cout << "\n";
  }
  std::cerr << "Sanity check done." << std::endl;
  std::cerr << "Avg neighbor/query: " << (float)totalNeighbors/state.numQueries << std::endl;
  std::cerr << "Total wrong neighbors: " << totalWrongNeighbors << std::endl;
  if (totalWrongNeighbors != 0) std::cerr << "Avg wrong dist: " << totalWrongDist / totalWrongNeighbors << std::endl;
}


