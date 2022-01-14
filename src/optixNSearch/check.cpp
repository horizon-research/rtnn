#include <iostream>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <iterator>

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

void sanityCheckKNN( RTNNState& state, int batch_id ) {
  bool printRes = false;
  srand(time(NULL));
  std::vector<unsigned int> randQ {rand() % state.numQueries, rand() % state.numQueries, rand() % state.numQueries, rand() % state.numQueries, rand() % state.numQueries};
  //std::vector<unsigned int> randQ {46401};
  //std::vector<unsigned int> randQ {1606};

  for (unsigned int q = 0; q < state.numQueries; q++) {
    if (std::find(randQ.begin(), randQ.end(), q) == randQ.end()) continue;
    float3 query = state.h_queries[q];

    // generate ground truth res
    knn_queue topKQ;
    unsigned int size = 0;
    for (unsigned int p = 0; p < state.numPoints; p++) {
      float3 point = state.h_points[p];
      float3 diff = query - point;
      float dists = dot(diff, diff);
      if ((dists > 0) && (dists < state.radius * state.radius)) {
        knn_res_t res = std::make_pair(dists, p);
        if (size < state.knn) {
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
    std::unordered_set<float> gt_dists;
    for (unsigned int i = 0; i < size; i++) {
      if (printRes) std::cout << "[" << sqrt(topKQ.top().first) << ", " << topKQ.top().second << "] ";
      gt_idxs.insert(topKQ.top().second);
      gt_dists.insert(sqrt(topKQ.top().first));
      topKQ.pop();
    }
    if (printRes) std::cout << std::endl;

    // get the GPU data and check
    if (printRes) std::cout << "RTX: ";
    std::unordered_set<unsigned int> gpu_idxs;
    std::unordered_set<float> gpu_dists;
    for (unsigned int n = 0; n < state.knn; n++) {
      unsigned int p = static_cast<unsigned int*>( state.h_res[batch_id] )[ q * state.knn + n ];
      if (p == UINT_MAX) break;
      else {
        float3 diff = state.h_points[p] - query;
        float dists = dot(diff, diff);
        gpu_idxs.insert(p);
        gpu_dists.insert(sqrt(dists));
        if (printRes) {
          std::cout << "[" << sqrt(dists) << ", " << p << "] ";
        }
      }
    }
    if (printRes) std::cout << std::endl;

    // TODO: there are cases where there are multiple points are very close and
    // so depending on the order they are searched the result would be
    // different, although both CPU and GPU should both have implemented the
    // same FP standard. need to revisit this.
    // https://www.techiedelight.com/print-set-unordered_set-cpp/
    if (gt_dists != gpu_dists) {
      fprintf(stdout, "Incorrect query [%u] %f, %f, %f\n", q, query.x, query.y, query.z);
      std::cout << "GT:\n";
      std::copy(gt_dists.begin(),
            gt_dists.end(),
            std::ostream_iterator<float>(std::cout, " "));
            std::cout << "\n\n";
      std::cout << "RTX:\n";
      std::copy(gpu_dists.begin(),
            gpu_dists.end(),
            std::ostream_iterator<float>(std::cout, " "));
            std::cout << "\n\n";
      exit(1);
    }
  }
  std::cerr << "Sanity check done." << std::endl;
}

void sanityCheckRadius( RTNNState& state, int batch_id ) {
  // this is stateful in that it relies on state.params.limit
  // TODO: now use knn rather than limit. good?

  unsigned int totalNeighbors = 0;
  unsigned int totalWrongNeighbors = 0;
  double totalWrongDist = 0;
  for (unsigned int q = 0; q < state.numQueries; q++) {
    for (unsigned int n = 0; n < state.knn; n++) {
      unsigned int p = reinterpret_cast<unsigned int*>( state.h_res[batch_id] )[ q * state.knn + n ];
      //std::cout << p << std::endl; break;
      if (p == UINT_MAX) break;
      else {
        totalNeighbors++;
        float3 diff = state.h_points[p] - state.h_queries[q];
        float dists = dot(diff, diff);
        if (dists > state.radius * state.radius) {
          fprintf(stdout, "Point %u [%f, %f, %f] is not a neighbor of query %u [%f, %f, %f]. Dist is %lf.\n",
            p, state.h_points[p].x, state.h_points[p].y, state.h_points[p].z,
            q, state.h_queries[q].x, state.h_queries[q].y, state.h_queries[q].z,
            sqrt(dists));
          totalWrongNeighbors++;
          totalWrongDist += sqrt(dists);
          exit(1);
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

void checkFilteredQueries(RTNNState& state) {
  // sanity check for filtered queries
  for (unsigned int q = 0; q < state.numFltQs; q++) {
    for (unsigned int p = 0; p < state.numPoints; p++) {
      float3 diff = state.h_points[p] - state.h_fltQs[q];
      float dists = dot(diff, diff);
      if (dists < state.radius * state.radius) {
        fprintf(stdout, "Query %u [%f, %f, %f] shouldn't be filtered; conflicting query %u [%f, %f, %f]. Dist is %lf.\n",
          q, state.h_queries[q].x, state.h_queries[q].y, state.h_queries[q].z,
          p, state.h_points[p].x, state.h_points[p].y, state.h_points[p].z,
          sqrt(dists));
      }
    }
  }

  std::cerr << "Filtered queries sanity check done." << std::endl;
}

void sanityCheck(RTNNState& state) {
  for (int i = 0; i < state.numOfBatches; i++) {
  //for (int i = 0; i < 1; i++) {
    state.numQueries = state.numActQueries[i];
    state.h_queries = state.h_actQs[i];

    // for empty batches, skip sanity check.
    if (state.numQueries == 0) continue;

    if (state.searchMode == "radius") sanityCheckRadius( state, i );
    else sanityCheckKNN( state, i );

  }
  //checkFilteredQueries(state);
}
