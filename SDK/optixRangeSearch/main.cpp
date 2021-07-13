//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <cuda_runtime.h>

#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <sutil/Timing.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include <iomanip>
#include <cstring>
#include <fstream>
#include <string>
#include <random>
#include <cstdlib>
#include <queue>
#include <unordered_set>

#include "optixRangeSearch.h"
#include "state.h"
#include "func.h"
#include "grid.h"

int main( int argc, char* argv[] )
{
  WhittedState state;
  state.radius = 2; // this is the given radius
  state.params.radius = 2; // this indicates the search radius of a launch
  state.params.knn = 50;

  parseArgs( state, argc, argv );

  readData(state);

  std::cout << "========================================" << std::endl;
  std::cout << "numPoints: " << state.numPoints << std::endl;
  std::cout << "numQueries: " << state.numTotalQueries << std::endl;
  std::cout << "searchMode: " << state.searchMode << std::endl;
  std::cout << "radius: " << state.radius << std::endl;
  std::cout << "K: " << state.params.knn << std::endl;
  std::cout << "Same P and Q? " << std::boolalpha << state.samepq << std::endl;
  std::cout << "Partition? " << std::boolalpha << state.partition << std::endl;
  std::cout << "qGasSortMode: " << state.qGasSortMode << std::endl;
  std::cout << "pointSortMode: " << std::boolalpha << state.pointSortMode << std::endl;
  std::cout << "querySortMode: " << std::boolalpha << state.querySortMode << std::endl;
  std::cout << "cellRadiusRatio: " << std::boolalpha << state.crRatio << std::endl; // only useful when preSort == 1/2
  std::cout << "sortingGAS: " << state.sortingGAS << std::endl; // only useful when qGasSortMode != 0
  std::cout << "Gather? " << std::boolalpha << state.toGather << std::endl;
  std::cout << "reorderPoints? " << std::boolalpha << state.reorderPoints << std::endl; // only useful under samepq and toGather
  std::cout << "========================================" << std::endl << std::endl;

  try
  {
    // Set up CUDA device and stream
    setupCUDA(state);

    Timing::reset();

    uploadData(state);

    sortParticles(state, POINT, state.pointSortMode); // if partition is enabled, we do it here.
    if (!state.samepq) sortParticles(state, QUERY, state.querySortMode); // when samepq, queries are sorted using the point sort mode.

    setupOptiX(state);

    int batch;
    if (state.partition) batch = 2;
    else {
      batch = 1;
      state.numActQueries[0] = state.numQueries;
      state.d_actQs[0] = state.params.queries;
      state.h_actQs[0] = state.h_queries;
    }

    Timing::startTiming("total search time");
    //TODO: try a better scheduling here.
    //for (int i = 0; i < batch; i++) {
    for (int i = batch - 1; i >= 0; i--) {
      fprintf(stdout, "\n************** Batch %u **************\n", i);
      state.numQueries = state.numActQueries[i];
      state.params.queries = state.d_actQs[i];
      state.h_queries = state.h_actQs[i];

      // it's possible that certain batches have 0 query (e.g., state.partThd too low).
      if (state.numQueries == 0) continue;

      // create the GAS using the current order of points and the launchRadius of the current batch.
      createGeometry (state, i); // batch_id ignored if not partition.

      if (state.qGasSortMode) gasSortSearch(state, i);

      search(state, i);
    }
    CUDA_SYNC_CHECK();
    Timing::stopTiming(true);

    cleanupState( state );
  }
  catch( std::exception& e )
  {
      std::cerr << "Caught exception: " << e.what() << "\n";
      return 1;
  }

  return 0;
}
