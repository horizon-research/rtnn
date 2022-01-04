#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# These CUDA files need to be compiled as OBJ, not as PTX
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/thrust_helper.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/grid.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/aabb.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

OPTIX_add_sample_executable( optixRangeSearch target_name
  #${SAMPLES_CUDA_DIR}/helpers.h
  main.cpp
  search.cpp
  optix.cpp
  sort.cpp
  check.cpp
  util.cpp
  camera.cu
  geometry.cu
  thrust_helper.cu
  grid.cu
  aabb.cu
  optixRangeSearch.h
  state.h
  grid.h
  helper_linearIndex.h
  helper_mortonCode.h
  #OPTIONS -rdc true
)

link_directories(${PCL_LIBRARY_DIRS})

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )

message(STATUS ${KNN})
if(KNN)
  #https://stackoverflow.com/questions/9017573/define-preprocessor-macro-through-cmake
  add_compile_definitions(K=${KNN})

  # TODO: copy sutil.cpp so that we don't need to recompile it all the time
  # https://forums.developer.nvidia.com/t/optix-compile-error-in-ptxstring/70636/2
  set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS} \\\n  \"-DK=${KNN}\",")
  #message(STATUS ${CUDA_NVRTC_OPTIONS})
  # regenerate the header. pay attention to the paths. https://cmake.org/cmake/help/latest/command/configure_file.html.
  configure_file(../sampleConfig.h.in ../sampleConfig.h @ONLY)
endif()
