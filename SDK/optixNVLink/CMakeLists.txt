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

set_source_files_properties( ${CMAKE_CURRENT_SOURCE_DIR}/optixNVLink_kernels.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
  )

# Under windows, look for nvml.lib in the cuda toolkit, and the the general cuda path
if( WIN32 )
  find_file( NVML_LIBRARY
      NO_DEFAULT_PATH
      NAMES nvml.lib
      PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 $ENV{CUDA_PATH}/lib/x64
      )
endif()

# Under Unix, look for the libnvidia-ml library
if( UNIX )
  find_library( NVML_LIBRARY libnvidia-ml.so )
endif()

# If an NVML library was found, compile using it
if( NVML_LIBRARY )
  set( OPTIX_USE_NVML ON )
endif()

# Set the value of OPTIX_USE_NVML in nvml_configure.h
configure_file( nvml_configure.h.in nvml_configure.h )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

# Set up source files for executable
OPTIX_add_sample_executable( optixNVLink target_name
  ${SAMPLES_CUDA_DIR}/helpers.h
  ${SAMPLES_CUDA_DIR}/random.h
  optixNVLink_kernels.cu
  optixNVLink.cu
  optixNVLink.cpp
  optixNVLink.h

  nvml_configure.h
  )
set_property( TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON )

# Set up libraries to link in
# Not linking NVML_LIBRARY, since it will be loaded at runtime
target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
