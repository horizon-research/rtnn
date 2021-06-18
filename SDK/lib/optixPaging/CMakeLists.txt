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

set( OPTIX_PAGING ${OptiX_INCLUDE}/optixPaging )

include_directories( ${OptiX_INCLUDE} )

cuda_wrap_srcs( optixPaging OBJ generated_files ${OPTIX_PAGING}/optixPaging.cu "" OPTIONS "" )

add_library( optixPaging STATIC
  ${OPTIX_PAGING}/optixPaging.h
  ${OPTIX_PAGING}/optixPagingImpl.cpp
  ${OPTIX_PAGING}/optixPaging.cu
  ${generated_files}
)

target_include_directories( optixPaging PUBLIC ${OptiX_INCLUDE} )

# These CUDA files need to be compiled as OBJ, not as PTX
set_source_files_properties( ${OPTIX_PAGING}/optixPaging.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

# This file is just for reference in the IDE, don't compile it.
set_source_files_properties( ${OPTIX_PAGING}/optixPagingImpl.cpp PROPERTIES HEADER_FILE_ONLY TRUE )

set_property(TARGET optixPaging PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
