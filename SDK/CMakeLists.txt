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

# Welcome to the OptiX SDK build.  We have chosen CMake, because it can generate multiple
# build systems for multiple architectures from a single script.  There are many resources
# for CMake on-line at http://www.cmake.org and their wiki page,
# http://www.cmake.org/Wiki/CMake, in addition to the documentation that comes with the
# distribution.  There is also a book available if you wish to delve more deeply into
# various topics.

# If you wish to create your own project and use the SDK as a template there are a number
# of things you should do.
#
# 1. You should copy the contents of the SDK to a place of your choice.
#
# 2. You can remove any sample's directory you don't wish to build.  Be careful about
#    the following directories.
#
#    a. CMake - contains helper scripts that make this all work.  You should keep this.
#
#    b. sutil and putil
#              - Almost all of the samples make use of this shared code one way or another, so
#               you should probably keep them until you have your own frameowrk for your
#               code.
#
#    d. data  - This directory contains the cow.obj file used as an example for
#               many of the samples.  You can move cow.obj anywhere as long as
#               you fix all the file paths in the samples you wish to use it in.
#
# 3. You should update the list of sub directories that CMake needs to process below (look
#    for the comment "List of samples found in subdirectories.")
#

# The basic flow of execution of this file is to do the following.
#
# 1. Setup the project and other global settings.  This involves processing some helper
#    scripts.
#
# 2. Look for external dependencies, CUDA, and OptiX.
#
# 3. Process all the subdirectories' CMakeLists.txt files.  These files create all the
#    executable and library targets that are used to build the SDK.
#
# 4. As a convenience on Windows, copy the OptiX dlls into the build directories, so OptiX
#    doesn't have to be in the path to run the samples.
#
# 5. Set a CMake variable that indicates we have configured for the first time.  This
#    allows us to override and set varibles' defaults while allowing them to be modified
#    later.

# If you have any questions, don't feel shy about posting to the OptiX forums:
# https://devtalk.nvidia.com/default/board/90/


# This sets up the name of our project.  For our purposes the main thing this controls is
# the name of the VS solution file.
project(OptiX-Samples)

# This enforces a particular version of CMake that we require to process the script files
# properly.  We rely on CXX_STANDARD, which was added in CMake 3.1.
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# As of CMake 2.6 policies were introduced in order to provide a mechanism for
# adding backwards compatibility one feature at a time.  We will just specify
# that all policies will use version 2.6 symantics.
cmake_policy(VERSION 2.6)

if( POLICY CMP0072 )
  # FindOpenGL prefers GLVND by default when available
  cmake_policy(SET CMP0072 NEW)
endif()

if( POLICY CMP0074 )
  # find_package uses <PackageName>_ROOT variables.
  cmake_policy(SET CMP0074 NEW)
endif()

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/../SDK/CMake"
  ${CMAKE_MODULE_PATH}
  )

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Tells CMake to build all the libraries as shared libraries by default.  This can be
# overrided by individual libraries later.
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

##########
# Process our custom setup scripts here.

# Enable C++11. Needs to be done before the include of ConfigCompilerFlags.cmake below.
set(GCC_LIBSTDCPP11 ON)

# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# For Xcode 5, gcc is actually clang, so we have to tell CUDA to treat the compiler as
# clang, so that it doesn't mistake it for something else.
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

# CUDA 8 is broken for generating dependencies during configure
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)

# Find at least a 5.0 version of CUDA.
find_package(CUDA 5.0 REQUIRED)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)


# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if(NOT PASSED_FIRST_CONFIGURE)
  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch sm_60)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  set(flag "--use_fast_math")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()
  
  set(flag "-lineinfo")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  if (CUDA_VERSION VERSION_LESS "3.0")
    set(flag "--keep")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()

  if( APPLE )
    # Undef'ing __BLOCKS__ for OSX builds.  This is due to a name clash between OSX 10.6
    # C headers and CUDA headers
    set(flag "-U__BLOCKS__")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()

  if(CMAKE_CXX_STANDARD EQUAL 11)
    set(SAMPLES_NVRTC_CXX "-std=c++11")
  else()
    set(SAMPLES_NVRTC_CXX "")
  endif()
  set(CUDA_NVRTC_FLAGS ${SAMPLES_NVRTC_CXX} -arch compute_60 -use_fast_math -lineinfo -default-device -rdc true -D__x86_64 CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
endif(NOT PASSED_FIRST_CONFIGURE)

mark_as_advanced(CUDA_NVRTC_FLAGS)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Create a flag for mac which will allow apps to add the local cuda toolkit
# install path to the app's rpath.
if( APPLE )
  set( CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib" )
endif()

# Locate the NVRT distribution.  Search the SDK first, then look in the system.
set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../" CACHE PATH "Path to OptiX installed location.")

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories(
  "${OptiX_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  )

# Select whether to use NVRTC or NVCC to generate PTX
set(CUDA_NVRTC_ENABLED ON CACHE BOOL "Use NVRTC to compile PTX at run-time instead of NVCC at build-time")

##################################################################
# SUtil compilation

set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

if (WIN32)
  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR ${SAMPLES_PTX_DIR})
else (WIN32)
  if ( USING_GNU_C AND NOT APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846" )
  endif()
endif (WIN32)

set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")

set(SAMPLES_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../SDK/support")

# NVRTC include paths relative to the sample path
set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"cuda\", \\
  \"sutil\", \\
  \".\", ")

# NVRTC absolute include paths to the headers used to build the samples
set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")

# Build a null-terminated option list for NVRTC
set(CUDA_NVRTC_OPTIONS)
foreach(flag ${CUDA_NVRTC_FLAGS})
  set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS} \\\n  \"${flag}\",")
endforeach()
set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS}")

configure_file(sampleConfig.h.in sampleConfig.h @ONLY)

# Path to sutil.h that all the samples need
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     "${CMAKE_BINARY_DIR}/include"
                     ${CMAKE_CURRENT_BINARY_DIR}
                     ${CUDA_INCLUDE_DIRS}
                     )

set(SAMPLES_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

#########################################################
# OPTIX_add_sample_executable
#
# Convience function for adding samples to the code.  You can copy the contents of this
# funtion into your individual project if you wish to customize the behavior.  Note that
# in CMake, functions have their own scope, whereas macros use the scope of the caller.
function(OPTIX_add_sample_executable target_name_base target_name_var)

  set( target_name ${target_name_base} )
  set( ${target_name_var} ${target_name} PARENT_SCOPE )

  # These calls will group PTX and CUDA files into their own directories in the Visual
  # Studio projects.
  if (NOT CUDA_NVRTC_ENABLED)
    source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  endif()
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
  # code here, so that we can use our own name for the target.  target_name is used in the
  # creation of the output file names, and we want this to be unique for each target in
  # the SDK.
  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  if (CUDA_NVRTC_ENABLED)

    # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
    set(cu_obj_source_files)
    foreach(file ${source_files})
      get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
      if(${_cuda_source_format} MATCHES "OBJ")
        list(APPEND cu_obj_source_files ${file})
      endif()
    endforeach()

    # Create the rules to build the OBJ from the CUDA files.
    CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )
  else()

    # Create the rules to build the PTX and OBJ from the CUDA files.
    CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )
  endif()


  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  add_executable(${target_name}
    ${source_files}
    ${generated_files}
    ${cmake_options}
    )

  # Most of the samples link against the sutil library and the optix library.  Here is the
  # rule that specifies this linkage.
  target_link_libraries( ${target_name}
    ${GLFW_LIB_NAME}
    imgui
    sutil_7_sdk
    )

  set_target_properties( ${target_name} PROPERTIES
    COMPILE_DEFINITIONS
    "OPTIX_SAMPLE_NAME_DEFINE=${target_name};OPTIX_SAMPLE_DIR_DEFINE=${target_name}" )

  if( UNIX AND NOT APPLE )
    # Force using RPATH instead of RUNPATH on Debian
    target_link_libraries( ${target_name} "-Wl,--disable-new-dtags" )
  endif()

  if(USING_GNU_CXX)
    target_link_libraries( ${target_name} m ) # Explicitly link against math library (C samples don't do that by default)
  endif()
endfunction()

#########################################################
#  List of samples found in subdirectories.
#
# If you wish to start your own sample, you can copy one of the sample's directories.
# Just make sure you rename all the occurances of the sample's name in the C code as well
# and the CMakeLists.txt file.
add_subdirectory( optixCallablePrograms )
add_subdirectory( optixCurves           )
add_subdirectory( optixCutouts          )
add_subdirectory( optixDemandTexture    )
add_subdirectory( optixDynamicMaterials )
add_subdirectory( optixHair             )
add_subdirectory( optixHello            )
add_subdirectory( optixMeshViewer       )
add_subdirectory( optixMultiGPU         )
add_subdirectory( optixNVLink           )
add_subdirectory( optixPathTracer       )
add_subdirectory( optixRaycasting       )
add_subdirectory( optixSimpleMotionBlur )
add_subdirectory( optixSphere           )
add_subdirectory( optixTriangle         )
add_subdirectory( optixWhitted          )
add_subdirectory( example01_helloOptix  )
add_subdirectory( optixRangeSearch      )

# Our sutil library.  The rules to build it are found in the subdirectory.
add_subdirectory(sutil)
# Our other libraries.
add_subdirectory( lib/optixPaging )
# Third-party support libraries.
add_subdirectory(support)

#################################################################

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")
