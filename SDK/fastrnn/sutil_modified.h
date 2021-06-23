// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis
// The following code has been retrieved from the NVIDIA Optix SDK and some parts have been
// stripped/modified for the purposes of this work

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

#pragma once

#include <string>
#include <fstream>
#include <map>

#include "Exception_modified.h"

static bool fileExists(const char* path)
{
    std::ifstream str(path);
    return static_cast<bool>(str);
}

static bool fileExists(const std::string& path)
{
    return fileExists(path.c_str());
}

static bool readSourceFile(std::string& str, const std::string& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str());
    if (file.good())
    {
        // Found usable source file
        std::stringstream source_buffer;
        source_buffer << file.rdbuf();
        str = source_buffer.str();
        return true;
    }
    return false;
}

void getPtxString(std::string & ptx, const char* fileDir, const char* filename)
{
    std::string file = std::string(fileDir) + std::string(filename);

    if (!readSourceFile(ptx, file))
    {
        std::string err = "Couldn't open source file " + file;
        throw std::runtime_error(err.c_str());
    }
}
