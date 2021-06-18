//
// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <optix.h>
#include <optix_stubs.h>

#include "optixHair.h"

#include <sutil/Trackball.h>

#include <string>


// forward declarations
struct GLFWwindow;

class Renderer
{
  public:
    Renderer( HairState* pState );

    Camera defaultCamera() const;

  protected:
    void             render() const;
    HairState* const m_pState;
};

class FileRenderer : public Renderer
{
  public:
    FileRenderer( HairState* pState );

    void renderFile( const std::string& fileName ) const;
};

class WindowRenderer : public Renderer
{
  public:
    WindowRenderer( HairState* pState );

    ~WindowRenderer();

    void run() const;

  protected:
    //
    // GLFW callbacks
    //
    static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods );
    static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos );
    static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y );
    static void windowIconifyCallback( GLFWwindow* window, int32_t iconified );
    static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ );
    static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll );

  private:
    static WindowRenderer* GetRenderer( GLFWwindow* window );
    GLFWwindow*            m_window        = nullptr;
    sutil::Trackball       m_trackball     = {};
    int32_t                m_mouseButton   = -1;
    bool                   m_minimized     = false;
};
