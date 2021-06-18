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
#include "Renderers.h"

#include "Hair.h"

#include <GLFW/glfw3.h>
#include <sutil/GLDisplay.h>

//
// Renderer base class
//
Renderer::Renderer( HairState* pState )
    : m_pState( pState )
{}

void Renderer::render() const
{
    renderFrame( m_pState );
}

//
// FileRenderer
//
FileRenderer::FileRenderer( HairState* pState )
    : Renderer( pState )
{
}

void FileRenderer::renderFile( const std::string& fileName ) const
{
    render();

    // save result image
    sutil::ImageBuffer buffer;
    buffer.data         = m_pState->outputBuffer.getHostPointer();
    buffer.width        = m_pState->width;
    buffer.height       = m_pState->height;
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
    sutil::saveImage( fileName.c_str(), buffer, false );
}

//void FileRenderer::update( HairState::Event event ){};

//
// WindowRenderer
//
WindowRenderer::WindowRenderer( HairState* pState )
    : Renderer( pState )
{
    // Initialize the trackball
    m_trackball.setCamera( &pState->camera );
    m_trackball.setMoveSpeed( 10.0f );
    m_trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    m_trackball.setGimbalLock(true);

    m_window = sutil::initUI( "optixHair", pState->width, pState->height );
    glfwSetMouseButtonCallback( m_window, mouseButtonCallback );
    glfwSetCursorPosCallback( m_window, cursorPosCallback );
    glfwSetWindowSizeCallback( m_window, windowSizeCallback );
    glfwSetWindowIconifyCallback( m_window, windowIconifyCallback );
    glfwSetKeyCallback( m_window, keyCallback );
    glfwSetScrollCallback( m_window, scrollCallback );
    glfwSetWindowUserPointer( m_window, this );
}

WindowRenderer::~WindowRenderer()
{
    m_pState->outputBuffer.deletePBO();
    sutil::cleanupUI( m_window );
}

void WindowRenderer::run() const
{
    sutil::GLDisplay gl_display;

    std::chrono::duration<double> state_update_time( 0.0 );
    std::chrono::duration<double> render_time( 0.0 );
    std::chrono::duration<double> display_time( 0.0 );


    do
    {
        auto t0 = std::chrono::steady_clock::now();
        glfwPollEvents();
        updateParams( m_pState );
        auto t1 = std::chrono::steady_clock::now();
        state_update_time += t1 - t0;
        t0 = t1;

        render();
        t1 = std::chrono::steady_clock::now();
        render_time += t1 - t0;
        t0 = t1;

        // Display
        int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
        int framebuf_res_y = 0;  //
        glfwGetFramebufferSize( m_window, &framebuf_res_x, &framebuf_res_y );
        gl_display.display( m_pState->width, m_pState->height, framebuf_res_x, framebuf_res_y,
                            m_pState->outputBuffer.getPBO() );
        t1 = std::chrono::steady_clock::now();
        display_time += t1 - t0;

        sutil::displayStats( state_update_time, render_time, display_time );

        glfwSwapBuffers( m_window );
    } while( !glfwWindowShouldClose( m_window ) );
    CUDA_SYNC_CHECK();
}

//
// StaticMethods
//
WindowRenderer* WindowRenderer::GetRenderer( GLFWwindow* window )
{
    return static_cast<WindowRenderer*>( glfwGetWindowUserPointer( window ) );
}

void WindowRenderer::mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );
    WindowRenderer* pRenderer = GetRenderer( window );
    HairState*      pState = pRenderer->m_pState;

    if( action == GLFW_PRESS )
    {
        pRenderer->m_mouseButton = button;
        pRenderer->m_trackball.startTracking( static_cast<int>( xpos ), static_cast<int>( ypos ) );

        pState->params.subframe_index = 0u;
    }
    else
    {
        pRenderer->m_mouseButton = -1;
    }
}


void WindowRenderer::cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    WindowRenderer* pRenderer = GetRenderer( window );
    HairState*      pState = pRenderer->m_pState;

    if( pRenderer->m_mouseButton == GLFW_MOUSE_BUTTON_LEFT )
    {
        pRenderer->m_trackball.setViewMode( sutil::Trackball::LookAtFixed );
        pRenderer->m_trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ),
                                              pRenderer->m_pState->width, pRenderer->m_pState->height );
        pState->params.subframe_index = 0u;
    }
    else if( pRenderer->m_mouseButton == GLFW_MOUSE_BUTTON_RIGHT )
    {
        pRenderer->m_trackball.setViewMode( sutil::Trackball::EyeFixed );
        pRenderer->m_trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ),
                                              pRenderer->m_pState->width, pRenderer->m_pState->height );
        pState->params.subframe_index = 0u;
    }
}


void WindowRenderer::windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    WindowRenderer* pRenderer = GetRenderer( window );
    HairState*      pState = pRenderer->m_pState;

    // Keep rendering at the current resolution when the window is minimized.
    if( pRenderer->m_minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    updateSize( pState, res_x, res_y );
    pState->params.subframe_index = 0u;
}


void WindowRenderer::windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    WindowRenderer* pRenderer = GetRenderer( window );
    HairState*      pState = pRenderer->m_pState;
    pRenderer->m_minimized     = ( iconified > 0 );
    pState->params.subframe_index = 0u;
}


void WindowRenderer::keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    WindowRenderer* pRenderer = GetRenderer( window );
    HairState*      pState    = pRenderer->m_pState;
    if( action == GLFW_PRESS )
    {
        switch( key )
        {
            case GLFW_KEY_Q:
            case GLFW_KEY_ESCAPE:
            {
                glfwSetWindowShouldClose( window, true );
            }
            break;
            case GLFW_KEY_1:
            {
                pState->pHair->setSplineMode( Hair::LINEAR_BSPLINE );
                makeHairGAS( pState );
                makeInstanceAccelerationStructure( pState );
                pState->params.handle = pState->hIAS;
                makeProgramGroups( pState );
                makePipeline( pState );
                makeSBT( pState );
                pState->params.subframe_index = 0u;

                std::cout << "Switched to linear b-spline geometry." << std::endl;
            }
            break;
            case GLFW_KEY_2:
            {
                pState->pHair->setSplineMode( Hair::QUADRATIC_BSPLINE );
                makeHairGAS( pState );
                makeInstanceAccelerationStructure( pState );
                pState->params.handle = pState->hIAS;
                makeProgramGroups( pState );
                makePipeline( pState );
                makeSBT( pState );
                pState->params.subframe_index = 0u;

                std::cout << "Switched to quadratic b-spline geometry." << std::endl;
            }
            break;
            case GLFW_KEY_3:
            {
                pState->pHair->setSplineMode( Hair::CUBIC_BSPLINE );
                makeHairGAS( pState );
                makeInstanceAccelerationStructure( pState );
                pState->params.handle = pState->hIAS;
                makeProgramGroups( pState );
                makePipeline( pState );
                makeSBT( pState );
                pState->params.subframe_index = 0u;
                std::cout << "Switched to cubic b-spline geometry." << std::endl;
            }
            break;
            case GLFW_KEY_S:
            {
                pState->pHair->setShadeMode( Hair::SEGMENT_U );
                makeSBT( pState );
                pState->params.subframe_index = 0u;
                std::cout << "Switched to per-segment u shading." << std::endl;
            }
            break;
            case GLFW_KEY_R:
            {
                pState->pHair->setShadeMode( Hair::STRAND_U );
                makeSBT( pState );
                pState->params.subframe_index = 0u;
                std::cout << "Switched to root-to-tip u shading." << std::endl;
            }
            break;
            case GLFW_KEY_I:
            {
                pState->pHair->setShadeMode( Hair::STRAND_IDX );
                makeSBT( pState );
                pState->params.subframe_index = 0u;
                std::cout << "Switched to per-hair color shading." << std::endl;
            }
            break;
            case GLFW_KEY_C:
            {
                pState->pHair->setRadiusMode( Hair::CONSTANT_R );
                makeHairGAS( pState );
                makeInstanceAccelerationStructure( pState );
                pState->params.handle = pState->hIAS;
                makeProgramGroups( pState );
                makePipeline( pState );
                makeSBT( pState );
                pState->params.subframe_index = 0u;
                std::cout << "Switched to constant radius hair geometry." << std::endl;
            }
            break;
            case GLFW_KEY_T:
            {
                pState->pHair->setRadiusMode( Hair::TAPERED_R );
                makeHairGAS( pState );
                makeInstanceAccelerationStructure( pState );
                pState->params.handle = pState->hIAS;
                makeProgramGroups( pState );
                makePipeline( pState );
                makeSBT( pState );
                pState->params.subframe_index = 0u;
                std::cout << "Switched to tapered radius hair geometry." << std::endl;
            }
            break;
        }  // switch
    }      // if "press"
}


void WindowRenderer::scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    WindowRenderer* pRenderer = GetRenderer( window );

    HairState*      pState = pRenderer->m_pState;
    if( pRenderer->m_trackball.wheelEvent( (int)yscroll ) ) {
        pState->params.subframe_index = 0u;
    }
}
