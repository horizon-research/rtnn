// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis
#pragma once

#include <cuda_runtime.h>
#include <string>

namespace bvh_radSearch
{
    class cudaTimer final
    {
        public:

            explicit cudaTimer(void) noexcept
            {
                cudaEventCreate(&this->mStart);
                cudaEventCreate(&this->mEnd);
            }

            ~cudaTimer(void)
            {
                if (this->mStart)
                {
                    cudaEventDestroy(this->mStart);
                    cudaEventDestroy(this->mEnd);
                }
            }

            void start(void) noexcept
            {
                cudaEventRecord(this->mStart);
            }

            float stop(void) noexcept
            {
                cudaEventRecord(this->mEnd);
                cudaEventSynchronize(this->mEnd);

                float ms = 0.f;
                cudaEventElapsedTime(&ms, this->mStart, this->mEnd);
                return ms;
            }

        protected:

            // Empty

        private:

            cudaEvent_t mStart;
            cudaEvent_t mEnd;
    };
}
