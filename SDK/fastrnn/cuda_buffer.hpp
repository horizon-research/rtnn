// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <type_traits>
#include <vector>

#include "Exception_modified.h"

namespace bvh_radSearch
{
    template<typename T = CUdeviceptr>
    class cudaBuffer final
    {
        public:

            using _Type = T;

            explicit cudaBuffer(void) noexcept :
                mArraySize(0),
                mByteSize(0),
                mHandle(0)
            { /* Empty */ }

            ~cudaBuffer(void) { this->destroy(); }

            size_t mArraySize;
            size_t mByteSize;

            void alloc(const size_t pSize = 1, int32_t pDefValue = 0)
            {
                if (!pSize) return;

                this->destroy();

                this->mArraySize = pSize;
                this->mByteSize = pSize * sizeof(T);

                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
                    &this->mHandle), this->mByteSize));

                this->memset(pDefValue);
            }

            void allocRaw(const size_t pSize = 1, int32_t pDefValue = 0)
            {
                this->destroy();

                this->mArraySize = pSize;
                this->mByteSize = pSize;

                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
                    &this->mHandle), this->mByteSize));

                this->memset(pDefValue);
            }

            void reset(void) const
            {
                CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(
                    this->mHandle), 0, this->mByteSize));
            }

            void destroy(void)
            {
                if (this->mHandle)
                {
                    this->mArraySize = 0;
                    this->mByteSize = 0;

                    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->mHandle)));

                    this->mHandle = 0;
                }
            }

            template<typename V>
            void copyHostToDevice(
                const V* pData,
                const ptrdiff_t pSize = 0,
                const ptrdiff_t pOffset = 0) const
            {
                CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void*>(this->mHandle + pOffset),
                    reinterpret_cast<const void*>(pData),
                    (pSize ? pSize * sizeof(T) : this->mByteSize),
                    cudaMemcpyHostToDevice));
            }

            template<typename V>
            void copyDeviceToHost(
                V* pData,
                const ptrdiff_t pSize = 0,
                const ptrdiff_t pOffset = 0) const
            {
                CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void*>(pData),
                    reinterpret_cast<const void*>(this->mHandle + pOffset),
                    (pSize ? pSize * sizeof(T) : this->mByteSize),
                    cudaMemcpyDeviceToHost));
            }

            void memset(int32_t pValue = 0, size_t pCount = 0) const
            {
                CUDA_CHECK(cudaMemset(
                    reinterpret_cast<void*>(this->mHandle),
                    pValue,
                    pCount ? pCount : this->mByteSize));
            }

            std::vector<T> getData(void)
            {
                std::vector<T> data(this->mArraySize);
                this->copyDeviceToHost(data.data());
                return data;
            }

            template<typename V>
            inline V* getPtr(void) noexcept { return reinterpret_cast<V*>(this->mHandle); }

            inline T* getPtr(void) noexcept { return reinterpret_cast<T*>(this->mHandle); }

            inline CUdeviceptr& getRawPtr(void) noexcept { return this->mHandle; }

        protected:

            // Empty

        private:

            CUdeviceptr mHandle;
    };
}
