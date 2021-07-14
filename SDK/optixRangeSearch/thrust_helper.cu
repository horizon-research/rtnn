#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

// this can't be in the main cpp file since the file containing cuda kernels to
// be compiled by nvcc needs to have .cu extensions. See here:
// https://github.com/NVIDIA/thrust/issues/614

void sortByKey( thrust::device_ptr<float> d_key_ptr, thrust::device_ptr<unsigned int> d_val_ptr, unsigned int N, cudaStream_t stream ) {
  thrust::sort_by_key(thrust::cuda::par.on(stream), d_key_ptr, d_key_ptr + N, d_val_ptr);
}

void sortByKey( thrust::device_ptr<float> d_key_ptr, thrust::device_ptr<unsigned int> d_val_ptr, unsigned int N ) {
  thrust::sort_by_key(d_key_ptr, d_key_ptr + N, d_val_ptr);
}

void sortByKey( thrust::device_ptr<unsigned int> d_vec_key_ptr, thrust::device_ptr<unsigned int> d_vec_val_ptr, unsigned int N, cudaStream_t stream ) {
  thrust::sort_by_key(thrust::cuda::par.on(stream), d_vec_key_ptr, d_vec_key_ptr + N, d_vec_val_ptr);
}

void sortByKey( thrust::device_ptr<unsigned int> d_vec_key_ptr, thrust::device_ptr<unsigned int> d_vec_val_ptr, unsigned int N ) {
  thrust::sort_by_key(d_vec_key_ptr, d_vec_key_ptr + N, d_vec_val_ptr);
}

void sortByKey( thrust::device_vector<float>* d_key, thrust::device_ptr<float3> d_val_ptr ) {
  thrust::sort_by_key(d_key->begin(), d_key->end(), d_val_ptr);
}

void sortByKey( thrust::device_ptr<float> d_key_ptr, thrust::device_ptr<float3> d_val_ptr, unsigned int N ) {
  thrust::sort_by_key(d_key_ptr, d_key_ptr + N, d_val_ptr);
}

void sortByKey( thrust::device_ptr<unsigned int> d_key_ptr, thrust::device_ptr<float3> d_val_ptr, unsigned int N ) {
  thrust::sort_by_key(d_key_ptr, d_key_ptr + N, d_val_ptr);
}

void sortByKey( thrust::device_ptr<unsigned int> d_key_ptr, thrust::device_ptr<char> d_val_ptr, unsigned int N ) {
  thrust::sort_by_key(d_key_ptr, d_key_ptr + N, d_val_ptr);
}

void gatherByKey ( thrust::device_vector<unsigned int>* d_vec_val, thrust::device_ptr<float3> d_orig_val_ptr, thrust::device_ptr<float3> d_new_val_ptr ) {
  thrust::gather(d_vec_val->begin(), d_vec_val->end(), d_orig_val_ptr, d_new_val_ptr);
}

void gatherByKey ( thrust::device_ptr<unsigned int> d_key_ptr, thrust::device_ptr<float3> d_orig_val_ptr, thrust::device_ptr<float3> d_new_val_ptr, unsigned int N, cudaStream_t stream ) {
  thrust::gather(thrust::cuda::par.on(stream), d_key_ptr, d_key_ptr + N, d_orig_val_ptr, d_new_val_ptr);
}

void gatherByKey ( thrust::device_ptr<unsigned int> d_key_ptr, thrust::device_ptr<float3> d_orig_val_ptr, thrust::device_ptr<float3> d_new_val_ptr, unsigned int N ) {
  thrust::gather(d_key_ptr, d_key_ptr + N, d_orig_val_ptr, d_new_val_ptr);
}

void gatherByKey ( thrust::device_ptr<unsigned int> d_key_ptr, thrust::device_vector<float>* d_orig_queries, thrust::device_ptr<float> d_new_val_ptr, unsigned int N, cudaStream_t stream ) {
  thrust::gather(thrust::cuda::par.on(stream), d_key_ptr, d_key_ptr + N, d_orig_queries->begin(), d_new_val_ptr);
}

void gatherByKey ( thrust::device_ptr<unsigned int> d_key_ptr, thrust::device_vector<float>* d_orig_queries, thrust::device_ptr<float> d_new_val_ptr, unsigned int N ) {
  thrust::gather(d_key_ptr, d_key_ptr + N, d_orig_queries->begin(), d_new_val_ptr);
}

void gatherByKey ( thrust::device_ptr<unsigned int> d_key_ptr, thrust::device_ptr<float> d_orig_val_ptr, thrust::device_ptr<float> d_new_val_ptr, unsigned int N ) {
  thrust::gather(d_key_ptr, d_key_ptr + N, d_orig_val_ptr, d_new_val_ptr);
}

void genSeqDevice(thrust::device_ptr<unsigned int> d_init_val_ptr, unsigned int numPrims) {
  thrust::sequence(d_init_val_ptr, d_init_val_ptr + numPrims);
}

void genSeqDevice(thrust::device_ptr<unsigned int> d_init_val_ptr, unsigned int numPrims, cudaStream_t stream) {
  thrust::sequence(thrust::cuda::par.on(stream),
       d_init_val_ptr, d_init_val_ptr + numPrims);
}

// https://forums.developer.nvidia.com/t/thrust-and-streams/53199
void exclusiveScan(thrust::device_ptr<unsigned int> d_src_ptr, unsigned int N, thrust::device_ptr<unsigned int> d_dest_ptr, cudaStream_t stream) {
  thrust::exclusive_scan(thrust::cuda::par.on(stream),
    d_src_ptr,
    d_src_ptr + N,
    d_dest_ptr);
}

void exclusiveScan(thrust::device_ptr<unsigned int> d_src_ptr, unsigned int N, thrust::device_ptr<unsigned int> d_dest_ptr) {
  thrust::exclusive_scan(
    d_src_ptr,
    d_src_ptr + N,
    d_dest_ptr);
}

void fillByValue(thrust::device_ptr<unsigned int> d_src_ptr, unsigned int N, int value, cudaStream_t stream) {
  thrust::fill(thrust::cuda::par.on(stream), d_src_ptr, d_src_ptr + N, value);
}

void fillByValue(thrust::device_ptr<unsigned int> d_src_ptr, unsigned int N, int value) {
  thrust::fill(d_src_ptr, d_src_ptr + N, value);
}

struct is_true
{
  __host__ __device__
    bool operator()(const bool x)
    {
      return x;
    }
};

struct is_false
{
  __host__ __device__
    bool operator()(const bool x)
    {
      return !x;
    }
};

// https://forums.developer.nvidia.com/t/using-thrust-copy-if-with-a-parameter/119735/6
// https://www.bu.edu/pasi/files/2011/01/NathanBell3-12-1000.pdf#page=18
struct isSameID
{
    char kID;
    isSameID(char id) {kID = id;}

  __host__ __device__
    bool operator()(const char x)
    {
        return (x == kID);
    }

};

void copyIfIdMatch(float3* source, unsigned int N, thrust::device_ptr<char> mask, thrust::device_ptr<float3> dest, char id) {
    thrust::copy_if(thrust::device_pointer_cast(source),
                    thrust::device_pointer_cast(source) + N,
                    mask, dest, isSameID(id));
}

void copyIfStencil(float3* source, unsigned int N, thrust::device_ptr<bool> mask, thrust::device_ptr<float3> dest, bool cond) {
  if (cond)
    thrust::copy_if(thrust::device_pointer_cast(source),
                    thrust::device_pointer_cast(source) + N,
                    mask, dest, is_true());
  else
    thrust::copy_if(thrust::device_pointer_cast(source),
                    thrust::device_pointer_cast(source) + N,
                    mask, dest, is_false());
}

unsigned int countById(thrust::device_ptr<char> val, unsigned int N, char id) {
  unsigned int numOfActiveQueries = thrust::count(val, val + N, id);
  return numOfActiveQueries;
}
