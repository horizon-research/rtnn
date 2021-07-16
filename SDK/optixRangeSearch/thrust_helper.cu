#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>

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

struct is_nonzero
{
  __host__ __device__
    bool operator()(const bool x)
    {
      return x;
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

struct isInRange
{
    char kmin, kmax;
    isInRange(char min, char max) {kmin = min; kmax = max;}

  __host__ __device__
    bool operator()(const char x)
    {
        return ((x >= kmin) && (x <= kmax));
    }
};

void copyIfIdMatch(float3* source, unsigned int N, thrust::device_ptr<char> mask, thrust::device_ptr<float3> dest, char id) {
    thrust::copy_if(thrust::device_pointer_cast(source),
                    thrust::device_pointer_cast(source) + N,
                    mask, dest, isSameID(id));
}

void copyIfIdInRange(float3* source, unsigned int N, thrust::device_ptr<char> mask, thrust::device_ptr<float3> dest, char min, char max) {
    thrust::copy_if(thrust::device_pointer_cast(source),
                    thrust::device_pointer_cast(source) + N,
                    mask, dest, isInRange(min, max));
}

void copyIfNonZero(float3* source, unsigned int N, thrust::device_ptr<bool> mask, thrust::device_ptr<float3> dest) {
    thrust::copy_if(thrust::device_pointer_cast(source),
                    thrust::device_pointer_cast(source) + N,
                    mask, dest, is_nonzero());
}

unsigned int countById(thrust::device_ptr<char> val, unsigned int N, char id) {
  unsigned int numOfActiveQueries = thrust::count(val, val + N, id);
  return numOfActiveQueries;
}

unsigned int uniqueByKey(thrust::device_ptr<unsigned int> key, unsigned int N, thrust::device_ptr<unsigned int> dest) {
  // thrust unique_by_key returns "a pair of iterators at end of the ranges [key_first, keys_new_last) and [values_first, values_new_last)."
  // https://stackoverflow.com/questions/54532859/count-number-of-unique-elements-using-thrust-unique-by-key-when-the-set-of-value
  auto end = thrust::unique_by_key(key, key + N, dest);
  return thrust::get<0>(end) - key;
}

void thrustCopyD2D(thrust::device_ptr<unsigned int> d_dst, thrust::device_ptr<unsigned int> d_src, unsigned int N) {
    cudaMemcpy(
                reinterpret_cast<void*>( thrust::raw_pointer_cast(d_dst) ),
                thrust::raw_pointer_cast(d_src),
                N * sizeof( unsigned int ),
                cudaMemcpyDeviceToDevice
    );
}

// https://github.com/NVIDIA/thrust/blob/master/examples/histogram.cu
unsigned int thrustGenHist(const thrust::device_ptr<char> d_value_ptr, thrust::device_vector<unsigned int>& d_histogram, unsigned int N) {
    // first make a copy of d_value since we are going to sort it.
    thrust::device_vector<char> d_value(N);
    thrust::copy(d_value_ptr, d_value_ptr + N, d_value.begin());

    thrust::sort(d_value.begin(), d_value.end());
    unsigned int num_bins = d_value.back() + 1;

    d_histogram.resize(num_bins);

    thrust::counting_iterator<char> search_begin(0);
    thrust::upper_bound(d_value.begin(), d_value.end(),
            search_begin, search_begin + num_bins,
            d_histogram.begin());

    thrust::adjacent_difference(d_histogram.begin(), d_histogram.end(),
            d_histogram.begin());

    return num_bins;
}
