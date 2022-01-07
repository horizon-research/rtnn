CUDA_HOME := /usr/local/cuda

all: memstatlib.cpp ; g++ -I${CUDA_HOME}/include -fPIC -shared -o libmemstatlib.so memstatlib.cpp -ldl -L${CUDA_HOME}/lib64 -lcudart
