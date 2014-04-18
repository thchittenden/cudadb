#include "util.h"

void setCudaHeapSize(size_t bytes) {
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, bytes);
}
