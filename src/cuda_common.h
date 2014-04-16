#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

#include <assert.h>

#define WARP_INDEX() (threadIdx.x)
#define BLOCK_INDEX() (blockDim.x * blockIdx.x + WARP_INDEX())

#define ASSERT(x) assert(x)

#endif
