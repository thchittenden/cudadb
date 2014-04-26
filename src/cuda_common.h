#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

#include <assert.h>
#include <stdio.h>

#define WARP_SIZE() 32
#define LANE_INDEX() (threadIdx.x % WARP_SIZE())
#define WARP_INDEX() (threadIdx.x / WARP_SIZE())

#define BLOCK_SIZE() (blockDim.x)
#define BLOCK_INDEX() (blockIdx.x)

#define GRID_SIZE() (gridDim.x)
#define GRID_INDEX() (gridIdx.x)

#if defined(__DEBUG__) || defined(__ASSERTS__) 
	#define ASSERT(x) assert(x)
	#define DEBUG(lane, expr) if(LANE_INDEX() == lane) { expr; }
#else
	#define ASSERT(x)
	#define DEBUG(x, y)
#endif

#define SUCCESS cudaSuccess
#define ERROR cudaErrorUnknown

#endif
