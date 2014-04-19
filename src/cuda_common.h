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

#define HANDLE_ERROR(expr, handler) do { \
		cudaError_t _call_res = expr; \
		if(_call_res != cudaSuccess) { \
			printf("CUDA_ERROR %s(%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(_call_res)); \
			handler; \
		} \
	} while(0);

#if defined(__DEBUG__) || defined(__ALWAYS_DEBUG__) 
	#define ASSERT(x) assert(x)
	#define DEBUG(lane, expr) if(LANE_INDEX() == lane) { expr; }
#else
	#define ASSERT(x)
	#define DEBUG(x, y)
#endif

#endif
