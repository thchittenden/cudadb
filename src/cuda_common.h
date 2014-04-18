#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

#include <assert.h>
#include <stdio.h>

#define WARP_INDEX() (threadIdx.x)
#define BLOCK_INDEX() (blockDim.x * blockIdx.x + WARP_INDEX())

#define HANDLE_ERROR(expr, handler) do { \
		cudaError_t _call_res = expr; \
		if(_call_res != cudaSuccess) { \
			printf("CUDA_ERROR %s(%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(_call_res)); \
			handler; \
		} \
	} while(0);

#if defined(__DEBUG__) || defined(__ALWAYS_DEBUG__) 
	#define ASSERT(x) assert(x)
	#define DEBUG(lane, expr) if(WARP_INDEX() == lane) { expr; }
#else
	#define ASSERT(x)
	#define DEBUG(x, y)
#endif

#endif
