#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include "cuda_common.h"


/**
 *	Returns the index of the first lane in the warp where cond is true. 32 if all lanes are false. 
 */
__device__ int __first_lane_true_idx(bool cond) {
	return (__clz(__brev(__ballot(cond))));
}

/**
 *	Returns true for the first lane in the warp where cond is true. False elswehere.
 */
__device__ bool __first_lane_true(bool cond) {
	return __first_lane_true_idx(cond) == LANE_INDEX();
}

__device__ bool __lanes_active() {
	return __popc(__ballot(true));
}

__device__ int __broadcast(int var, int src) {
#if __CUDA_ARCH__ >= 300
	// use shfl intrinsic
	return __shfl(var, src);
#else
	// use shared memory method
	__shared__ int tmp[32];
	if(LANE_INDEX() == src)
		tmp[WARP_INDEX()] = var;

	__threadfence_block();

	return tmp[WARP_INDEX()];
#endif
}

__device__ void* __broadcast_ptr(void* ptr, int src) {
	#if __CUDA_ARCH__ >= 300

		union ptr_t {
			void *ptr;
			int   words[2];
		};

		ptr_t p;
		p.ptr = ptr;

		p.words[0] = __shfl(p.words[0], src);
		p.words[1] = __shfl(p.words[1], src);

		return p.ptr;

	#else
		__shared__ void* tmp[32];
		if(LANE_INDEX() == src) 
			tmp[WARP_INDEX()] = ptr;

		__threadfence_block();
		
		return tmp[WARP_INDEX()];

	#endif
}

#endif
