#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include "cuda_common.h"

enum order {
	ORD_NA,
	ORD_LT,
	ORD_EQ,
	ORD_GT
};

/**
 *	Returns the index of the first lane in the warp where cond is true. 32 if all lanes are false. 
 */
static __device__ int __first_lane_true_idx(bool cond) {
	return (__clz(__brev(__ballot(cond))));
}

/**
 *	Returns true for the first lane in the warp where cond is true. False elswehere.
 */
static __device__ bool __first_lane_true(bool cond) {
	return __first_lane_true_idx(cond) == LANE_INDEX();
}

/**
 *	Returns the index of the last lane in the warp where cond is true. -1 if all lanes are true.
 */	
static __device__ int __last_lane_true_idx(bool cond) {
	return 31 - __clz(__ballot(cond));
}

static __device__ bool __lanes_active() {
	return __popc(__ballot(true));
}

static __device__ int __warp_vote(bool cond) {
	return __popc(__ballot(cond));
}

static __device__ int __broadcast(int var, int src) {
#if __CUDA_ARCH__ >= 300
	// use shfl intrinsic
	return __shfl(var, src);
#else
	// use shared memory method
	__shared__ int tmp;
	if(LANE_INDEX() == src)
		tmp = var;

	__threadfence_block();

	return tmp;
#endif
}

static __device__ void* __broadcast_ptr(void* ptr, int src) {
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
		__shared__ void* tmp;
		if(LANE_INDEX() == src) 
			tmp = ptr;

		__threadfence_block();
		
		return tmp;

	#endif
}

static __device__ int __warp_exc_sum(int input) {
	__shared__ int scratch[WARP_SIZE()];
	int idx = LANE_INDEX();
	scratch[idx] = input;
	if(idx >=  1) scratch[idx] += scratch[idx -  1];
	if(idx >=  2) scratch[idx] += scratch[idx -  2];
	if(idx >=  4) scratch[idx] += scratch[idx -  4];
	if(idx >=  8) scratch[idx] += scratch[idx -  8];
	if(idx >= 16) scratch[idx] += scratch[idx - 16];
	scratch[idx] = max(0, scratch[idx] - 1);
	return scratch[idx];
}

/**
 *	Compares _x1[offset:offset+size] with _x2[offset:offset+size]
 */
template <typename T>
static __device__ order memcmp(T* _x1, T* _x2, size_t offset, size_t size) {
	if(_x1 == NULL || _x2 == NULL)
		return ORD_NA;
	char *x1 = (char*)_x1, *x2 = (char*)_x2;
	for(int i = offset; i < offset + size; i++) {
		if (x1[i] < x2[i]) return ORD_LT;
		if (x1[i] > x2[i]) return ORD_GT;
	}
	return ORD_EQ;
}

#endif
