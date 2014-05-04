#ifndef _COMMON_H_
#define _COMMON_H_

#include <assert.h>
#include <stdio.h>

#if defined(__ASSERTS__)
	#define ASSERT(x) assert(x)
#else
	#define ASSERT(x)
#endif

#if defined(__DEBUG__)
	#define DEBUGP(str, ...) printf(str, ## __VA_ARGS__)
#else
	#define DEBUGP(...)
#endif

#define HANDLE_ERROR(expr, handler) do { \
		cudaError_t _call_res = expr; \
		if(_call_res != cudaSuccess) { \
			printf("CUDA_ERROR %s(%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(_call_res)); \
			handler; \
		} \
	} while(0);

#endif
