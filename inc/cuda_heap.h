#ifndef _HEAP_H_
#define _HEAP_H_

int init_heap(size_t size);

__device__ void* dev_malloc_node();

#endif
