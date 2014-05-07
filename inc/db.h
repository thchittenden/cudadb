#ifndef _DB_H_
#define _DB_H_

#include "common.h"
#include "db_index.h"
#include "db_types.h"
#include "db_util.h"

#include <cuda.h>
#include <utility>

void db_init(size_t size) {
	// initialize kernel heap size
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);

	// favor the cache, we don't use much shared memory
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	HANDLE_ERROR(cudaGetLastError(), exit(1));
}

template <typename T, typename... A>
table<T> *db_table_create(A T::*... idxs) {
	static_assert(sizeof...(A) > 0, "must declare a primary index");
	size_t offsets[] = {offset_member(idxs)...};
	size_t sizes[]   = {sizeof(A)...};

	// create table and indices
	table<T> *t = new table<T>;
	ASSERT(t != NULL);

	// create stream
	cudaStreamCreate(&t->table_stream);

	// create offset map
	for(int i = 0; i < sizeof...(A); i++) {
		DEBUGP("table %p idx %d offset/size: %lu/%lu\n", t, i, offsets[i], sizes[i]);
		table_index_info x;
		x.idx = i;
		x.key_offset = offsets[i];
		x.key_size = sizes[i];
		t->offset_to_index_info.insert(std::make_pair(offsets[i], x));
	}

	// create indexes
	int res = db_indexes_create(t, (size_t *)&offsets, (size_t *)&sizes);
	if(res < 0) {
		delete t;
		return NULL;
	}

	return t;
}

template <typename T>
void db_table_destroy(table<T> *t) {
	ASSERT(t != NULL);
	db_indexes_destroy<T>(t);
	cudaStreamDestroy(t->table_stream);
	delete t;
}


#endif
