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
}

template <typename T, typename... A>
table<T> *db_table_create(A T::*... idxs) {
	static_assert(sizeof...(A) > 0, "must declare a primary index");
	size_t offsets[] = {offset_member(idxs)...};
	size_t sizes[]   = {sizeof(A)...};

	// create table and indices
	table<T> *t = new table<T>;
	ASSERT(t != NULL);
	for(int i = 0; i < sizeof...(A); i++) {
		DEBUGP("table %p idx %d offset/size: %lu/%lu\n", t, i, offsets[i], sizes[i]);
		table_index<T>* idx = db_index_create<T>(offsets[i], sizes[i]);
		t->offset_to_index.insert(std::make_pair(offsets[i], idx));
	}
	return t;
}

template <typename T>
void db_table_destroy(table<T> *t) {
	ASSERT(t != NULL);
	auto idx_it = t->offset_to_index.begin();
	while(idx_it != t->offset_to_index.end()) {
		db_index_destroy<T>(std::get<1>(*idx_it));
		idx_it++;
	}
	delete t;
}


#endif
