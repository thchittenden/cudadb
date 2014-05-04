#ifndef _DB_SELECT_H_
#define _DB_SELECT_H_

#include "common.h"
#include "db_types.h"
#include "db_util.h"

#include <utility>

template <typename T, int N>
void db_dev_select_block(table<T>*, criteria<T, N>*, select_state<T>*, result_buffer<T>*);

/**
 *	Creates a new select statement for type T and member/value pair params. The
 *	returned handle can be used to retrieve the selected results.
 */
template <typename T, typename... A>
select_handle<T, sizeof...(A)> *db_select_prepare(table<T>* t, std::pair<A T::*, A>... params) {
	ASSERT(t != NULL);
	size_t offsets[] = {offset_member(std::get<0>(params))...};

	// generate criteria
	criteria<T, sizeof...(A)> crit;
	APPLY(crit.model.*std::get<0>(params) = std::get<1>(params));
	for(int i = 0; i < sizeof...(A); i++) {
		auto idx_it = t->offset_to_index_info.find(offsets[i]);
		if(idx_it == t->offset_to_index_info.end()) {
			DEBUGP("could not find index in table %p for offset %lu...\n", t, offsets[i]);
			return NULL;
		}
		crit.crit_idxs[i] = std::get<1>(*idx_it).idx; 
	}

	// create select_handle;
	select_handle<T, sizeof...(A)>* handle = new select_handle<T, sizeof...(A)>;
	if(handle == NULL) {
		goto err0;
	}
	handle->t = t;
	handle->buf_cur_idx = 0;

	// print debug info
	for(int i = 0; i < sizeof...(A); i++) 
		DEBUGP("select handle %p idx %d has criteria for offset %lu\n", handle, i, offsets[i]);

	// allocate device memory and transfer criteria
	char *dev_buf;
	HANDLE_ERROR(cudaMalloc(&dev_buf, sizeof(crit) + sizeof(*handle->dev_buf) + sizeof(*handle->dev_state)), goto err1);
	handle->dev_crit  = (criteria<T, sizeof...(A)>*)dev_buf;
	handle->dev_state = (select_state<T>*)(dev_buf + sizeof(*handle->dev_crit));
	handle->dev_buf   = (result_buffer<T>*)(dev_buf + sizeof(*handle->dev_crit) + sizeof(*handle->dev_state));
	HANDLE_ERROR(cudaMemcpyAsync(handle->dev_crit, &crit, sizeof(*handle->dev_crit), cudaMemcpyHostToDevice, t->table_stream), goto err2);
	HANDLE_ERROR(cudaMemsetAsync(handle->dev_state, 0, sizeof(*handle->dev_state), t->table_stream), goto err2);

	// allocate host buffers
	HANDLE_ERROR(cudaHostAlloc(&handle->host_buf_cur, sizeof(*handle->host_buf_cur), cudaHostAllocDefault), goto err2);
	HANDLE_ERROR(cudaHostAlloc(&handle->host_buf_next, sizeof(*handle->host_buf_next), cudaHostAllocDefault), goto err2);
	ASSERT(handle->host_buf_cur != NULL);
	ASSERT(handle->host_buf_next != NULL);
	
	// initialize cuda events
	HANDLE_ERROR(cudaEventCreateWithFlags(&handle->event_cur,  cudaEventDisableTiming), goto err2);
	HANDLE_ERROR(cudaEventCreateWithFlags(&handle->event_next, cudaEventDisableTiming), goto err2);

	// start first two select blocks
	db_dev_select_block(t, handle->dev_crit, handle->dev_state, handle->dev_buf);
	HANDLE_ERROR(cudaMemcpyAsync(handle->host_buf_cur, handle->dev_buf, sizeof(*handle->host_buf_cur), cudaMemcpyDeviceToHost, t->table_stream), goto err2);
	HANDLE_ERROR(cudaEventRecord(handle->event_cur, t->table_stream), goto err2);

	db_dev_select_block(t, handle->dev_crit, handle->dev_state, handle->dev_buf);
	HANDLE_ERROR(cudaMemcpyAsync(handle->host_buf_next, handle->dev_buf, sizeof(*handle->host_buf_next), cudaMemcpyDeviceToHost, t->table_stream), goto err2);
	HANDLE_ERROR(cudaEventRecord(handle->event_next, t->table_stream), goto err2);
	
	// return handle for user
	return handle;

err2:
	cudaFree(dev_buf);	
err1:
	delete handle;
err0:
	return NULL;
}

template <typename T, int N>
T *db_select_next(select_handle<T, N> *handle) {
	ASSERT(handle != NULL);

	// wait for host_buf_cur to be valid
	cudaEventSynchronize(handle->event_cur);
	if(handle->buf_cur_idx == SELECT_CHUNK_SIZE) {
		// we've exhausted current chunk, wait for next chunk and use that
		cudaEventSynchronize(handle->event_next); //TODO sync after new request?

		// swap result buffers
		result_buffer<T>* tmp = handle->host_buf_next;
		handle->host_buf_next = handle->host_buf_cur;
		handle->host_buf_cur  = tmp;
		handle->buf_cur_idx = 0;

		if(handle->host_buf_cur->nvalid == SELECT_CHUNK_SIZE) {
			// there are more results, fetch them
			db_dev_select_block(handle->t, handle->dev_crit, handle->dev_state, handle->dev_buf);
			HANDLE_ERROR(cudaMemcpyAsync(handle->host_buf_next, handle->dev_buf, sizeof(*handle->host_buf_next), cudaMemcpyDeviceToHost, handle->t->table_stream), goto err0);
			HANDLE_ERROR(cudaEventRecord(handle->event_next, handle->t->table_stream), goto err0);
		}
	}

	// either return next result and increment idx or return NULL
	if(handle->buf_cur_idx < handle->host_buf_cur->nvalid) {
		return &handle->host_buf_cur->results[handle->buf_cur_idx++];
	} else {
		return NULL;
	}


err0:
	// user should call db_select_destroy now 
	return NULL;
}

template <typename T, int N>
void db_select_destroy(select_handle<T, N> *handle) {
	ASSERT(handle != NULL);
	cudaFreeHost(handle->host_buf_cur);
	cudaFreeHost(handle->host_buf_next);
	cudaFree(handle->dev_crit); // also frees dev_buf since they were allocated together
	delete handle;
}

#endif
