#ifndef _DB_SELECT_H_
#define _DB_SELECT_H_

#include "common.h"
#include "db_types.h"
#include "db_util.h"

#include <utility>

template <typename T>
void db_dev_select_block(select_handle<T>*);

/**
 *	Creates a new select statement for table T and criteria crit. The
 *	returned handle can be used to retrieve the selected results.
 */
template <typename T>
select_handle<T> *db_select_prepare(table<T>* t, criteria<T> idx_crit, criteria<T> crit) {
	ASSERT(t != NULL);

	// create select_handle;
	select_handle<T>* handle = new select_handle<T>;
	if(handle == NULL) {
		return NULL;
	}
	handle->t = t;
	handle->buf_cur_idx = 0;
	
	// get index offset and lookup index
	// TODO should support combiners
	crit_block* idx_crit_block = (crit_block*)idx_crit.crit;
	ASSERT(idx_crit_block.t != AND_TAG && ix_crit_block.t != OR_TAG);
	auto idx_info = t->offset_to_index_info.find(idx_crit_block->comp.offset);
	ASSERT(idx_info != t->offset_to_index_info.end());
	handle->dev_idx = &t->dev_indexes[idx_info->second.idx];
	
	// set crit string lengths
	handle->dev_idx_crit_str_len = idx_crit.len;
	handle->dev_crit_str_len     = crit.len;

	// allocate device memory
	char *dev_buf;
	HANDLE_ERROR(cudaMalloc(&dev_buf, idx_crit.len + crit.len + sizeof(*handle->dev_buf) + sizeof(*handle->dev_state)), goto err1);
	handle->dev_idx_crit_str = dev_buf;
	handle->dev_crit_str 	 = dev_buf + idx_crit.len;
	handle->dev_state 	 	 = (select_state<T>*)(dev_buf + idx_crit.len + crit.len);
	handle->dev_buf   	 	 = (result_buffer<T>*)(dev_buf + idx_crit.len + crit.len + sizeof(*handle->dev_state));
	
	// transfer criteria
	HANDLE_ERROR(cudaMemcpyAsync(handle->dev_idx_crit_str,	// destination
				idx_crit.crit,							// source
				idx_crit.len,			 				// size
				cudaMemcpyHostToDevice, 				// direction
				t->table_stream), goto err2);			// stream
	HANDLE_ERROR(cudaMemcpyAsync(handle->dev_crit_str,	// destination
				crit.crit,	 							// source
				crit.len,				 				// size
				cudaMemcpyHostToDevice, 				// direction
				t->table_stream), goto err2);			// stream

	// zero out select state
	HANDLE_ERROR(cudaMemsetAsync(handle->dev_state, 0, sizeof(*handle->dev_state), t->table_stream), goto err2);

	// allocate pinned host buffers
	HANDLE_ERROR(cudaHostAlloc(&handle->host_buf_cur, sizeof(*handle->host_buf_cur), cudaHostAllocDefault), goto err2);
	HANDLE_ERROR(cudaHostAlloc(&handle->host_buf_next, sizeof(*handle->host_buf_next), cudaHostAllocDefault), goto err3);
	ASSERT(handle->host_buf_cur != NULL);
	ASSERT(handle->host_buf_next != NULL);
	
	// initialize cuda events
	HANDLE_ERROR(cudaEventCreateWithFlags(&handle->event_cur,  cudaEventDisableTiming), goto err4);
	HANDLE_ERROR(cudaEventCreateWithFlags(&handle->event_next, cudaEventDisableTiming), goto err4);

	// start first two select blocks
	db_dev_select_block(handle);
	HANDLE_ERROR(cudaMemcpyAsync(handle->host_buf_cur,	// destination
				handle->dev_buf, 						// source
				sizeof(*handle->host_buf_cur), 			// size
				cudaMemcpyDeviceToHost, 				// direction
				t->table_stream), goto err4);			// stream
	HANDLE_ERROR(cudaEventRecord(handle->event_cur, t->table_stream), goto err4);

	db_dev_select_block(handle);
	HANDLE_ERROR(cudaMemcpyAsync(handle->host_buf_next,	// destination
				handle->dev_buf, 						// source
				sizeof(*handle->host_buf_next), 		// size
				cudaMemcpyDeviceToHost, 				// direction
				t->table_stream), goto err4);			// stream
	HANDLE_ERROR(cudaEventRecord(handle->event_next, t->table_stream), goto err4);
	
	// return handle for user
	return handle;
	
	// error returns
err4:
	cudaFreeHost(handle->host_buf_next);
err3:
	cudaFreeHost(handle->host_buf_cur);
err2:
	cudaFree(dev_buf);	
err1:
	delete handle;
	return NULL;
}

template <typename T>
T *db_select_next(select_handle<T> *handle) {
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
			db_dev_select_block(handle);
			HANDLE_ERROR(cudaMemcpyAsync(handle->host_buf_next,	// destination
						handle->dev_buf, 						// source
						sizeof(*handle->host_buf_next), 		// size
						cudaMemcpyDeviceToHost, 				// direction
						handle->t->table_stream), goto err0);	// stream
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

template <typename T>
void db_select_destroy(select_handle<T> *handle) {
	ASSERT(handle != NULL);
	cudaFreeHost(handle->host_buf_cur);
	cudaFreeHost(handle->host_buf_next);
	cudaFree(handle->dev_idx_crit_str); // also frees dev_buf since they were allocated together
	delete handle;
}

#endif
