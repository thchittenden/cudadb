#ifndef _DB_TYPES_H_
#define _DB_TYPES_H_

#include "db_defs.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <map>

// device side struct
template <typename T>
struct btree_node {
	btree_node* children[BTREE_NODE_CHILDREN];
	T elems[BTREE_NODE_KEYS + 1]; // pad for safe access by all warps
};

// device side struct
template <typename T>
struct btree {
	btree_node<T>* root;
};

// device side struct
template <typename T>
struct table_index {
	btree<T*> tree;
	size_t key_offset;
	size_t key_size;
};

// device side struct
template <typename T>
struct select_state {
	btree_node<T*>* node_stack[BTREE_MAX_DEPTH];
	int node_idx[BTREE_MAX_DEPTH];
	int stack_idx;
};

template <typename T, int N>
struct criteria {
	T model;
	int crit_idxs[N];
};

template <typename T>
struct result_buffer {
	int nvalid;
	T results[SELECT_CHUNK_SIZE];
};

struct table_index_info {
	int idx;
	size_t key_offset;
	size_t key_size;
};

template <typename T>
struct table {
	// map from offset to index info
	std::map<size_t, table_index_info> offset_to_index_info;

	// pointer to array of pointer indexes on device
	table_index<T>* dev_indexes;

	// stream associated with table
	cudaStream_t table_stream;

	// number of elements inserted into table
	int size;
};

template <typename T, int N>
struct select_handle {
	
	table<T>* t;
	criteria<T, N>* dev_crit;
	select_state<T>* dev_state;
	result_buffer<T>* host_buf_cur;
	result_buffer<T>* host_buf_next;
	result_buffer<T>* dev_buf;
	cudaEvent_t event_cur;
	cudaEvent_t event_next;

	int buf_cur_idx;
};


#endif 
