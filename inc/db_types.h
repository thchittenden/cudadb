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
	T* elems[BTREE_NODE_KEYS + 1]; // pad for safe access by all warps
};

// device side struct
template <typename T>
struct btree {
	btree_node<T>* roots[BTREE_NUM_ROOTS];
	size_t key_offset;
	size_t key_size;
};

// device side struct
template <typename T>
struct select_state {
	btree_node<T*>* node_stack[BTREE_MAX_DEPTH];
	int node_idx[BTREE_MAX_DEPTH];
	int stack_idx;
	int root_idx;
};

template <typename T>
struct table_index {
	btree<T*> *dev_btree;
};

template <typename T, int N>
struct criteria {
	T model;
	table_index<T> idx[N];
};

template <typename T>
struct result_buffer {
	int nvalid;
	T results[SELECT_CHUNK_SIZE];
};

template <typename T, int N>
struct select_handle {
	
	criteria<T, N>* dev_crit;
	select_state<T>* dev_state;
	result_buffer<T>* host_buf_cur;
	result_buffer<T>* host_buf_next;
	result_buffer<T>* dev_buf;
	cudaEvent_t event_cur;
	cudaEvent_t event_next;

	int buf_cur_idx;
};

template <typename T>
struct table {
	std::map<size_t, table_index<T>*> offset_to_index;
	int size;
};

#endif 
