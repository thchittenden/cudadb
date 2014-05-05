#ifndef _DB_TYPES_H_
#define _DB_TYPES_H_

#include "db_defs.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <map>

/**
 *	Device side struct of a node of a btree. We pad elems up to a size of 32
 *	so that warps can safely and easily access the memory.
 */
template <typename T>
struct btree_node {
	btree_node* children[BTREE_NODE_CHILDREN];
	T elems[BTREE_NODE_KEYS + 1]; // pad for safe access by all warps
};

/**
 *	Device side struct of a btree.
 */
template <typename T>
struct btree {
	btree_node<T>* root;
};

/**
 *	Devices side struct of a table index.
 */
template <typename T>
struct table_index {
	btree<T*> tree;
	size_t key_offset;
	size_t key_size;
};

/**
 *	Device side struct containing information about progress through current
 *	select statement.
 */
template <typename T>
struct select_state {
	btree_node<T*>* node_stack[BTREE_MAX_DEPTH];
	int node_idx[BTREE_MAX_DEPTH];
	int stack_idx;
};

/**
 *	Enum of possible comparison operators for selects.
 */
enum cmp_op {
	LT,
	LE,
	EQ,
	GE,
	GT,
	NE,
};

/**
 *	Host/device side struct containing information about the criteria to apply
 *	to a given select statement.
 */
template <typename T, int N>
struct criteria {
	T model;
	int crit_idxs[N];
	cmp_op crit_ops[N];
};

/**
 *	Host/device side struct containing the results of the select statement.
 */
template <typename T>
struct result_buffer {
	int nvalid;
	T results[SELECT_CHUNK_SIZE];
};

/**
 *	Host side struct containing information about the table indexes. Idx is
 *	an index into the array of table indices kept on the device.
 */
struct table_index_info {
	int idx;
	size_t key_offset;
	size_t key_size;
};

/**
 *	Host side struct containing information about a table and its associated
 *	stream.
 */
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

/**
 *	Host side struct containing information about an open select statement.
 */
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
