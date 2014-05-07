#ifndef _DB_TYPES_H_
#define _DB_TYPES_H_

#include "db_defs.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <map>

#if defined(__CUDACC__) // NVCC
	#define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
	#define ALIGN(n) __attribute__((aligned(n)))
#else 
	#define ALIGN(n) 
#endif

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
enum crit_tag {
	LT_TAG,
	LE_TAG,
	EQ_TAG,
	GE_TAG,
	GT_TAG,

	AND_TAG,
	OR_TAG
};

/**
 *	Host side struct containing the criteria string.
 */
template <typename T>
struct criteria {
	size_t len;
	char*  crit;
};

/**
 *	Host/device side struct of a sub-block of the criteria string.
 */
struct crit_compare_block {
	size_t size;
	size_t offset;
	char val[];
} ALIGN(8);

/**
 *	Host/device side struct of a sub-block of the criteria string.
 */
struct crit_combine_block {
	size_t size;
	char sub_blocks[];
} ALIGN(8);

/**
 *	Host/device side union of block times.
 */
struct crit_block {
	crit_tag tag;
	union {
		crit_compare_block comp;
		crit_combine_block comb;
	};
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
template <typename T>
struct select_handle {

	// pointer to the current select's table
	table<T>* t;

	// the index that the select is using
	table_index<T>* dev_idx;

	// criteria on index fields
	char* dev_idx_crit_str;
	int   dev_idx_crit_str_len;

	// criteria on all fields
	char* dev_crit_str;
	int   dev_crit_str_len;

	// device pointer to the select state
	select_state<T>* dev_state;

	// host pointer to the first result buffer
	result_buffer<T>* host_buf_cur;

	// host pointer to the second result buffer
	result_buffer<T>* host_buf_next;

	// device pointer to the result buffer
	result_buffer<T>* dev_buf;

	// inidicator whether host_buf_cur is full
	cudaEvent_t event_cur;

	// indicator whether host_buf_next is full
	cudaEvent_t event_next;

	// current pointer into host_buf_cur
	int buf_cur_idx;
};


#endif 
