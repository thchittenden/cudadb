#include "common.h"
#include "db_types.h"
#include "db_index.h"
#include "db_defs.h"

#include "cuda_common.h"
#include "cuda_util.h"

/**
 *	Initializes a new btree.
 *
 *	BLOCK SIZE: 32x1
 *	GRID  SIZE: 1x1
 */
template <typename T>
static __global__ void dev_indexes_create(int n, table_index<T>* ti) {
	ASSERT(ti != NULL);
	int idx  = LANE_INDEX();

	// initialize root
	btree_node<T*>* my_root;
	if(idx < n) {
		my_root = (btree_node<T*>*)malloc(sizeof(*my_root));
		ASSERT(my_root != NULL);
		ti[idx].tree.root = my_root;
	}

	// zero each root
	btree_node<T*>* cur_root;
	for(int i = 0; i < n; i++) {
		cur_root = (btree_node<T*>*)__broadcast_ptr(my_root, i);
		cur_root->elems[idx] = NULL;
		cur_root->children[idx] = NULL;
	}
}

/**
 *	Creates an index which will reference the data.
 */
template <typename T>
int db_indexes_create(table<T>* t, size_t* key_offsets, size_t* key_sizes) {
	
	// get number of indexes
	int n = t->offset_to_index_info.size();

	// allocate table indexes
	table_index<T>* dev_indexes;
	HANDLE_ERROR(cudaMalloc(&dev_indexes, n*sizeof(*dev_indexes)), goto err0);
	
	// copy key_offsets and key_sizes to device
	HANDLE_ERROR(cudaMemcpy2DAsync(
		&dev_indexes[0].key_offset, // destination
		sizeof(table_index<T>), 	// destination pitch
		key_offsets,				// source
		sizeof(size_t),				// source pitch
		1*sizeof(size_t), 			// width in bytes
		n,							// height
		cudaMemcpyHostToDevice,		// kind
		t->table_stream				// stream
	), goto err1);
	HANDLE_ERROR(cudaMemcpy2DAsync(
		&dev_indexes[0].key_size,	// destination
		sizeof(table_index<T>),		// destination pitch
		key_sizes,					// source
		sizeof(size_t),				// source pitch
		1*sizeof(size_t), 			// width in bytes
		n,							// height 
		cudaMemcpyHostToDevice,		// kind
		t->table_stream				// stream
	), goto err1);

	// initialize table indexes
	dev_indexes_create<<<1, 32, 0, t->table_stream>>>(n, dev_indexes);
	
	// return success
	t->dev_indexes = dev_indexes;
	return 0;

err1:
	cudaFree(dev_indexes);
err0:
	t->dev_indexes = NULL;
	return -1;
}

/**
 *	Destroys a btree and frees all contained nodes.
 *
 *	BLOCK SIZE: 32x1
 *  GRID  SIZE: 15x1
 */
template <typename T>
static __global__ void dev_index_destroy(table_index<T>* ti) {
	int idx = LANE_INDEX();
	int bidx = BLOCK_INDEX();

	__shared__ btree_node<T*>* node_stack[BTREE_MAX_DEPTH];
	__shared__ int node_idx[BTREE_MAX_DEPTH];
	int stack_idx = 0;

	node_stack[0] = ti[bidx].tree.root;
	node_idx[0]   = 0;
	while(stack_idx >= 0) {
		ASSERT(stack_idx < BTREE_MAX_DEPTH);
		ASSERT(node_idx[stack_idx] <= BTREE_NODE_CHILDREN); // may be equal if about to free

		if(node_idx[stack_idx] < BTREE_NODE_CHILDREN &&
			node_stack[stack_idx]->children[node_idx[stack_idx]] != NULL) {
			
			// more children, free them
			node_idx[stack_idx] += 1;
			node_idx[stack_idx + 1] = 0;
			node_stack[stack_idx + 1] = node_stack[stack_idx]->children[node_idx[stack_idx] - 1];
			stack_idx += 1;
		} else {
			// last child, free node
			if(idx == 0) 
				free(node_stack[stack_idx]);
			stack_idx -= 1;
		}
	}
}

/**
 * 	Destroys an index.
 */
template <typename T>
void db_indexes_destroy(table<T>* t) {
	// get number of indexes
	int n = t->offset_to_index_info.size();

	// launch destroy kernel
	dev_index_destroy<<<n, 32, 0, t->table_stream>>>(t->dev_indexes);

	// free indexes
	cudaFree(t->dev_indexes);
}

// explicit instantiations since nvcc doesn't support C++11
#define TABLE(T) \
	template __global__ void dev_indexes_create<T>(int n, table_index<T>*); \
	template int db_indexes_create<T>(table<T>*, size_t*, size_t*); \
	template __global__ void dev_index_destroy<T>(table_index<T>*); \
	template void db_indexes_destroy<T>(table<T>*);
#include "db_exp.h"
#undef TABLE
