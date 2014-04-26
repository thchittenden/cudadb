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
template <typename PT>
static __global__ void dev_btree_create(btree<PT>* tree, size_t key_offset, size_t key_size) {
	ASSERT(tree != NULL);
	int idx = LANE_INDEX();

	// initialize key info
	tree->key_offset = key_offset;
	tree->key_size = key_size;

	// initialize all roots
	btree_node<PT>* my_node;
	if(idx < BTREE_NUM_ROOTS) {
		my_node = (btree_node<PT> *)malloc(sizeof(btree_node<PT>));
		ASSERT(my_node != NULL);
		tree->roots[idx] = my_node;
	}

	// zero all roots in warp
	btree_node<PT>* cur_node;
	for(int i = 0; i < BTREE_NUM_ROOTS; i++) {
		cur_node = (btree_node<PT>*)__broadcast_ptr(my_node, i);
		cur_node->elems[idx] = NULL;
		cur_node->children[idx] = NULL;
	}
}

/**
 *	Creates an index which will reference the data.
 */
template <typename T>
table_index<T>* db_index_create(size_t key_offset, size_t key_size) {
	table_index<T>* t = new table_index<T>;
	
	// allocate and initialize btree
	cudaMalloc(&t->dev_btree, sizeof(btree<T*>));
	dev_btree_create<T*><<<1, 32>>>(t->dev_btree, key_offset, key_size);

	// return index
	return t;
}

/**
 *	Destroys a btree and frees all contained nodes.
 *
 *	BLOCK SIZE: 32x1
 *  GRID  SIZE: 15x1
 */
template <typename PT>
static __global__ void dev_btree_destroy(btree<PT> *tree) {
	int idx = LANE_INDEX();
	int bidx = BLOCK_INDEX();

	__shared__ btree_node<PT>* node_stack[BTREE_MAX_DEPTH];
	__shared__ int node_idx[BTREE_MAX_DEPTH];
	int stack_idx = 0;

	node_stack[0] = tree->roots[bidx];
	node_idx[0]   = 0;
	while(stack_idx >= 0) {
		ASSERT(stack_idx < BTREE_MAX_DEPTH);
		ASSERT(node_idx[stack_idx] <= BTREE_NODE_CHILDREN); // may be equal if about to free

		if(node_idx[stack_idx] < BTREE_NODE_CHILDREN &&
			node_stack[stack_idx]->children[node_idx[stack_idx]] != NULL) {
			// more children, free them
			node_idx[stack_idx] += 1;
			node_idx[stack_idx + 1] = 0;
			node_stack[stack_idx + 1] = node_stack[stack_idx]->children[node_idx[stack_idx]];
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
void db_index_destroy(table_index<T> *t) {
	dev_btree_destroy<T*><<<15, 32>>>(t->dev_btree);
	cudaFree(t->dev_btree);
	delete t;
}

// explicit instantiations since nvcc doesn't support C++11
#define TABLE(T) \
	template __global__ void dev_btree_create<T*>(btree<T*>*, size_t, size_t); \
	template table_index<T>* db_index_create<T>(size_t, size_t); \
	template __global__ void dev_btree_destroy<T*>(btree<T*>*); \
	template void db_index_destroy<T>(table_index<T>*);
#include "db_exp.h"
#undef TABLE
