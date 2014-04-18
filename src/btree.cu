#include "btree.h"
#include "cuda_common.h"
#include "cuda_util.h"

#include <cuda.h>

#define BTREE_KEY_NOT_PRESENT (1 << (sizeof(int)*8 - 1)) 

#define BTREE_NUM_ROOTS 15

#define BTREE_NODE_KEYS 31
#define BTREE_NODE_CHILDREN (BTREE_NODE_KEYS + 1)

// due to high branching factor (~16), never going to be taller than 8
#define BTREE_MAX_DEPTH 8

struct btree_node {
	int keys[BTREE_NODE_KEYS + 1]; // add an extra key slot to make processing easier/align children
	btree_node* children[BTREE_NODE_CHILDREN];
};

struct btree {
	btree_node* roots[BTREE_NUM_ROOTS];
};

/**
 *	Initializes a new btree.
 *
 *	BLOCK SIZE: 32x1
 *	GRID SIZE:  1x1
 */
static __global__ void dev_btree_create(btree *tree) {
	ASSERT(tree != NULL);

	int idx = WARP_INDEX();

	// initialize all roots
	btree_node* my_node;
	if(idx < BTREE_NUM_ROOTS) {
		my_node = (btree_node *)malloc(sizeof(btree_node));
		ASSERT(my_node != NULL);
		tree->roots[idx] = my_node;
	}

	// zero all roots
	btree_node* cur_node;
	for(int i = 0; i < BTREE_NUM_ROOTS; i++) {
		cur_node = (btree_node*)__broadcast_ptr(my_node, i);
		cur_node->keys[idx] = BTREE_KEY_NOT_PRESENT;
		cur_node->children[idx] = NULL;
	}
}

btree *btree_create() {
	btree* tree;
	HANDLE_ERROR(cudaMalloc(&tree, sizeof(btree)), return NULL);
	dev_btree_create<<<1, 32>>>(tree);
	return tree;
}

/**
 *	Determines whether a single element is in the tree.
 *  
 *  BLOCK_SIZE: 32x1
 *  GRID_SIZE:  1x1
 */
static __global__ void dev_btree_contains(btree *tree, int x, bool *res) {
	ASSERT(tree != NULL);
	ASSERT(res != NULL);

	int idx = WARP_INDEX();
	int bidx = BLOCK_INDEX();

	btree_node *cur = tree->roots[bidx];
	ASSERT(cur != NULL);
	while(cur != NULL) {
		
		int key = cur->keys[idx];

		// check if key matches
		if(__any(key == x)) {
			*res = true;
			return;
		} 

		// find index into node, key[31] never present so we don't go out of bounds
		int node_idx = __first_lane_true_idx(key == BTREE_KEY_NOT_PRESENT || key > x);
		cur = cur->children[node_idx];
	}

	return;
}


bool btree_contains(btree *tree, int x) {
	
	bool res = false;
	bool *res_d; // TODO reuse bools
	HANDLE_ERROR(cudaMalloc(&res_d, sizeof(bool)), exit(1));
	HANDLE_ERROR(cudaMemset(res_d, 0, sizeof(bool)), exit(1));
	dev_btree_contains<<<BTREE_NUM_ROOTS, 32>>>(tree, x, res_d);
	HANDLE_ERROR(cudaMemcpy(&res, res_d, sizeof(bool), cudaMemcpyDeviceToHost), exit(1));
	HANDLE_ERROR(cudaFree(res_d), exit(1));
	return res;

}

/**
 *	Inserts a value into an unfull node.
 *
 *  BLOCK SIZE: 32x1
 *  GRID SIZE:  1x1
 */
static __device__ void dev_btree_node_insert(btree_node *node, int x, btree_node *xp) {
	ASSERT(node != NULL);

	int idx = WARP_INDEX();

	int key = node->keys[idx];
	ASSERT(__any(idx < BTREE_NODE_KEYS && key == BTREE_KEY_NOT_PRESENT));

	// get the first lane where the key is greater than the new value or an empty lane
	int insert_idx = __first_lane_true_idx(key == BTREE_KEY_NOT_PRESENT || key > x);
	ASSERT(insert_idx < BTREE_NODE_KEYS);

	// shift old keys
	if(insert_idx <= idx && key != BTREE_KEY_NOT_PRESENT) {
		ASSERT(idx + 1 < BTREE_NODE_KEYS);
		ASSERT(idx + 2 < BTREE_NODE_CHILDREN);
		node->keys[idx + 1] = key;
		node->children[idx + 2] = node->children[idx + 1];
	}

	// insert new key
	node->keys[insert_idx] = x;
	node->children[insert_idx + 1] = xp;
}

/**
 *	Traverses back up the tree inserting values and splitting nodes as necessary.
 *
 *	BLOCK SIZE: 32x1
 *	GRID SIZE:  1x1
 */
static __device__ void dev_btree_insert_upsweep(btree *tree, 
												int root,
												btree_node *(*path)[BTREE_MAX_DEPTH], 
												int path_level, 
												int split, 
												btree_node *rchild) {
	
	int idx = WARP_INDEX();
	
	__shared__ int scratch_keys[BTREE_NODE_KEYS + 1];
	__shared__ btree_node* scratch_children[BTREE_NODE_CHILDREN + 1];

	btree_node* new_node;
	btree_node* cur = (*path)[path_level];
	while(true) {

		// get parent keys
		int key = cur->keys[idx];

		if(__all(idx >= BTREE_NODE_KEYS || key != BTREE_KEY_NOT_PRESENT)) {
			// need to split node, allocate and distribute
			if(idx == 0) {
				new_node = (btree_node*)malloc(sizeof(btree_node));
				ASSERT(new_node != NULL);
			}
			new_node = (btree_node*)__broadcast_ptr(new_node, 0);			
			
			// zero the node
			new_node->keys[idx] = BTREE_KEY_NOT_PRESENT;
			new_node->children[idx] = NULL;
		
			// initialize shared node, scratch_keys contains all the keys + split,
			// scratch_rchildren contains the corresponding right children for each key.
			// we only include right children since the leftmost child never moves.
			scratch_keys[idx] = key;
			scratch_children[idx] = cur->children[idx];
			
			// insert split into scratch node
			int insert_idx = __first_lane_true_idx(key == BTREE_KEY_NOT_PRESENT || key > split);
			if(idx >= insert_idx && idx < BTREE_NODE_KEYS) {
				scratch_keys[idx + 1] = scratch_keys[idx];
				scratch_children[idx + 2] = scratch_children[idx + 1];
			}
			scratch_keys[insert_idx] = split;
			scratch_children[insert_idx + 1] = rchild;
			
			// write scratch data out to new nodes
#if BTREE_NODE_KEYS % 2 == 0
			// even number of keys, no first child in SM
			cur->keys[idx] = idx < BTREE_NODE_KEYS/2 ? scratch_keys[idx] : BTREE_KEY_NOT_PRESENT;
			cur->children[idx + 1] = idx < BTREE_NODE_CHILDREN/2 ? scratch_rchildren[idx] : NULL;
			
			new_node->keys[idx] = idx < BTREE_NODE_KEYS/2 ? scratch_keys[idx + BTREE_NODE_KEYS/2 + 1] : BTREE_KEY_NOT_PRESENT;
			new_node->children[idx] = idx < (BTREE_NODE_CHILDREN + 1)/2 ? scratch_rchildren[idx + BTREE_NODE_CHILDREN/2] : NULL;

			// update split and rchild
			rchild = new_node;
			split = scratch_keys[BTREE_NODE_KEYS/2];
#else
			// odd number of keys, first child in SM TODO bias
			cur->keys[idx] = idx < BTREE_NODE_KEYS/2 ? scratch_keys[idx] : BTREE_KEY_NOT_PRESENT;
			cur->children[idx] = idx < BTREE_NODE_CHILDREN/2 ? scratch_children[idx] : NULL;

			new_node->keys[idx] = idx < (BTREE_NODE_KEYS+1)/2 ? scratch_keys[idx + (BTREE_NODE_KEYS+1)/2] : BTREE_KEY_NOT_PRESENT;
			new_node->children[idx] = idx < BTREE_NODE_CHILDREN/2 + 1 ? scratch_children[idx + BTREE_NODE_CHILDREN/2] : NULL;
			
			// update split and rchild
			rchild = new_node;
			split = scratch_keys[BTREE_NODE_KEYS/2];
#endif

			// check if we're at root and need a new node
			if(path_level == 0) {
				if(idx == 0) {
					new_node = (btree_node*)malloc(sizeof(btree_node));	
					ASSERT(new_node != NULL);
					tree->roots[root] = new_node;
				}
				new_node = (btree_node*)__broadcast_ptr(new_node, 0);
				new_node->keys[idx] = BTREE_KEY_NOT_PRESENT;
				new_node->children[idx] = idx == 0 ? cur : NULL;
				cur = new_node;
			} else {
				cur = (*path)[--path_level];
			}

		} else {
			// node not full, insert into 
			dev_btree_node_insert(cur, split, rchild);
			break;
		}

	}

}

/**
 *	Inserts a single element into the tree.
 *
 * 	BLOCK_SIZE: 32x1
 *  GRID_SIZE:  1x1
 */
static __device__ void dev_btree_insert(btree *tree, int root, int x) {
	ASSERT(tree != NULL);
	int idx = WARP_INDEX();

	__shared__ btree_node* path[BTREE_MAX_DEPTH];
	int path_level = 0;
	if(idx == 0) {
		path[0] = tree->roots[root];
	}

	__threadfence(); //ensure write to SM is seen

	// find a leaf node
	btree_node *cur = path[path_level];
	while(1) {
		
		int key = cur->keys[idx];

		if(__any(key == x)) {
			// value already in tree
			return; 
		}
		
		// break if we've reached a leaf node
		if(cur->children[0] == NULL) {
			break;
		} 
		
		// store node in path and advance, key[31] never present so we won't go out of bounds
		int node_idx = __first_lane_true_idx(key == BTREE_KEY_NOT_PRESENT || key > x);
		cur = cur->children[node_idx];
		path[++path_level] = cur;
	}

	// cur is now a leaf node, insert into it
	dev_btree_insert_upsweep(tree, root, &path, path_level, x, NULL);
}

static __global__ void dev_btree_insert_single(btree *tree, int root, int x) {
	dev_btree_insert(tree, root, x);
}

void btree_insert(btree *tree, int x) {
	
	static int next_insert = 0;
	dev_btree_insert_single<<<1, 32>>>(tree, next_insert, x);
	next_insert = (next_insert + 1) % BTREE_NUM_ROOTS;

}

static __global__ void dev_btree_insert_bulk(btree *tree, int *xs, int n) {
	for(int i = BLOCK_INDEX(); i < n; i += GRID_SIZE()) {
		dev_btree_insert(tree, BLOCK_INDEX(), xs[i]);
	}
}

void btree_insert_bulk(btree *tree, int *xs, int n) {

	int *xs_d;
	HANDLE_ERROR(cudaMalloc(&xs_d, sizeof(int) * n), exit(1));
	HANDLE_ERROR(cudaMemcpy(xs_d, xs, sizeof(int) * n, cudaMemcpyHostToDevice), exit(1));
	dev_btree_insert_bulk<<<BTREE_NUM_ROOTS, 32>>>(tree, xs_d, n);
	HANDLE_ERROR(cudaFree(xs_d), exit(1));
	return;
}
