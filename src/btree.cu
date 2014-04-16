#include "btree.h"
#include "cuda_common.h"

#include <cuda.h>

#define BTREE_KEY_NOT_PRESENT (1 << (sizeof(int)*8 - 1)) 

#define BTREE_NODE_KEYS 32
#define BTREE_NODE_CHILDREN (BTREE_NODE_KEYS + 1)

struct btree_node {
	int keys[BTREE_NODE_KEYS]; 
	btree_node *children[BTREE_NODE_CHILDREN];
	btree_node *parent;
};

struct btree {
	btree_node *root;
};

/**
 *	Initializes a new btree.
 *
 *	BLOCK SIZE: 32x1
 *	GRID SIZE:  1x1
 */
static __global__ void dev_btree_create(btree *tree) {
	
	int idx = WARP_INDEX();
	
	if(idx == 0) {
		tree->root = (btree_node *)malloc(sizeof(btree_node));
		tree->root->children[BTREE_NODE_CHILDREN - 1] = NULL;
	}
	tree->root->keys[idx] = BTREE_KEY_NOT_PRESENT;
	tree->root->children[idx] = NULL;

}

btree *btree_create() {
	btree* tree;
	cudaMalloc(&tree, sizeof(btree));
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

	int idx = WARP_INDEX();
	
	btree_node *cur = tree->root;
	while(cur != NULL) {
		
		int key = cur->keys[idx];

		// check if key matches
		if(__any(key == x)) {
			*res = true;
			return;
		} 

		// find index into node
		int node_idx = __ballot(key != BTREE_KEY_NOT_PRESENT && key < x);
		cur = cur->children[node_idx];

	}

	*res = false;
	return;
}

bool btree_contains(btree *tree, int x) {
	
	bool res;
	bool *res_d;
	cudaMalloc(&res_d, sizeof(bool));
	dev_btree_contains<<<1, 32>>>(tree, x, res_d);
	cudaMemcpy(&res, res_d, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(res_d);
	return res;

}

/**
 *	Inserts a value into an unfull node.
 *
 *  BLOCK SIZE: 32x1
 *  GRID SIZE:  1x1
 */
__device__ static void dev_btree_node_insert(btree_node *node, int x, btree_node *xp) {

	int idx = WARP_INDEX();

	int key = node->keys[idx];

	int fill = __ballot(key != BTREE_KEY_NOT_PRESENT);
	ASSERT(fill < BTREE_NODE_KEYS);

	int insert_idx = __ballot(key != BTREE_KEY_NOT_PRESENT && key < x);
	ASSERT(insert_idx < BTREE_NODE_KEYS);

	// shift old keys
	if(insert_idx <= idx && idx < fill) {
		ASSERT(idx + 1 < BTREE_NODE_KEYS);
		ASSERT(idx + 2 < BTREE_NODE_CHILDREN);
		node->keys[idx + 1] = key;
		node->children[idx + 2] = node->children[idx + 1];
	}

	// insert new key
	node->keys[insert_idx] = x;
	node->children[insert_idx + 1] = xp;
	xp->parent = node;
}

/**
 *	Traverses back up the tree inserting values and splitting nodes as necessary.
 *
 *	BLOCK SIZE: 32x1
 *	GRID SIZE:  1x1
 */
__device__ static void dev_btree_insert_upsweep(btree *tree, btree_node *cur, int split, btree_node *rchild) {
	
	__shared__ btree_node* new_rchild;
	__shared__ btree_node* new_root;
	int idx = WARP_INDEX();

	while(cur != NULL) {

		// get parent keys
		int key = cur->keys[idx];

		// get parent fill
		int fill = __ballot(key == BTREE_KEY_NOT_PRESENT);

		if(fill == BTREE_NODE_KEYS) {
			// need to split node, allocate and zero
			if(idx == 0) {
				new_rchild = (btree_node*)malloc(sizeof(btree_node));
			}
			new_rchild->keys[idx] = BTREE_KEY_NOT_PRESENT;
			new_rchild->children[idx + 1] = NULL;

			// copy current node over
			if(idx < BTREE_NODE_KEYS/2) {
				new_rchild->keys[idx] = cur->keys[idx + BTREE_NODE_KEYS/2];
				new_rchild->children[idx + 1] = cur->children[idx + BTREE_NODE_KEYS/2 + 1];
				cur->keys[idx + BTREE_NODE_KEYS/2] = BTREE_KEY_NOT_PRESENT;
				cur->children[idx + BTREE_NODE_KEYS/2 + 1] = NULL;
			}
			
			// insert rchild into new node, update rchild
			new_rchild->children[0] = rchild;
			rchild->parent = new_rchild;
			rchild = new_rchild;
			
			// check if we should swap current split with parent split
			if(idx == 0 && new_rchild->keys[0] < split) {
				int tmp = new_rchild->keys[0];
				new_rchild->keys[0] = split;
				split = tmp;

				btree_node *tmp2 = new_rchild->children[0];
				new_rchild->children[0] = new_rchild->children[1];
				new_rchild->children[1] = tmp2;
			}
			
			// check if we need to create new root
			if(cur->parent == NULL) {
				if(idx == 0) {
					new_root = (btree_node*)malloc(sizeof(btree_node));	
					new_root->children[0] = cur;
					tree->root = new_root;
					cur->parent = new_root;
				}
				new_root->keys[idx] = BTREE_KEY_NOT_PRESENT;
				new_root->children[idx + 1] = NULL;
			}	
			
			// update parent for next iteration
			cur = cur->parent;


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
static __global__ void dev_btree_insert(btree *tree, int x) {

	int idx = WARP_INDEX();

	// find a leaf node
	btree_node *cur = tree->root;
	while(1) {
		
		int key = cur->keys[idx];

		if(__any(key == x)) {
			// value already in tree
			return; 
		}

		// find index into node
		int node_idx = __ballot(key != BTREE_KEY_NOT_PRESENT && key < x);
		if(cur->children[node_idx] == NULL) {
			break;
		} 

		cur = cur->children[node_idx];
	}

	// cur is now a leaf node, insert into it
	dev_btree_insert_upsweep(tree, cur, x, NULL);
}

void btree_insert(btree *tree, int x) {
	
	dev_btree_insert<<<1, 32>>>(tree, x);

}
