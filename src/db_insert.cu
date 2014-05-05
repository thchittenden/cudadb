#include "common.h"
#include "db_types.h"

#include "cuda_common.h"
#include "cuda_util.h"

/**
 *	Inserts a value into an unfull node.
 *
 *  BLOCK SIZE: 32x1
 *  GRID SIZE:  1x1
 */
template <typename T>
static __device__ void dev_btree_node_insert(btree_node<T*>* node, T* x, size_t key_offset, size_t key_size, btree_node<T*>* xp) {
	ASSERT(node != NULL);

	int idx = LANE_INDEX();
	T* key = node->elems[idx];
	ASSERT(__any(idx < BTREE_NODE_KEYS && key == NULL)); // assert not full

	// get the first lane where the key is greater than the new value or an empty lane
	order cmp = memcmp(key, x, key_offset, key_size);
	int insert_idx = __first_lane_true_idx(cmp == ORD_NA || cmp == ORD_GT);
	ASSERT(insert_idx < BTREE_NODE_KEYS);

	// shift old elems
	if(insert_idx <= idx && key != NULL) {
		ASSERT(idx + 1 < BTREE_NODE_KEYS);
		ASSERT(idx + 2 < BTREE_NODE_CHILDREN);
		node->elems[idx + 1] = key;
		node->children[idx + 2] = node->children[idx + 1];
	}

	// insert new key
	node->elems[insert_idx] = x;
	node->children[insert_idx + 1] = xp;
}

template <typename T>
static __device__ btree_node<T*>* dev_btree_node_split(table_index<T>* index, btree_node<T*>* parent, btree_node<T*>* left) {
	ASSERT(index != NULL);
	ASSERT(left != NULL);

	// declare shared memory and get our warps chunk
	__shared__ T* scratch_elems[BTREE_NODE_KEYS+1];
	__shared__ btree_node<T*>* scratch_children[BTREE_NODE_CHILDREN];

	int idx = LANE_INDEX();
	T* key = left->elems[idx];
	ASSERT(__all(key != NULL || idx >= BTREE_NODE_KEYS)); // assert full

	// initialize shared memory
	scratch_elems[idx] = key;
	scratch_children[idx] = left->children[idx];

	// allocate new node
	btree_node<T*>* right;
	if(idx == 0) {
		right = (btree_node<T*>*)malloc(sizeof(*right));
		ASSERT(right != NULL);
	}
	right = (btree_node<T*>*)__broadcast_ptr(right, 0);

	// populate nodes TODO more effiently on left
	left->elems[idx] = idx < BTREE_NODE_KEYS/2 ? scratch_elems[idx] : NULL;
	left->children[idx] = idx < BTREE_NODE_CHILDREN/2 ? scratch_children[idx] : NULL;

	right->elems[idx] = idx < BTREE_NODE_KEYS/2 ? scratch_elems[idx + BTREE_NODE_KEYS/2 + 1] : NULL;
	right->children[idx] = idx < BTREE_NODE_CHILDREN/2 ? scratch_children[idx + BTREE_NODE_CHILDREN/2] : NULL;
	
	T* split = scratch_elems[BTREE_NODE_KEYS/2];

	if(parent == NULL) {
		// we were splitting root, make new root
		btree_node<T*>* new_root;
		if(idx == 0) {
			new_root = (btree_node<T*>*)malloc(sizeof(*new_root));
			ASSERT(new_root != NULL);
		}
		new_root = (btree_node<T*>*)__broadcast_ptr(new_root, 0);
		
		// initialize new root
		new_root->elems[idx] = idx == 0 ? split : NULL;
		new_root->children[idx] = idx == 0 ? left : idx == 1 ? right : NULL;
		
		// set new root in tree
		index->tree.root = new_root;
	} else {
		// insert right into parent, assume parent not full
		dev_btree_node_insert(parent, split, index->key_offset, index->key_size, right);
	}

	return right;
}

template <typename T>
__device__ void dev_insert_elem(table_index<T>* index, T* elem) {
	
	DEBUG(0, printf("inserting %p into index %p\n", elem, index));

	int idx = LANE_INDEX();
	size_t key_offset = index->key_offset;
	size_t key_size   = index->key_size;

	// traverse down to a leaf node, splitting as we go
	btree_node<T*>* parent = NULL;
	btree_node<T*>* cur = index->tree.root;

	while(1) {
		ASSERT(cur != NULL);
		
		T* key = cur->elems[idx];
		order cmp = memcmp(key, elem, key_offset, key_size);
	
		// get first key greater than our elem
		int node_idx = __first_lane_true_idx(cmp == ORD_NA || cmp == ORD_GT);

		// split the node if we need to
		if(__all(key != NULL || idx >= BTREE_NODE_KEYS)) {
			// node full, split it
			btree_node<T*>* new_node = dev_btree_node_split(index, parent, cur);
			if(node_idx >= BTREE_NODE_CHILDREN/2) {
				// need to go into right index
				cur = new_node;
				node_idx -= BTREE_NODE_CHILDREN/2;
			}
		}

		// break if we've reached a leaf node
		if(cur->children[0] == NULL) { // TODO we know height, only recurse n times
			break;
		} 
		
		// store node in path and advance, key[31] never present so we won't go out of bounds
		parent = cur;
		cur = cur->children[node_idx];
	}

	// cur now equals leaf, insert into it
	dev_btree_node_insert(cur, elem, index->key_offset, index->key_size, (btree_node<T*>*)NULL);
}

template <typename T>
__global__ void dev_insert(table_index<T>* indexes, T* elems, int n) {

	// get this blocks tree
	table_index<T>* index = &indexes[BLOCK_INDEX()];
	
	// insert each element 
	for(int i = 0; i < n; i++) {
		dev_insert_elem(index, &elems[i]);
	}
}

template <typename T>
int db_dev_insert(table<T>* t, T* elems, int n) {

	// get number of indexes
	int nindexes = t->offset_to_index_info.size();

	// copy elements over to device
	T* dev_elems;
	HANDLE_ERROR(cudaMalloc(&dev_elems, sizeof(T)*n), goto err0);
	HANDLE_ERROR(cudaMemcpyAsync(dev_elems, elems, sizeof(T)*n, cudaMemcpyHostToDevice, t->table_stream), goto err1);

	// insert into indexes
	dev_insert<<<nindexes, 32, 0, t->table_stream>>>(t->dev_indexes, dev_elems, n);
	t->size += n;
	return 0;

err1:
	cudaFree(dev_elems);
err0:
	return -1;
}



#define TABLE(T) \
	template int db_dev_insert<T>(table<T>*, T*, int);
#include "db_exp.h"
#undef TABLE
