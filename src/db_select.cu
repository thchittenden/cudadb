#include "common.h"
#include "db_types.h"
#include "db_defs.h"

#include "cuda_util.h"

#define DEFAULT_CRITERIA 0

template <typename T, int N>
static __device__ order evaluate_criteria(table_index<T>* idxs, criteria<T, N>* crit, T* elem, int i) {
	table_index<T>* idx = &idxs[crit->crit_idxs[i]];
	cmp_op crit_op = crit->crit_ops[i];
	order res = memcmp(elem, &crit->model, idx->key_offset, idx->key_size);
	
	switch(crit_op) {
	case LT:
		return res == ORD_LT ? ORD_EQ : ORD_GT;
	case LE:
		return res == ORD_LT || res == ORD_EQ ? ORD_EQ : ORD_GT;
	case EQ:
		return res;
	case GE:
		return res == ORD_EQ || res == ORD_GT ? ORD_EQ : ORD_LT;
	case GT:
		return res == ORD_GT ? ORD_EQ : ORD_LT;
		//TODO support NE
	}

	// we'll never reach here
	ASSERT(false);
	return ORD_NA;
}

/**
 *	Evaluates a set of criteria on an element and returns true if it passes. False otherwise.
 */
template <typename T, int N>
static __device__ bool evaluate_criterias(table_index<T>* idxs, criteria<T, N>* crit, T* elem) {
	for(int i = 0; i < N; i++) {
		if(evaluate_criteria(idxs, crit, elem, i) != ORD_EQ) {
			return false;
		}
	}
	return true;
}


template <typename T, int N>
static __global__ void dev_select_block(table_index<T>* idxs, criteria<T, N>* _crit, select_state<T>* state, result_buffer<T>* res) {
	int idx = LANE_INDEX();

	// load criteria into shared memory and pick an arbitrary tree
	__shared__ criteria<T, N> crit;
	__shared__ table_index<T>* index;
	crit = *_crit;
	index = &idxs[crit.crit_idxs[0]];

	// load state
	__shared__ btree_node<T*>* node_stack[BTREE_MAX_DEPTH];
	__shared__ int node_idx[BTREE_MAX_DEPTH];
	int stack_idx = state->stack_idx;
	if(state->node_stack[0] == NULL) {
		// new select, initialize state
		node_stack[0] = index->tree.root;
		node_idx[0] = 0;
	} else {
		// old select, load in node_stack from state
		if(idx < BTREE_MAX_DEPTH) {
			node_stack[idx] = state->node_stack[idx];
			node_idx[idx] = state->node_idx[idx];
		}
	}
	DEBUG(0, printf("searching index %p node %p at index %d\n", index, node_stack[stack_idx], node_idx[stack_idx])); 

	// fill result buffer
	__shared__ int res_fill;
	res_fill = 0;
	while(res_fill < SELECT_CHUNK_SIZE && stack_idx >= 0) {
		
		if(node_idx[stack_idx] == BTREE_NODE_KEYS) {
			// we've finished this level, recurse up
			stack_idx--;
			continue;
		}
		
		// get current node
		btree_node<T*>* cur_node = node_stack[stack_idx];
		
		// fetch node contents
		T* elem = cur_node->elems[idx];
		btree_node<T*>* child_node = cur_node->children[idx];
		
		bool is_leaf = __broadcast(child_node == NULL, 0); //TODO could know tree height
		if(is_leaf) {
			// at leaf node, add as many as possible
			int max_elems = SELECT_CHUNK_SIZE - res_fill;
			int min_idx   = node_idx[stack_idx];
			bool is_valid = idx >= min_idx && evaluate_criterias(idxs, &crit, elem);
			int num_valid = __warp_vote(is_valid);
			int my_idx    = __warp_exc_sum(is_valid);

			if(is_valid && my_idx < max_elems) {
				res->results[res_fill + my_idx] = *elem;
			}
			res_fill += min(num_valid, max_elems);

			if(max_elems < num_valid) {
				// more on this level, store node_idx 
				if(is_valid && my_idx == max_elems - 1) {
				ASSERT(idx < WARP_SIZE() - 1);
					node_idx[stack_idx] = idx + 1;
				}
			} else {
				// finished this node //TODO break early?
				stack_idx--;
			}

		} else {

			// not a leaf node, 
			int num_elems = __warp_vote(elem != NULL);
			int num_children = num_elems + 1;
			ASSERT(idx >= num_elems || elem != NULL);

			// compare all elements
			order ord = evaluate_criteria(idxs, &crit, elem, DEFAULT_CRITERIA);
			
			// find first non-LT node
			int child_idx, init_idx = node_idx[stack_idx];
			if(init_idx == 0) {
				// first recurse, find first valid index or last index if no one is valid
				child_idx = __first_lane_true_idx(ord != ORD_LT);
			} else {
				child_idx = init_idx;
				if(child_idx > num_children - 1) {
					// we've finished this node, go back up
					stack_idx--;
					continue;
				}
				if(__any(idx == child_idx - 1 && ord != ORD_EQ)) {
					// we stop recursing when the element before the child_idx
					// is not equal
					break;
				}
			}
		
			// check if node before was valid and if so, add it
			if(idx == child_idx - 1 && ord == EQ && evaluate_criterias(idxs, &crit, elem)) {
				res->results[res_fill] = *elem;
				res_fill++;
			}

			// now recurse into child node
			node_idx[stack_idx] = child_idx + 1;
			stack_idx++;
			node_idx[stack_idx] = 0;
			node_stack[stack_idx] = cur_node->children[child_idx];
		}

	}
	res->nvalid = res_fill;
	DEBUG(0, printf("found %d elements\n", res_fill));

	// store state
	state->stack_idx = stack_idx;
	if(idx < BTREE_MAX_DEPTH) {
		state->node_stack[idx] = node_stack[idx];
		state->node_idx[idx] = node_idx[idx];
	}
}

template <typename T, int N>
void db_dev_select_block(table<T>* t, criteria<T, N>* crit, select_state<T>* state, result_buffer<T>* res) {

	dev_select_block<T, N><<<1, 32, 0, t->table_stream>>>(t->dev_indexes, crit, state, res);

}

// explicit instantiation until nvcc supports variadic templates
#define CRITERIA(T, N) \
	template void db_dev_select_block<T, N>(table<T>*, criteria<T, N>*, select_state<T>*, result_buffer<T>*);
#include "db_exp.h"
#undef CRITERIA

