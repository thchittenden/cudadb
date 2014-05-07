#include "common.h"
#include "db_types.h"
#include "db_defs.h"

#include "cuda_util.h"

template <typename T>
static __device__ order evaluate_compare_criteria(void* crit_str, T* elem) {
	ASSERT(crit_str != NULL);
	if(elem == NULL) {
		return ORD_NA;
	}

	crit_block* block = (crit_block*)crit_str;
	crit_tag op       = block->tag;
	size_t key_offset = block->comp.offset;
	size_t key_size   = block->comp.size;
	order res = memcmp((char *)elem + key_offset, (char *)&block->comp.val, key_size);
	
	switch(op) {
	case LT_TAG:
		return res == ORD_LT ? ORD_EQ : ORD_GT;
	case LE_TAG:
		return res == ORD_LT || res == ORD_EQ ? ORD_EQ : ORD_GT;
	case EQ_TAG:
		return res;
	case GE_TAG:
		return res == ORD_GT || res == ORD_EQ ? ORD_EQ : ORD_LT;
	case GT_TAG:
		return res == ORD_GT ? ORD_EQ : ORD_LT;
	default:
		ASSERT(false); // bad crit tag
	}

	// should never reach here
	return ORD_NA;
}

template <typename T>
static __device__ bool evaluate_criteria(char* crit_str, T* elem) {
	__shared__ int  eval_rem  [CRITERIA_MAX_DEPTH];
	__shared__ bool eval_op   [CRITERIA_MAX_DEPTH];
	__shared__ crit_block* crit_ptr;
	crit_ptr = (crit_block*)crit_str; 

	int eval_idx = 0;

	// initialize first level 
	bool res = true;
	eval_rem  [eval_idx] = 1;
	eval_op   [eval_idx] = true;

	// TODO i think we can replace eval_stack with a single boolean
	while(eval_idx >= 0) {
		
		// loop until the current result is different than what the operation requires to complete
		while(eval_rem[eval_idx] > 0 && res == eval_op[eval_idx]) {
			
			switch(crit_ptr->tag) {
				case OR_TAG:
				case AND_TAG: {
					bool def = crit_ptr->tag == AND_TAG;
					eval_rem[eval_idx] -= 1;
					eval_idx += 1;
					eval_rem [eval_idx] = crit_ptr->comb.size;
					eval_op  [eval_idx] = def;
					res = def;

					//advance crit_ptr
					crit_ptr = (crit_block*)(&crit_ptr->comb.sub_blocks);
				} continue;
				
				case LT_TAG:
				case LE_TAG:
				case EQ_TAG:
				case GE_TAG:
				case GT_TAG: {
					res = evaluate_compare_criteria(crit_ptr, elem) == ORD_EQ;
					
					// advance crit_ptr
					crit_ptr = (crit_block*)(&crit_ptr->comp.val[ALIGN_UP(crit_ptr->comp.size, 8)]);
				} break;
			}

			ASSERT(eval_idx < CRITERIA_MAX_DEPTH);
			eval_rem[eval_idx] -= 1;
		}
		
		
		// finished a level, go down and update result
		eval_idx -= 1;
	}

	return res;
}

template <typename T>
static __global__ void dev_select_block(
							table_index<T>* index, 
							int index_crit_str_len,
							char* _index_crit_str, 
							int crit_str_len,
							char* _crit_str, 
							select_state<T>* state, 
							result_buffer<T>* res) {
	int idx = LANE_INDEX();

	// declare dynamic shared memory
	extern __shared__ char crit_cache[];
	char* index_crit_str = &crit_cache[0];
	char* crit_str       = &crit_cache[index_crit_str_len];

	// load criteria into shared memory
	for(int i = idx; i < index_crit_str_len; i += WARP_SIZE()) {
		index_crit_str[i] = _index_crit_str[i];
	} 
	for(int i = idx; i < crit_str_len; i += WARP_SIZE()) {
		crit_str[i] = _crit_str[i];
	}

	// load state
	__shared__ btree_node<T*>* node_stack[BTREE_MAX_DEPTH];
	__shared__ int node_idx[BTREE_MAX_DEPTH];
	int stack_idx = state->stack_idx;
	if(state->node_stack[0] == NULL) {
		// new select, TODO determine best index out of criteria
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
			bool is_valid = idx >= min_idx && evaluate_criteria(crit_str, elem);
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
			order ord = evaluate_compare_criteria(index_crit_str, elem);
			
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
			if(idx == child_idx - 1 && ord == ORD_EQ && evaluate_criteria(crit_str, elem)) {
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

template <typename T>
void db_dev_select_block(select_handle<T>* handle) {
	
	// allocate dynamic shared memory for criteria 
	int dmem = handle->dev_idx_crit_str_len + handle->dev_crit_str_len;

	// launch select block
	dev_select_block<T><<<1, 32, dmem, handle->t->table_stream>>>(
		handle->dev_idx, 
		handle->dev_idx_crit_str_len, 
		handle->dev_idx_crit_str, 
		handle->dev_crit_str_len, 
		handle->dev_crit_str,
		handle->dev_state,
		handle->dev_buf);

}

// explicit instantiation until nvcc supports variadic templates
#define TABLE(T) \
	template void db_dev_select_block<T>(select_handle<T>*);
#include "db_exp.h"
#undef CRITERIA

