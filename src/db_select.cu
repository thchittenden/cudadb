#include "db_types.h"
#include "db_defs.h"

enum ORD {
	LT,
	EQ,
	GT
};

template <typename T>
static __device__ ORD evaluate_criteria(T* _x1, T* _x2, size_t offset, size_t size) {
	char *x1 = (char*)_x1, *x2 = (char*)_x2;
	for(int i = offset; i < offset + size; i++) {
		if (x1[i] < x2[i]) return LT;
		if (x1[i] > x2[i]) return GT;
	}
	return EQ;
}

template <typename T, int N>
static __global__ void cuda_select_block(criteria<T, N>* sel_crit, select_state<T>* state, result_buffer<T>* res) {

	int idx = LANE_INDEX();

	// load criteria into shared memory and pick an arbitrary tree
	__shared__ criteria<T, N> crit = *sel_crit;
	__shared__ btree<T*>* tree = &crit.

	// load state
	__shared__ btree_node<T*>* node_stack[BTREE_MAX_DEPTH];
	__shared__ int node_idx[BTREE_MAX_DEPTH];
	int root_idx = state.root_idx;
	int stack_idx = state.stack_idx;
	if(state->node_stack[0] == NULL) {
		// new select, initialize state
		node_stack[0] = tree->roots[0];
	} else {
		// old select, load in node_stack from state
		if(idx < BTREE_MAX_DEPTH) {
			node_stack[idx] = state->node_stack[idx];
			node_idx[idx] = state->node_idx[idx];
		}
	}

	// fill result buffer
	int res_fill = 0;
	while(res_fill < SELECT_CHUNK_SIZE) {
		btree_node<T*>* cur_node = node_stack[stack_idx];
		T* elem = cur_node->elem[idx];
		ORD res = evaluate_criteria
	}
	res->nvalid = res_fill;

	// store state
	state->root_idx = root_idx;
	state->stack_idx = stack_idx;
	if(idx < BTREE_MAX_DEPTH) {
		state->node_stack[idx] = node_stack[idx];
		state->node_idx[idx] = node_idx[idx];
	}
}

template <typename T, int N>
void db_select_block(criteria<T, N>* crit, select_state<T>* state, result_buffer<T>* res) {

	cuda_select_block<T, N><<<1, 32>>>(crit, state, res);

}

// explicit instantiation until nvcc supports variadic templates
#define CRITERIA(T, N) \
	template void db_select_block<T, N>(criteria<T, N>*, select_state<T>*, result_buffer<T>*);
#include "db_exp.h"
#undef CRITERIA

