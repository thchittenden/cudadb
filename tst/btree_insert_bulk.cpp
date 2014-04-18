#include "btree.h"

#include <algorithm>
#include <assert.h>
#include "CycleTimer.h"
#include <set>

#define SIZE 10000

int main() {

	int *N = new int[SIZE];
	for(int i = 0; i < SIZE; i++) {
		N[i] = i;
	}
	std::random_shuffle(N, N + SIZE);

	double pre_create = CycleTimer::currentSeconds();
	btree *t = btree_create();
	double post_create = CycleTimer::currentSeconds();
	printf("create took %f ms\n", 1000.f*(post_create - pre_create));

	double pre_insert = CycleTimer::currentSeconds();
	btree_insert_bulk(t, N, SIZE);
	double post_insert = CycleTimer::currentSeconds();
	printf("insert took %f ms\n", 1000.f*(post_insert - pre_insert));

	for(int i = 0; i < SIZE; i++) {
		assert(btree_contains(t, N[i]));
	}

	std::set<int> tree;	
	pre_insert = CycleTimer::currentSeconds();
	for(int i = 0; i < SIZE; i++) {
		tree.insert(N[i]);
	}
	post_insert = CycleTimer::currentSeconds();
	printf("std::insert took %f ms\n", 1000.f*(post_insert - pre_insert));


}
