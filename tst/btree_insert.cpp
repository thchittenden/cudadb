#include "btree.h"

#include <assert.h>

int main() {
	
	btree *t = btree_create();

	btree_insert(t, 10);
	btree_insert(t, 5);
	btree_insert(t, 2);
	
	assert(btree_contains(t, 10));
	assert(btree_contains(t, 5));
	assert(btree_contains(t, 2));
	assert(!btree_contains(t, 3));
	assert(!btree_contains(t, 11));

	for(int i = 100; i < 1000; i++) {
		btree_insert(t, i);
	}

	for(int i = 100; i < 1000; i++) {
		assert(btree_contains(t, i));
	}

}
