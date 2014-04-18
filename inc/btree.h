#ifndef _BTREE_H_
#define _BTREE_H_

struct btree;

btree *btree_create();

bool btree_contains(btree *, int);

void btree_insert(btree *, int);

void btree_insert_bulk(btree *, int *, int);

#endif

