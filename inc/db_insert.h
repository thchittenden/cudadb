#ifndef _DB_INSERT_H_
#define _DB_INSERT_H_

#include "db_types.h"

template <typename T>
int db_dev_insert(table<T>*, T*, int);

template <typename T>
int db_insert(table<T>* t, T& elem) {
	return db_dev_insert(t, &elem, 1);
}

template <typename T>
int db_insert_bulk(table<T>* t, T* elems, int n) {
	return db_dev_insert(t, elems, n);
}	

#endif
