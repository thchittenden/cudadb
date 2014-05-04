#ifndef _DB_INDEX_H_
#define _DB_INDEX_H_

#include "db_types.h"

/**
 *	Creates a regular index which will reference the relevent data. 
 */
template <typename T>
int db_indexes_create(table<T>*, size_t* key_offsets, size_t* key_sizes);

/**
 * 	Destroys a primary index.
 */
template <typename T>
void db_indexes_destroy(table<T>*);

#endif
