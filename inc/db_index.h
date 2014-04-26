#ifndef _DB_INDEX_H_
#define _DB_INDEX_H_

#include "db_types.h"

/**
 *	Creates a regular index which will reference the relevent data. 
 */
template <typename T>
table_index<T>* db_index_create(size_t member_offset, size_t member_size);

/**
 * 	Destroys a primary index.
 */
template <typename T>
void db_index_destroy(table_index<T> *t);

#endif
