#ifndef _DB_UTIL_H_
#define _DB_UTIL_H_

#include <cstddef> // for size_t

/**
 *	Applies an operation across a parameter packet.
 */
#define _GLUE(x, y) x ## y
#define GLUE(x, y) _GLUE(x, y)
#define APPLY(expr) int GLUE(_apply, __LINE__)[] = {(expr, 0)...}; (void)GLUE(_apply,__LINE__);

#define ALIGN_UP(p, x) (decltype(p))(((intptr_t)p + x-1) & ~(x-1)) 

/**
 *	Returns the offset into struct T of an arbitrary member M.
 */	
template <typename T, typename M>
std::size_t offset_member(M T::* m) {
	return reinterpret_cast<std::size_t>(&(((T*)0)->*m));
}

#endif
