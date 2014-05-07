#ifndef _DB_CRITERIA_H_
#define _DB_CRITERIA_H_

#include <db_types.h>
#include <db_util.h>

#include <type_traits>
#include <cstring>

namespace _impl {
	// "hide" helper methods 
	template <typename T, typename A>
	criteria<T> crit_compare(crit_tag t, A T::* m, A val) {
		
		// calculate length and allocate block, align up to 8
		size_t len = ALIGN_UP(sizeof(crit_block) + sizeof(A), 8);
		crit_block* block = (crit_block*)new char[len];

		// populate block
		block->tag 	       = t;
		block->comp.offset = offset_member(m);
		block->comp.size   = sizeof(A);
		*(A*)(char*)(&block->comp.val) = val; // cast through char* to prevent aliasing errors
	
		// create criteria container and return
		criteria<T> ret;
		ret.len = len;
		ret.crit = (char*)block;
		return ret;
	}

	template <typename... Cs, typename T = typename std::common_type<Cs...>::type>
	T crit_combine(crit_tag t, Cs... crits) {
		static_assert(sizeof...(Cs) > 0, "cannot have empty combiner");
		char* sub_datas[] = {crits.crit...};
		size_t sub_lens[] = {crits.len...};
		size_t sub_idxs[sizeof...(Cs)];
		size_t len = sizeof(crit_block);

		// calculate indexes and total length
		for(int i = 0; i < sizeof...(Cs); i++) {
			sub_idxs[i] = i == 0 ? 0 : sub_idxs[i - 1] + sub_lens[i - 1];
			len += sub_lens[i];
		}

		// allocate block
		crit_block* block = (crit_block*)new char[len];

		// populate block
		block->tag = t;
		block->comb.size = sizeof...(Cs);
		for(int i = 0; i < sizeof...(Cs); i++) {
			memcpy(&block->comb.sub_blocks[sub_idxs[i]], sub_datas[i], sub_lens[i]);
			delete[] sub_datas[i];
		}

		// create, fill and return criteria
		T ret;
		ret.len = len;
		ret.crit = (char*)block;

		return ret;
	}
};

/**
 *	Compare criteria
 */
template <typename T, typename A>
criteria<T> LT(A T::* m, A val) {
	return _impl::crit_compare(LT_TAG, m, val);
}

template <typename T, typename A>
criteria<T> LE(A T::* m, A val) {
	return _impl::crit_compare(LE_TAG, m, val);
}

template <typename T, typename A>
criteria<T> EQ(A T::* m, A val) {
	return _impl::crit_compare(EQ_TAG, m, val);
}

template <typename T, typename A>
criteria<T> GE(A T::* m, A val) {
	return _impl::crit_compare(GE_TAG, m, val);
}

template <typename T, typename A>
criteria<T> GT(A T::* m, A val) {
	return _impl::crit_compare(GT_TAG, m, val);
}

/**
 *	Combine criteria
 */
template <typename... Cs>
typename std::common_type<Cs...>::type OR(Cs... crits) {
	return _impl::crit_combine(OR_TAG, crits...);
}

template <typename... Cs>
typename std::common_type<Cs...>::type AND(Cs... crits) {
	return _impl::crit_combine(AND_TAG, crits...);
}


#endif
