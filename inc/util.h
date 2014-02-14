#pragma once
#include <math.h> 

enum acti { SIGM, TANH, SOFTMAX };



template<typename P1>
P1* reverse_endian(P1* p) {
	std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p)+sizeof(P1));
	return p;
}

template <typename Iter, typename P2>
void uniform_rand(Iter begin, Iter end, P2 gap){
	float u;
	for (Iter it = begin; it != end; ++it){
		u = (P2)rand();
		*it = (P2)(u / RAND_MAX - 0.5) * 2 * gap;
		//std::cout << *it << std::endl;
	}
}

template <typename P1 >
void sigmoid_func(P1 x) {
	*x = 1 / (1 + exp(-(*x)));
}

template <typename P1 >
void tanh_func(P1 x) {
	*x = (exp(*x) - exp(-(*x))) / (exp(*x) + exp(-(*x)));
}

template <typename P1, typename P2>
void softmax_func(P1 x, P2 sum) {
	if (sum != 0)
		*x = *x / sum;
}
