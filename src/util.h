#ifndef SRC_UTIL_H_
#define SRC_UTIL_H_

#include <math.h>
#include <boost/numeric/ublas/matrix.hpp>

#include "macro.h"
namespace ub = boost::numeric::ublas;


namespace dl {
enum acti { SIGM, TANH, SOFTMAX };
void MatrixAct(ub::matrix<float> &m, const bool clamp);
void InitWeight(ub::matrix<float> &m, int in, int out);
void ResetMatrix(ub::matrix<float> &m, int in, int out);
void CorruptedMatrix(ub::matrix<float> m, const float corruption_level);
void matrix_shuffle(ub::matrix<float> &data_set, ub::matrix<float> &labels);

template<typename P1>
P1* ReverseEndian(P1* p) {
  std::reverse(reinterpret_cast<char*>(p),
               reinterpret_cast<char*>(p)+sizeof(P1));
  return p;
}

template <typename Iter, typename P2>
void UniformRand(Iter begin, Iter end, P2 gap) {
  float u;
  for (Iter it = begin; it != end; ++it) {
    u = (P2)rand();
    *it = (P2)(u / RAND_MAX - 0.5) * 2 * gap;
  }
}

template <typename P1 >
void SigmoidFunc(P1 x) {
  *x = 1 / (1 + exp(-(*x)));
}

template <typename P1 >
void TanhFunc(P1 x) {
  *x = (exp(*x) - exp(-(*x))) / (exp(*x) + exp(-(*x)));
}

template <typename P1, typename P2>
void SoftmaxFunc(P1 x, P2 sum) {
  if (sum != 0)
    *x = *x / sum;
}
}  // namespace dl

#endif  // SRC_UTIL_H_
