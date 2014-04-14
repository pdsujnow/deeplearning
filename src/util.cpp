#include <time.h>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "util.h"

namespace dl {
void MatrixAct(ub::matrix<float> &m, const bool isclamp) {
  auto it = m.begin1();
  auto it_end = m.end1();
  for (; it != it_end; ++it) {
    auto it_r = it.begin();
    auto it_r_end = it.end();
    for (auto iter = it_r; iter != it_r_end; iter++) {
      SigmoidFunc(iter);
      if (isclamp) {
        if (*iter > rand() / RAND_MAX)
          *iter = 1;
        else
          *iter = 0;
      }
    }
  }
}

void ResetMatrix(ub::matrix<float> &m, int in_dim_, int out_dim_) {
  ub::zero_matrix<float> zm(in_dim_, out_dim_);
  m = zm;
}


void InitWeight(ub::matrix<float> &m, int in_dim_, int out_dim_) {
  m.resize(in_dim_, out_dim_);
  auto it = m.begin1();
  auto it_end = m.end1();
  for (; it != it_end; ++it) {
    auto it_r = it.begin();
    auto it_r_end = it.end();
    UniformRand(it_r,
                it_r_end,
                4 * sqrt(6 / ((float)in_dim_ + (float)out_dim_)));
  }
}

void CorruptedMatrix(ub::matrix<float> m, const float corruption_level) {
  auto it = m.begin1();
  auto it_end = m.end1();
  for (; it != it_end; ++it) {
    auto it_r = it.begin();
    auto it_r_end = it.end();
    for (auto iter = it_r; iter != it_r_end; iter++) {
      if (corruption_level > rand() / RAND_MAX) {
        *iter = 0;
      }
    }
  }
}

void matrix_shuffle(ub::matrix<float> &data_set,ub::matrix<float> &labels) {
  int i = data_set.size1();
  int randi = 0;
  srand(static_cast<unsigned>(time(NULL)));
  for (int j = 0; j < i; ++j) {
    randi = rand() % i;
    swap(ub::matrix_row<ub::matrix<float>>(data_set, j), 
         ub::matrix_row<ub::matrix<float>>(data_set, randi));
    swap(ub::matrix_row<ub::matrix<float>>(labels, j),
         ub::matrix_row<ub::matrix<float>>(labels, randi));
  }
}

}  // namespace dl
