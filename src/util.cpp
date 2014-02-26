#include "util.h"


void matrix_act(matrix<float> *m, bool isclamp) {
	auto it = m->begin1();
	auto it_end = m->end1();
	for (; it != it_end; ++it) {
		auto it_r = it.begin();
		auto it_r_end = it.end();
		for (auto iter = it_r; iter != it_r_end; iter++) {
			sigmoid_func(iter);
			if (isclamp) {
				if (*iter > rand() / RAND_MAX)
					*iter = 1;
				else
					*iter = 0;
			}
		}
	}

}

void reset_matrix(matrix<float> *m, int in_dim_, int out_dim_) {
	zero_matrix<float> zm(in_dim_, out_dim_);
	*m = zm;
}


void init_weight(matrix<float> *m, int in_dim_, int out_dim_) {
	m->resize(in_dim_, out_dim_);
	auto it = m->begin1();
	auto it_end = m->end1();
	for (; it != it_end; ++it) {
		auto it_r = it.begin();
		auto it_r_end = it.end();
		uniform_rand(it_r, it_r_end, 4 * sqrt(6 / ((float)in_dim_ + (float)out_dim_)));
	}
}

matrix<float> corrupted_matrix(matrix<float> m, float corruption_level) {
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
	return m;
}