#pragma once

#include "layer.h"
#include "util.h"

class logistic_regression_layer :public layer_base {
public:
	logistic_regression_layer() = default;
	logistic_regression_layer(int in_dim, int out_dim, acti acti) :layer_base(in_dim, out_dim, acti) {}

	void calc_errorterm(matrix<float> d, matrix<float> w) override {
		if (acti_func == SIGM) {
			scalar_matrix<float> m(output.size1(), output.size2());
			delta = element_prod(-error, element_prod(output, m - output));
		}
		else if (acti_func == SOFTMAX) {
			delta = -error;
		}
	}
};