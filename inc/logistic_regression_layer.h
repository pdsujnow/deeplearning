#pragma once

#include "layer.h"


class logistic_regression_layer :public layer_base {
public:
	logistic_regression_layer() = default;
	logistic_regression_layer(int in_dim, int out_dim, acti acti) :layer_base(in_dim, out_dim, acti) {
		init_para();
	}
	void calc_errorterm(matrix<float> d, matrix<float> w) override;
};