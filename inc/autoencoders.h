
#pragma once

#include "layer.h"
#include "util.h"
#include "fully_connected_layer.h"
#include "logistic_regression_layer.h"

class autoencoders : public fully_connected_layer {
public:
	autoencoders() = default;
	autoencoders(matrix<float> data_set, int in_dim, int out_dim, bool isdenoising, acti acti = SIGM) : fully_connected_layer(in_dim, out_dim, acti) {
		auto_train(data_set, isdenoising);
	}

	void auto_train(matrix<float> data_set, bool isdenoising) {
		fully_connected_layer rr(data_set.size2(), 100, SIGM);
		logistic_regression_layer ll(100, data_set.size2(), SOFTMAX);
		add(&rr);
		add(&ll);
		train(data_set, data_set, isdenoising);
		W = (*layer_vec.begin())->get_W();
		B = (*layer_vec.begin())->get_B();
	}
	
};