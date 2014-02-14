
#pragma once

#include "layer.h"

class fully_connected_layer : public layer_base {
public:
	fully_connected_layer() = default;
	fully_connected_layer(int in_dim, int out_dim, acti acti) : layer_base(in_dim, out_dim, acti) {
		init_para();
	}
	void calc_errorterm(matrix<float> d, matrix<float> w) override;
};