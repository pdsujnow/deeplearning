
#pragma once

#include "layer.h"
#include "util.h"

class fully_connected_layer : public layer_base {
public:
	fully_connected_layer() = default;
	fully_connected_layer(int in_dim, int out_dim, acti acti) : layer_base(in_dim, out_dim, acti) {}

	void calc_errorterm(matrix<float> delta_up, matrix<float> W_up) override {
		matrix<float> deri_act;
		scalar_matrix<float> m(output.size1(), output.size2());
		if (acti_func == SIGM) {
			deri_act = element_prod(output, m - output);
		}
		else if (acti_func == TANH) {
			deri_act = m - element_prod(output, output);
		}
		delta = element_prod(prod(delta_up, trans(W_up)), deri_act);
	}
};