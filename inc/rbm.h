
#pragma once

#include "layer.h"
#include "util.h"
#include "fully_connected_layer.h"
#include "logistic_regression_layer.h"

class rbm : public fully_connected_layer {
public:
	rbm() = default;
	rbm(matrix<float> data_set, int in_dim, int out_dim, acti acti) : fully_connected_layer(in_dim, out_dim, acti) {
		init_para(in_dim, out_dim);
		rbm_train(data_set);
	}
	void init_para(int,int);
	void rbm_train(matrix<float> data_set);

	void gibbs(matrix<float> m);
	void prop(matrix<float> m);
protected:
	double err;
	matrix<float> weight;
	matrix<float> visbiases;
	matrix<float> hidbiases;
	matrix<float> hid;
	matrix<float> negdata;
	matrix<float> neghid;
	matrix<float> weightinc;
	matrix<float> visbiasinc;
	matrix<float> hidbiasinc;
};