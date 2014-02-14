#include "layer.h"
//#include "fully_connected_layer.h"
//#include "logistic_regression_layer.h"
#include "util.h"

//using namespace cnn;

void layer_base::init_para() {
	momentum = 0.5;
	learning_rate = 2;
	W.resize(in_dim_, out_dim_);
	auto it = W.begin1();
	auto it_end = W.end1();
	for (; it != it_end; ++it) {
		auto it_r = it.begin();
		auto it_r_end = it.end();
		uniform_rand(it_r, it_r_end, 4 * sqrt(6 / ((float)in_dim_ + (float)out_dim_)));
	}
}

void layer_base::calc_activation() {
	matrix<float> wx = prod(input, W);
	if (B.size1() == 0){
		zero_matrix<float> rB(input.size1(), out_dim_);
		B = rB;
	}
	wx = wx - B;
	auto it = wx.begin1();
	auto it_end = wx.end1();
	for (; it != it_end; ++it) {
		float sum_row = 0;
		auto it_r = it.begin();
		auto it_r_end = it.end();

		for (auto iter = it_r; iter != it_r_end; iter++) {
			if (acti_func == SIGM)
				sigmoid_func(iter);
			else if (acti_func == TANH)
				tanh_func(iter);
			else if (acti_func == SOFTMAX) {
				*iter = exp(*iter);
				sum_row += *iter;
			}
		}
		if (acti_func == SOFTMAX)
		for (auto iter = it_r; iter != it_r_end; iter++)
			softmax_func(iter, sum_row);
	}
	output = wx;
}

void layer_base::calc_errorterm(matrix<float> d, matrix<float> w) {

}

void layer_base::calc_dpara(matrix<float> output_down) {
	dW = learning_rate * prod(trans(delta), output_down) / delta.size1();
	dB = learning_rate * trans(delta) / delta.size1();

	if (mW.size1() == 0) {
		zero_matrix<float> m(dW.size1(), dW.size2());
		mW = m;
	}

	if (mB.size1() == 0) {
		zero_matrix<float> m(dB.size1(), dB.size2());
		mB = m;
	}

	mW = momentum * mW + dW;
	mB = momentum * mB + dB;
	dW = mW;
	dB = mB;
	W = W - trans(dW);
	B = B - trans(dB);
}



matrix<float> layer_base::get_delta() {
	return delta;
}

matrix<float> layer_base::get_input() {
	return input;
}

void layer_base::recv_data(matrix<float> data_set, matrix<float> label) {
	input = data_set;
	input_l = label;
}

matrix<float> layer_base::get_W() {
	return W;
}

matrix<float> layer_base::calc_error() {
	error = input_l - output;
	return error;
}

matrix<float> layer_base::get_output() {
	return output;
}

void layer_base::add(layer_base *layer) {
	layer_vec.push_back(layer);
}

void layer_base::forward_prop(matrix<float> input, matrix<float> label) {
	float sum = 0;
	auto it = layer_vec.begin();
	auto it_end = layer_vec.end();
	(*it)->recv_data(input, label);
	for (; it != it_end; ++it) {
		(*it)->calc_activation();
		if ((it + 1) != it_end)
			(*(it + 1))->recv_data((*it)->get_output(), label);
		else
			error = (*it)->calc_error();
	}
	error = element_prod(error, error);

	auto it1 = error.begin1();
	auto it1_end = error.end1();
	for (; it1 != it1_end; ++it1) {
		auto it_r = it1.begin();
		auto it_r_end = it1.end();
		for (; it_r != it_r_end; it_r++) {
			sum += *it_r;
		}
	}
	lossfunc = 0.5 * sum / error.size1();
}

void layer_base::back_prop() {

	auto it = layer_vec.begin();
	auto it_end = layer_vec.end();

	for (auto iter = it_end-1; it != iter+1; iter--) {

		if (iter == it_end-1) 
			(*iter)->calc_errorterm(error, error);
		else 
			(*iter)->calc_errorterm((*(iter + 1))->get_delta(), (*(iter + 1))->get_W());

		if (iter == it)	{
			(*iter)->calc_dpara((*iter)->get_input());
			break;
		}
		else
			(*iter)->calc_dpara((*(iter - 1))->get_output());
	}
}

void layer_base::train(matrix<float> data_set, matrix<float> labels) {
	
	int i, j;
	int batch_size = 100;
	int n_epochs = 100;
	int batch_index = data_set.size1() / batch_size;
	for (i = 0; i < n_epochs; i++) {
		for (j = 0; j < batch_index; j++) {
			//printf("loss function:%f", lossfunc);
			matrix_range<matrix<float> > train_labels_p(labels,
				range(j*batch_size, (j + 1) * batch_size),
				range(0, labels.size2()));
			matrix_range<matrix<float> > train_images_p(data_set,
				range(j*batch_size, (j + 1) * batch_size),
				range(0, data_set.size2()));
			forward_prop(train_images_p, train_labels_p);
			back_prop();
			printf("loss function:%lf\n", lossfunc);
		}
	}

}