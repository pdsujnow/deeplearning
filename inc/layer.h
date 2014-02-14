#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <vector>

using namespace boost::numeric::ublas;

class layer_base {
public:
	layer_base() = default;
	layer_base(int in_dim, int out_dim, int act) : out_dim_(out_dim), in_dim_(in_dim),output_func(act){ init_para(); };
	void init_para();
	void add(layer_base *layer);
	void train(matrix<float> data_set, matrix<float> labels);
	void forward_prop(matrix<float> input, matrix<float> label);
	void back_prop();
	void calc_activation();
	void calc_dpara(matrix<float> o);
	void recv_data(matrix<float> data_set, matrix<float> label);

	matrix<float> get_delta();
	matrix<float> get_output();
	matrix<float> calc_error();
	matrix<float> get_W();
	matrix<float> get_input();
	
	virtual void calc_errorterm(matrix<float> d, matrix<float> w);
protected:
	int out_dim_, in_dim_;
	int acti_func;

	matrix<float> W;
	matrix<float> error;
	matrix<float> output;
	matrix<float> input;
	matrix<float> input_l;
	matrix<float> delta;
	matrix<float> dW;
	matrix<float> mW;
	matrix<float> B;
	matrix<float> dB;
	matrix<float> mB;

	float learning_rate;
	float momentum;
private:
	std::vector<layer_base *> layer_vec;
	double lossfunc;
	int output_func;
};
