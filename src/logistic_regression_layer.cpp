#include "logistic_regression_layer.h"
#include "util.h"

void logistic_regression_layer::calc_errorterm(matrix<float> null1, matrix<float> null2) {
	if (acti_func == SIGM) {
		scalar_matrix<float> m(output.size1(), output.size2());
		delta = element_prod(-error, element_prod(output, m - output));
	}
	else if (acti_func == SOFTMAX) {
		delta = -error;
	}
}