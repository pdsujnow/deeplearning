#include "fully_connected_layer.h"
#include "util.h"

void fully_connected_layer::calc_errorterm(matrix<float> delta_up, matrix<float> W_up) {
	matrix<float> deri_act;
	scalar_matrix<float> m(output.size1(), output.size2());
	if (acti_func == SIGM) {
		deri_act = element_prod(output, m - output);
	}
	else if (acti_func == TANH) {
		deri_act = m - element_prod(output, output);
	}
	delta = element_prod(prod(delta_up, trans(W_up) ), deri_act);

}