#include "rbm.h"
#include "util.h"

void rbm::init_para(int in_dim, int out_dim) {
	init_weight(&weight, in_dim, out_dim);
	reset_matrix(&weightinc, in_dim, out_dim);
}

void rbm::rbm_train(matrix<float> data_set) {
	int i, j;
	batch_size = 100;
	n_epochs = 100;
	batch_index = data_set.size1() / batch_size;

	for (i = 0; i < n_epochs; i++) {
		err = 0;
		for (j = 0; j < batch_index; j++) {
			matrix_range<matrix<float> > train_labels_p(data_set,
				range(j*batch_size, (j + 1) * batch_size),
				range(0, data_set.size2()));
			matrix_range<matrix<float> > train_images_p(data_set,
				range(j*batch_size, (j + 1) * batch_size),
				range(0, data_set.size2()));
			prop(train_images_p);
			gibbs(train_images_p);
			
		}
	}
}

void rbm::prop(matrix<float> data) {
	reset_matrix(&visbiasinc, batch_size, in_dim_);
	reset_matrix(&hidbiasinc, batch_size, out_dim_);
	hid = prod(data, weight);
	if (hidbiases.size1() == 0) {
		reset_matrix(&hidbiases, batch_size, out_dim_);
	}
	hid = hid - hidbiases;
	matrix_act(&hid, true);

	negdata = prod(hid, trans(weight));
	if (visbiases.size1() == 0) {
		reset_matrix(&visbiases, batch_size, in_dim_);
	}
	negdata = negdata - visbiases;
	matrix_act(&negdata, false);

	neghid = prod(negdata, weight);
	neghid = neghid - hidbiases;
	matrix_act(&neghid, false);	
}

void rbm::gibbs(matrix<float> data) {
	
	scalar_matrix<float> sm(batch_size, batch_size);
	scalar_vector<float> svl(data.size1());
	scalar_vector<float> svr(data.size2());
	weightinc = momentum*weightinc + learning_rate*(prod(trans(data), hid) - prod(trans(negdata), neghid)) / batch_size;
	visbiasinc = momentum*visbiasinc + learning_rate*(prod(sm, data - negdata)) / batch_size;
	hidbiasinc = momentum*hidbiasinc + learning_rate*(prod(sm, hid - neghid)) / batch_size;

	weight = weight + weightinc;
	visbiases = visbiases + visbiasinc;
	hidbiases = hidbiases + hidbiasinc;
	
	err = err + inner_prod(prod(svl, element_prod(data - negdata, data - negdata)), svr) / batch_size;
	std::cout << err << std::endl;
}