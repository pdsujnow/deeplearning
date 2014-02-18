#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <fstream>
#include <memory>

#include "layer.h"
#include "fully_connected_layer.h"
#include "logistic_regression_layer.h"
#include "autoencoders.h"
#include "util.h"
#include "exception.h"
using namespace boost::numeric::ublas;

matrix<float> parse_mnist_labels(const std::string& label_file) {
	char label;
	int i;
	unsigned int max_value = 0;
	std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);

	if (ifs.bad() || ifs.fail())
		p_exception("failed to open file:" + label_file, expt::error);

	int magic_number, num_items;

	ifs.read((char*)&magic_number, 4);
	ifs.read((char*)&num_items, 4);

	reverse_endian(&magic_number);
	reverse_endian(&num_items);
	vector<unsigned int> label_rev(num_items);
	if (magic_number != 0x00000801 || num_items <= 0)
		p_exception("MNIST label-file format error", expt::error);

	for (i = 0; i < num_items; i++) {
		ifs.read((char*)&label, 1);
		label_rev(i) = label;
		if (label_rev(i)>max_value)
			max_value = label_rev(i);
	}
	matrix<float> labels(zero_matrix<float>(num_items, max_value + 1));
	for (i = 0; i < num_items; i++) {
		labels(i, label_rev(i)) = 1;
	}
	ifs.close();
	return labels;
}

matrix<float> parse_mnist_images(const std::string& image_file) {
	int i, j;
	std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

	int magic_number, num_items, num_rows, num_cols;

	ifs.read((char*)&magic_number, 4);
	ifs.read((char*)&num_items, 4);
	ifs.read((char*)&num_rows, 4);
	ifs.read((char*)&num_cols, 4);

	reverse_endian(&magic_number);
	reverse_endian(&num_items);
	reverse_endian(&num_rows);
	reverse_endian(&num_cols);
	int feature = num_cols*num_rows;

	std::shared_ptr<char> r_image(new char[feature]);
	matrix<float> images(num_items, feature);
	auto image_vec = r_image.get();
	for (i = 0; i < num_items; i++) {

		ifs.read((char*)&image_vec[0], feature);

		for (j = 0; j < feature; j++) {
			images(i, j) = image_vec[j];
		}
	}
	ifs.close();
	return images;
}

matrix<float> normal(matrix<float> m) {
	unsigned int i;
	float mean;
	float variance;
	scalar_vector<float> v((m).size1());
	for (i = 0; i < (m).size2(); i++) {
		matrix_column<matrix<float> > mc(m, i);
		mean = sum(mc) / (m).size2();
		mc = mc - mean*v;
		variance = pow(sum(element_prod(mc, mc)) / (m).size2(), 2);
		mc = mc / variance;
	}
	return m;
}

void normalize(matrix<float> *m) {
	float mean;
	float variance;
	scalar_vector<float> v((*m).size1());
	for (unsigned int i = 0; i < (*m).size2(); i++) {
		matrix_column<matrix<float> > mc(*m, i);
		mean = sum(mc) / (*m).size2();
		mc = mc - mean*v;
		variance = pow(sum(element_prod(mc, mc)) / (*m).size2(), 2);
		if (variance == 0) {
			continue;
		}
		mc = mc / variance;
	}
}

int main() {

	matrix<float> train_labels, testfc_labels;
	matrix<float> train_images, test_images;

	train_labels = parse_mnist_labels("t10k-labels-idx1-ubyte");
	train_images = parse_mnist_images("t10k-images-idx3-ubyte");
	layer_base lbase;
	autoencoders rr(train_images, 784, 100, SIGM);
	logistic_regression_layer ll(100, train_labels.size2(), SOFTMAX);
	lbase.add(&rr);
	lbase.add(&ll);
	lbase.train(train_images, train_labels);
}