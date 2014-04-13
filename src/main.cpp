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
#include "Rbm.h"

namespace ub = boost::numeric::ublas;
using namespace boost::numeric::ublas;
namespace dl {

matrix<float> Normalize(matrix<float> m) {
  float mean;
  float svariance;
  scalar_vector<float> v(m.size1());
  for (unsigned int i = 0; i < m.size2(); i++) {
    matrix_column<matrix<float> > mc(m, i);
    mean = sum(mc) / mc.size();
    mc = mc - mean*v;
    svariance = sqrt(sum(element_prod(mc, mc)) / (mc.size() - 1));
    if (svariance == 0) {
      continue;
    }
    mc = mc / svariance;
  }
  return m;
}

}  // namespace dl

matrix<float> ReadData(const std::string &filename,matrix<float> &data_set) {
  std::ifstream fin(filename.c_str());
  int num_items;
  int feature;
  int classes;
  fin >> num_items;
  fin >> feature;
  fin >> classes;
  data_set.resize(num_items, feature + 1);
  for (unsigned int i = 0; i < data_set.size1(); ++i) {
    for (unsigned int j = 0; j < data_set.size2(); ++j) {
      fin >> data_set(i, j);
    }
  }
  matrix_column<matrix<float>> l(data_set, 0);
  matrix<float> labels(zero_matrix<float>(num_items, classes));
  for (int k = 0; k < num_items; ++k) {
    labels(k, static_cast<unsigned>(l(k))) = 1;
  }
  fin.close();
  return labels;
}

int main() {
  using namespace dl;
  typedef std::shared_ptr<LayerBase> LayerPtr;
  std::string filename = "test_data.txt";
  matrix<float> data;
  matrix<float> label = ReadData(filename, data);
  matrix_range<matrix<float>> data_x(data, range(0, data.size1()), range(1, data.size2()));
  data_x = Normalize(data_x);

  //Rbm rb(train_images, 784, 100, SIGM);
  //Autoencoders rr(train_images, 784, 100, denoising);
  LayerPtr lbase = LayerPtr(new LayerBase());
  LayerPtr fl = LayerPtr(new FullyConnectedLayer(data_x.size2(), 2, SIGM));
  LayerPtr rl = LayerPtr(new LogisticRegressionLayer(2, 2, SIGM));
  lbase->add(fl);
  lbase->add(rl);
  lbase->train(data_x, label);
  lbase->test(data_x, label);
  return 0;
}
