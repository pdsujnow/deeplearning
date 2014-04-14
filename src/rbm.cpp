#include "Rbm.h"
#include "util.h"

namespace dl {
void Rbm::InitPara(int in_dim, int out_dim) {
  InitWeight(weight, in_dim, out_dim);
  ResetMatrix(weightinc, in_dim, out_dim);
}

void Rbm::RbmTrain(ub::matrix<float> &data_set) {
  int i, j;
  batch_size = 100;
  n_epochs = 100;
  batch_index = data_set.size1() / batch_size;

  for (i = 0; i < n_epochs; i++) {
    err = 0;
    for (j = 0; j < batch_index; j++) {
      ub::matrix_range<ub::matrix<float> > train_labels_p(data_set,
        ub::range(j*batch_size, (j + 1) * batch_size),
        ub::range(0, data_set.size2()));
      ub::matrix_range<ub::matrix<float> > train_images_p(data_set,
        ub::range(j*batch_size, (j + 1) * batch_size),
        ub::range(0, data_set.size2()));
      Prop(train_images_p);
      Gibbs(train_images_p);
    }
  }
}

void Rbm::Prop(const ub::matrix<float> &data) {
  ResetMatrix(visbiasinc, batch_size, in_dim_);
  ResetMatrix(hidbiasinc, batch_size, out_dim_);
  hid = prod(data, weight);
  if (hidbiases.size1() == 0) {
    ResetMatrix(hidbiases, batch_size, out_dim_);
  }
  hid = hid - hidbiases;
  MatrixAct(hid, true);

  negdata = prod(hid, trans(weight));
  if (visbiases.size1() == 0) {
    ResetMatrix(visbiases, batch_size, in_dim_);
  }
  negdata = negdata - visbiases;
  MatrixAct(negdata, false);

  neghid = prod(negdata, weight);
  neghid = neghid - hidbiases;
  MatrixAct(neghid, false);
}

void Rbm::Gibbs(const ub::matrix<float> &data) {
  ub::scalar_matrix<float> sm(batch_size, batch_size);
  ub::scalar_vector<float> svl(data.size1());
  ub::scalar_vector<float> svr(data.size2());
  weightinc = momentum*weightinc +
              learning_rate*(prod(trans(data), hid) -
              prod(trans(negdata), neghid)) /
              batch_size;

  visbiasinc = momentum*visbiasinc +
               learning_rate*(prod(sm, data - negdata)) /
               batch_size;
  hidbiasinc = momentum*hidbiasinc +
               learning_rate*(prod(sm, hid - neghid)) /
               batch_size;

  weight = weight + weightinc;
  visbiases = visbiases + visbiasinc;
  hidbiases = hidbiases + hidbiasinc;

  err = err +
  inner_prod(prod(svl, ub::element_prod(data - negdata, data - negdata)), svr) /
  batch_size;
  std::cout << err << std::endl;
}
}  // namespace dl
