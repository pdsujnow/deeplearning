#include <iostream>

#include "layer.h"
#include "util.h"

namespace dl {
void LayerBase::InitPara() {
  momentum = 0.5;
  learning_rate = 2;
  InitWeight(W, in_dim_, out_dim_);
}

void LayerBase::CalcActivation() {
  matrix<float> wx = prod(input, W);
  if (B.size1() == 0) {
    InitWeight(B, 1, out_dim_);
  }
  wx = wx + prod(scalar_matrix<float>(wx.size1(), 1), B);
  auto it = wx.begin1();
  auto it_end = wx.end1();
  for (; it != it_end; ++it) {
    float sum_row = 0;
    auto it_r = it.begin();
    auto it_r_end = it.end();

    for (auto iter = it_r; iter != it_r_end; iter++) {
      if (acti_func == SIGM)
        SigmoidFunc(iter);
      else if (acti_func == TANH)
        TanhFunc(iter);
      else if (acti_func == SOFTMAX) {
        *iter = exp(*iter);
        sum_row += *iter;
      }
    }
    if (acti_func == SOFTMAX)
    for (auto iter = it_r; iter != it_r_end; iter++)
      SoftmaxFunc(iter, sum_row);
  }
  output = wx;
}

void LayerBase::CalcDpara(const matrix<float> &output_down) {
  dW = learning_rate * prod(trans(delta), output_down) / delta.size1();
  if (mW.size1() == 0) {
    ResetMatrix(mW, dW.size1(), dW.size2());
  }
  mW = momentum * mW + dW;
  dW = mW;
  W = W - trans(dW);
}

void LayerBase::add(std::shared_ptr<LayerBase> layer) {
  layer_vec.push_back(layer);
}

void LayerBase::ForwardProp(const matrix<float> &input, const matrix<float> &label) {
  float sum = 0;
  matrix<float> op;
  auto it = layer_vec.begin();
  auto it_end = layer_vec.end();
  (*it)->recv_data(input, label);
  for (; it != it_end; ++it) {
    (*it)->CalcActivation();
    if ((it + 1) != it_end)
      (*(it + 1))->recv_data((*it)->get_output(), label);
    else{
      error = (*it)->calc_error();
      output = (*it)->get_output();
    }
  }
  error = element_prod(error, error);

  scalar_vector<float> svl(error.size1());
  scalar_vector<float> svr(error.size2());
  sum = inner_prod(prod(svl, error), svr);

  lossfunc = 0.5 * sum / error.size1();
}

void LayerBase::BackProp() {
  auto it = layer_vec.begin();
  auto it_end = layer_vec.end();

  for (auto iter = it_end - 1; it != iter + 1; --iter) {
    if (iter == it_end - 1)
      (*iter)->CalcErrorterm(error, error);
    else
      (*iter)->CalcErrorterm((*(iter + 1))->get_delta(),
                             (*(iter + 1))->get_W());
    if (iter == it)  {
      (*iter)->CalcDpara((*iter)->get_input());
      break;
    }
    else
      (*iter)->CalcDpara((*(iter - 1))->get_output());
  }
}

void LayerBase::test(const matrix<float> &data_set,
                     const matrix<float> &labels) {
  ForwardProp(data_set, labels);
  int numitems = output.size1();
  float error=0.0;
  for (int i = 0; i < numitems; ++i) {
    if ((output(i, 0) - output(i, 1))*(labels(i, 0) - labels(i, 1)) < 0) {
      ++error;
    }
  }
  float r = error / numitems;
}

void LayerBase::train(matrix<float> data_set,
                      matrix<float> labels,
                      bool isdenoising) {
  int i, j;
  float corruption_level = 0.2f;
  batch_size = data_set.size1()/10;
  n_epochs = 100;

  batch_index = data_set.size1() / batch_size;
  for (i = 0; i < n_epochs; i++) {
    matrix_shuffle(data_set,labels);
    for (j = 0; j < batch_index; j++) {
      matrix_range<matrix<float> > train_labels_p(labels,
        range(j*batch_size, (j + 1) * batch_size),
        range(0, labels.size2()));
      matrix_range<matrix<float> > train_images_p(data_set,
        range(j*batch_size, (j + 1) * batch_size),
        range(0, data_set.size2()));
      if (isdenoising == true) {
        CorruptedMatrix(train_images_p, corruption_level);
      }
      ForwardProp(train_images_p, train_labels_p);
      BackProp();
      printf("loss function:%lf\n", lossfunc);
    }
  }
}

}  // namespace dl
