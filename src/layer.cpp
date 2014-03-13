#include "layer.h"
#include "util.h"

namespace dl {
void LayerBase::InitPara() {
  momentum = 0.5;
  learning_rate = 0.1f;
  InitWeight(&W, in_dim_, out_dim_);
}

void LayerBase::CalcActivation() {
  matrix<float> wx = prod(input, W);
  if (B.size1() == 0) {
    ResetMatrix(&B, input.size1(), out_dim_);
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

void LayerBase::CalcDpara(matrix<float> output_down) {
  dW = learning_rate * prod(trans(delta), output_down) / delta.size1();
  dB = learning_rate * trans(delta) / delta.size1();

  if (mW.size1() == 0) {
    ResetMatrix(&mW, dW.size1(), dW.size2());
  }

  if (mB.size1() == 0) {
    ResetMatrix(&mB, dB.size1(), dB.size2());
  }

  mW = momentum * mW + dW;
  mB = momentum * mB + dB;
  dW = mW;
  dB = mB;
  W = W - trans(dW);
  B = B - trans(dB);
}

void LayerBase::add(LayerBase *layer) {
  layer_vec.push_back(layer);
}

void LayerBase::forward_Prop(matrix<float> input, matrix<float> label) {
  float sum = 0;
  auto it = layer_vec.begin();
  auto it_end = layer_vec.end();
  (*it)->recv_data(input, label);
  for (; it != it_end; ++it) {
    (*it)->CalcActivation();
    if ((it + 1) != it_end)
      (*(it + 1))->recv_data((*it)->get_output(), label);
    else
      error = (*it)->calc_error();
  }
  error = element_prod(error, error);

  scalar_vector<float> svl(error.size1());
  scalar_vector<float> svr(error.size2());
  sum = inner_prod(prod(svl, error), svr);

  lossfunc = 0.5 * sum / error.size1();
}

void LayerBase::back_Prop() {
  auto it = layer_vec.begin();
  auto it_end = layer_vec.end();

  for (auto iter = it_end - 1; it != iter + 1; iter--) {
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

void LayerBase::train(matrix<float> data_set,
                      matrix<float> labels,
                      bool isdenoising) {
  int i, j;
  float corruption_level = 0.2f;
  batch_size = 100;
  n_epochs = 100;

  batch_index = data_set.size1() / batch_size;
  for (i = 0; i < n_epochs; i++) {
    for (j = 0; j < batch_index; j++) {
      matrix_range<matrix<float> > train_labels_p(labels,
        range(j*batch_size, (j + 1) * batch_size),
        range(0, labels.size2()));
      matrix_range<matrix<float> > train_images_p(data_set,
        range(j*batch_size, (j + 1) * batch_size),
        range(0, data_set.size2()));
      if (isdenoising == true) {
        train_images_p = CorruptedMatrix(train_images_p, corruption_level);
      }
      forward_Prop(train_images_p, train_labels_p);
      back_Prop();
      printf("loss function:%lf\n", lossfunc);
    }
  }
}

}  // namespace dl
