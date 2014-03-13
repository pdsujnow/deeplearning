#ifndef SRC_AUTOENCODERS_H_
#define SRC_AUTOENCODERS_H_

#include "layer.h"
#include "util.h"
#include "fully_connected_layer.h"
#include "logistic_regression_layer.h"
#include "macro.h"

namespace dl {
class Autoencoders : public FullyConnectedLayer {
 public:
  Autoencoders() = default;
  Autoencoders(matrix<float> data_set,
                int in_dim,
                int out_dim,
                bool isdenoising, acti acti = SIGM) :
                FullyConnectedLayer(in_dim, out_dim, acti) {
    AutoTrain(data_set, isdenoising);
  }

  void AutoTrain(matrix<float> data_set, bool isdenoising) {
    FullyConnectedLayer rr(data_set.size2(), 100, SIGM);
    LogisticRegressionLayer ll(100, data_set.size2(), SOFTMAX);
    add(&rr);
    add(&ll);
    train(data_set, data_set, isdenoising);
    W = (*layer_vec.begin())->get_W();
    B = (*layer_vec.begin())->get_B();
  }
 private:
  DISALLOW_COPY_AND_ASSIGN(Autoencoders);
};
}  // namespace dl

#endif  // SRC_AUTOENCODERS_H_
