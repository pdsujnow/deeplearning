#ifndef SRC_AUTOENCODERS_H_
#define SRC_AUTOENCODERS_H_

#include "util.h"

#include "macro.h"

namespace ub = boost::numeric::ublas;

namespace dl {
class FullyConnectedLayer;

class Autoencoders : public FullyConnectedLayer {
 public:
  Autoencoders() = default;
  Autoencoders(const ub::matrix<float> &data_set,
                int in_dim,
                int out_dim,
                bool isdenoising, acti acti = SIGM) :
                FullyConnectedLayer(in_dim, out_dim, acti) {
    AutoTrain(data_set, isdenoising);
  }

  void AutoTrain(const ub::matrix<float> &data_set, bool isdenoising) {
    /*LayerPtr fl = LayerPtr(new FullyConnectedLayer(data_x.size2(), 2, SIGM));
    LayerPtr rl = LayerPtr(new LogisticRegressionLayer(2, 2, SIGM);
    add(rr);
    add(ll);
    train(data_set, data_set, isdenoising);
    W = (*layer_vec.begin())->get_W();
    B = (*layer_vec.begin())->get_B();*/
  }
 private:
  DISALLOW_COPY_AND_ASSIGN(Autoencoders);
};
}  // namespace dl

#endif  // SRC_AUTOENCODERS_H_
