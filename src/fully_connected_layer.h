#ifndef SRC_FULLY_CONNECTED_LAYER_H_
#define SRC_FULLY_CONNECTED_LAYER_H_

#include "util.h"
#include "macro.h"

namespace ub = boost::numeric::ublas;

namespace dl {
class LayerBase;
class FullyConnectedLayer : public LayerBase {
 public:
  FullyConnectedLayer() = default;
  FullyConnectedLayer(
    int in_dim,
    int out_dim, acti acti) :
    LayerBase(in_dim, out_dim, acti) {}

  void CalcErrorterm(const ub::matrix<float> &delta_up, const ub::matrix<float> &W_up) override {
    ub::matrix<float> deri_act;
    ub::scalar_matrix<float> m(output.size1(), output.size2());
    if (acti_func == SIGM) {
      deri_act = ub::element_prod(output, m - output);
    }
    else if (acti_func == TANH) {
      deri_act = m - ub::element_prod(output, output);
    }
    delta = ub::element_prod(prod(delta_up, trans(W_up)), deri_act);
  }
 private:
  DISALLOW_COPY_AND_ASSIGN(FullyConnectedLayer);
};
}  // namespace dl

#endif  // SRC_FULLY_CONNECTED_LAYER_H_
