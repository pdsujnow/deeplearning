#ifndef SRC_LOGISTIC_REGRESSION_LAYER_H_
#define SRC_LOGISTIC_REGRESSION_LAYER_H_

#include "layer.h"
#include "util.h"
#include "macro.h"

namespace dl {
class LogisticRegressionLayer :public LayerBase {
 public:
  LogisticRegressionLayer() = default;
  LogisticRegressionLayer(int in_dim, int out_dim, acti acti) :
    LayerBase(in_dim, out_dim, acti) {}

  void CalcErrorterm(matrix<float> d, matrix<float> w) override {
    if (acti_func == SIGM) {
      scalar_matrix<float> m(output.size1(), output.size2());
      delta = element_prod(-error, element_prod(output, m - output));
    }
    else if (acti_func == SOFTMAX) {
      delta = -error;
    }
  }
 private:
  DISALLOW_COPY_AND_ASSIGN(LogisticRegressionLayer);
};
}  // namespace dl

#endif  // SRC_LOGISTIC_REGRESSION_LAYER_H_
