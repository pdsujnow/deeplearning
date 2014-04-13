#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <vector>

#include "macro.h"

using namespace boost::numeric::ublas;

namespace dl {
class LayerBase {
 public:
  LayerBase() = default;
  LayerBase(int in_dim, int out_dim, int act) :
    out_dim_(out_dim), in_dim_(in_dim), output_func(act) { InitPara(); }
  void InitPara();
  void add(std::shared_ptr<LayerBase> layer);
  void train(matrix<float> data_set,
             matrix<float> labels,
             bool isdenoising = false);
  void test(const matrix<float> &data_set,
            const matrix<float> &labels);
  void ForwardProp(const matrix<float> &input, const matrix<float> &label);
  void BackProp();
  void CalcActivation();
  void CalcDpara(const matrix<float> &o);

  void recv_data(const matrix<float> &data_set, const matrix<float> &label) {
    input = data_set;
    input_l = label;
  }
  matrix<float> calc_error() {
    error = input_l - output;
    return error;
  }

  const matrix<float> &get_delta() const { return delta; }
  const matrix<float> &get_output() const { return output; }
  const matrix<float> &get_W() const { return W; }
  const matrix<float> &get_B() const { return B; }
  const matrix<float> &get_input() const { return input; }

  virtual void CalcErrorterm(const matrix<float> &d, const matrix<float> &w) {}

 protected:
  int out_dim_, in_dim_;
  int acti_func;
  int batch_size;
  int n_epochs;
  int batch_index;

  matrix<float> W;
  matrix<float> error;
  matrix<float> output;
  matrix<float> input;
  matrix<float> input_l;
  matrix<float> delta;
  matrix<float> dW;
  matrix<float> mW;
  matrix<float> B;

  float learning_rate;
  float momentum;

  std::vector<std::shared_ptr<LayerBase>> layer_vec;

 private:
  double lossfunc;
  int output_func;

  DISALLOW_COPY_AND_ASSIGN(LayerBase);
};
}  // namespace dl

#endif  // SRC_LAYER_H_
