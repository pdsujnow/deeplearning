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
  void add(LayerBase *layer);
  void train(matrix<float> data_set,
             matrix<float> labels,
             bool isdenoising = false);

  void forward_Prop(matrix<float> input, matrix<float> label);
  void back_Prop();
  void CalcActivation();
  void CalcDpara(matrix<float> o);

  void recv_data(matrix<float> data_set, matrix<float> label) {
    input = data_set;
    input_l = label;
  }
  matrix<float> calc_error() {
    error = input_l - output;
    return error;
  }

  matrix<float> get_delta() { return delta; }
  matrix<float> get_output() { return output; }
  matrix<float> get_W() { return W; }
  matrix<float> get_B() { return B; }
  matrix<float> get_input() { return input; }

  virtual void CalcErrorterm(matrix<float> d, matrix<float> w) {}

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
  matrix<float> dB;
  matrix<float> mB;

  float learning_rate;
  float momentum;

  std::vector<LayerBase *> layer_vec;

 private:
  double lossfunc;
  int output_func;

  DISALLOW_COPY_AND_ASSIGN(LayerBase);
};
}  // namespace dl

#endif  // SRC_LAYER_H_
