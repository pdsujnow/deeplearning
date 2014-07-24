#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <vector>

#include "macro.h"
namespace ub = boost::numeric::ublas;


namespace dl {
class LayerBase {
 public:
  LayerBase() = default;
  LayerBase(int in_dim, int out_dim, int act) :
    out_dim_(out_dim), in_dim_(in_dim), acti_func(act) { InitPara(); }
  virtual ~LayerBase();
  void InitPara();
  void add(std::shared_ptr<LayerBase> layer);
  void train(ub::matrix<float> data_set,
             ub::matrix<float> labels,
             bool isdenoising = false);
  void test(const ub::matrix<float> &data_set,
            const ub::matrix<float> &labels);
  void ForwardProp(const ub::matrix<float> &input, const ub::matrix<float> &label);
  void BackProp();
  void CalcActivation();
  void CalcDpara(const ub::matrix<float> &o);

  void recv_data(const ub::matrix<float> &data_set, const ub::matrix<float> &label) {
    input = data_set;
    input_l = label;
  }
  ub::matrix<float> calc_error() {
    error = input_l - output;
    return error;
  }

  const ub::matrix<float> &get_delta() const { return delta; }
  const ub::matrix<float> &get_output() const { return output; }
  const ub::matrix<float> &get_W() const { return W; }
  const ub::matrix<float> &get_B() const { return B; }
  const ub::matrix<float> &get_input() const { return input; }

  virtual void CalcErrorterm(const ub::matrix<float> &d, const ub::matrix<float> &w) {}

 protected:
  int out_dim_, in_dim_;
  int acti_func;
  int batch_size;
  int n_epochs;
  int batch_index;

  ub::matrix<float> W;
  ub::matrix<float> error;
  ub::matrix<float> output;
  ub::matrix<float> input;
  ub::matrix<float> input_l;
  ub::matrix<float> delta;
  ub::matrix<float> dW;
  ub::matrix<float> mW;
  ub::matrix<float> B;

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
