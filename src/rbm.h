#ifndef SRC_RBM_H_
#define SRC_RBM_H_

#include "layer.h"
#include "util.h"
#include "fully_connected_layer.h"
#include "logistic_regression_layer.h"
#include "macro.h"

namespace dl {
class Rbm : public FullyConnectedLayer {
 public:
	Rbm() = default;
	Rbm(matrix<float> data_set, int in_dim, int out_dim, acti acti) :
    FullyConnectedLayer(in_dim, out_dim, acti) {
		InitPara(in_dim, out_dim);
    RbmTrain(data_set);
	}
	void InitPara(int, int);
  void RbmTrain(matrix<float> data_set);

	void Gibbs(matrix<float> m);
	void Prop(matrix<float> m);

 protected:
	double err;
	matrix<float> weight;
	matrix<float> visbiases;
	matrix<float> hidbiases;
	matrix<float> hid;
	matrix<float> negdata;
	matrix<float> neghid;
	matrix<float> weightinc;
	matrix<float> visbiasinc;
	matrix<float> hidbiasinc;

 private:
  DISALLOW_COPY_AND_ASSIGN(Rbm);
};
}    // namespace dl

#endif  // SRC_RBM_H_
