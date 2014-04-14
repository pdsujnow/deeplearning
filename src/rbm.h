#ifndef SRC_RBM_H_
#define SRC_RBM_H_

#include "layer.h"
#include "util.h"
#include "fully_connected_layer.h"

#include "macro.h"

namespace ub = boost::numeric::ublas;

namespace dl {
class Rbm : public FullyConnectedLayer {
 public:
	Rbm() = default;
	Rbm(ub::matrix<float> &data_set, int in_dim, int out_dim, acti acti) :
    FullyConnectedLayer(in_dim, out_dim, acti) {
		InitPara(in_dim, out_dim);
    RbmTrain(data_set);
	}
	void InitPara(int, int);
  void RbmTrain(ub::matrix<float> &data_set);

	void Gibbs(const ub::matrix<float> &m);
	void Prop(const ub::matrix<float> &m);

 protected:
	double err;
	ub::matrix<float> weight;
	ub::matrix<float> visbiases;
	ub::matrix<float> hidbiases;
	ub::matrix<float> hid;
	ub::matrix<float> negdata;
	ub::matrix<float> neghid;
	ub::matrix<float> weightinc;
	ub::matrix<float> visbiasinc;
	ub::matrix<float> hidbiasinc;

 private:
  DISALLOW_COPY_AND_ASSIGN(Rbm);
};
}    // namespace dl

#endif  // SRC_RBM_H_
