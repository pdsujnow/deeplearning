#ifndef SRC_EXCEPTION_H_
#define SRC_EXCEPTION_H_

#include <boost/exception/exception.hpp>

#include "macro.h"

namespace  expt {
enum serverity { warning, error };
}

namespace dl {
class PException : virtual public boost::exception {
 public:
  PException() = default;
  PException(const std::string& msg, expt::serverity s);
  ~PException() {}

  virtual const char* What() const throw();

 private:
  std::string msg;

  DISALLOW_COPY_AND_ASSIGN(PException);
};
}  // namespace dl

#endif  // SRC_EXCEPTION_H_
