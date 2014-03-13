
#include <boost/exception/all.hpp>

#include "exception.h"
namespace dl {
PException::PException(const std::string& msg, expt::serverity s) {
  switch (s) {
  case expt::warning:
    std::cout << "warning: " << msg << std::endl;
    break;
  case expt::error:
    std::cout << "error: " << msg << std::endl;
    throw;
  default:
    assert(false);
    break;
  }
}

const char* PException::What() const throw() {
  return boost::diagnostic_information_what(*this);
}
}  // namespace dl
