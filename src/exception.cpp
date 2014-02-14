
#include <boost/exception/all.hpp>

#include "exception.h"

p_exception::p_exception(const std::string& msg, expt::serverity s) {
	switch (s) {
	case expt::warning:
		std::cout << "warning: "<< msg << std::endl;
		break;
	case expt::error :
		std::cout << "error: " << msg << std::endl;
		throw;
	default:
		break;
	}
}

const char* p_exception::what() const throw()
{
	return boost::diagnostic_information_what(*this);
}
